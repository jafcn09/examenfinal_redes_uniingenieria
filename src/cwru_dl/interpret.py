from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import hilbert
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score

from .config import CFG, CLASSES, DEVICE
from .data import fault_freqs
from .models import CNN1D
from .train import load_trained, predict, train_classifier


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, x, cls):
        self.model.eval(); x = x.to(DEVICE)
        out = self.model(x); self.model.zero_grad(set_to_none=True)
        out[torch.arange(len(x)), cls].sum().backward()
        w = self.grad.mean(dim=tuple(range(2, self.act.dim())), keepdim=True)
        cam = F.relu((w * self.act).sum(1))
        flat = cam.flatten(1)
        shape = (-1, *([1] * (cam.dim() - 1)))
        cam = cam - flat.min(1)[0].view(shape)
        return (cam / (cam.flatten(1).max(1)[0].view(shape) + 1e-8)).detach().cpu()


def dominant_freq(profile, fs_p: float, fmin: float = 80, fmax: float = 185) -> float:
    p = (profile - profile.mean()) * np.hanning(len(profile))
    spec = np.abs(np.fft.rfft(p, n=8192)); f = np.fft.rfftfreq(8192, 1 / fs_p)
    m = (f >= fmin) & (f <= fmax)
    return float(f[m][spec[m].argmax()])


def matches(f: float, target: float, tol: float = 0.12) -> bool:
    return abs(f - target) <= tol * target


def expected_freqs(rpm3: float) -> dict:
    ff = fault_freqs(rpm3)
    return {**{c: ("BPFI", ff["BPFI"]) for c in CLASSES if c.startswith("IR")}, **{c: ("BPFO", ff["BPFO"]) for c in CLASSES if c.startswith("OR")}}


def cam_frequency_table(cfg: CFG, cam_model, Xte, yte, rpm3: float, n: int = 64) -> pd.DataFrame:
    cam2d = GradCAM(cam_model, cam_model.cam_layer)
    rows = []
    for c, (fname, ftarget) in expected_freqs(rpm3).items():
        ci = CLASSES.index(c); idx = torch.nonzero(yte == ci).squeeze(1)[:n]
        cam = cam2d(Xte[idx], torch.full((len(idx),), ci, device=DEVICE)).numpy()
        fs_cam = cam.shape[-1] / (cfg.win / cfg.fs)
        f_cam = [dominant_freq(cam[k].sum(0), fs_cam) for k in range(len(idx))]
        fprof = cam.sum(2).mean(0); fbins = (np.arange(len(fprof)) + 0.5) * (cfg.fs / 2) / len(fprof)
        f_env = [dominant_freq(np.abs(hilbert(Xte[i].numpy())), cfg.fs) for i in idx]
        rows.append({"clase": c, "frecuencia teórica": f"{fname} = {ftarget:.1f} Hz", "n": len(idx),
                     "CAM: centroide espectral (Hz)": float((fprof * fbins).sum() / (fprof.sum() + 1e-8)),
                     "CAM: f dominante mediana (Hz)": float(np.median(f_cam)), "CAM: % coincide (±12 %)": 100 * np.mean([matches(f, ftarget) for f in f_cam]),
                     "Envolvente: f dominante mediana (Hz)": float(np.median(f_env)), "Envolvente: % coincide": 100 * np.mean([matches(f, ftarget) for f in f_env])})
    return pd.DataFrame(rows).round(1)


@torch.no_grad()
def attention_rollout(ast_model, x) -> np.ndarray:
    ast_model.eval(); _, atts = ast_model(x.to(DEVICE), output_attentions=True)
    R = torch.eye(atts[0].shape[-1], device=DEVICE)[None].repeat(len(x), 1, 1)
    for A in atts:
        A = A.mean(1) + torch.eye(A.shape[-1], device=DEVICE)[None]
        R = (A / A.sum(-1, keepdim=True)) @ R
    f_p, t_p = ast_model.grid
    return R[:, 0, 2:].reshape(len(x), f_p, t_p).cpu().numpy()


def mel_patch_labels(cfg: CFG, f_p: int) -> list:
    mel_hz = torchaudio.functional.melscale_fbanks(257, 20.0, cfg.fs / 2, 128, cfg.fs).argmax(0).numpy() * cfg.fs / 512
    return [f"{mel_hz[min(127, k * 10 + 8)]:.0f} Hz" for k in range(f_p)]


def ast_band_table(cfg: CFG, ast_model, Xte, yte, n: int = 32) -> pd.DataFrame:
    return pd.DataFrame({c: attention_rollout(ast_model, Xte[torch.nonzero(yte == CLASSES.index(c)).squeeze(1)[:n]]).mean((0, 2)) for c in CLASSES},
                        index=mel_patch_labels(cfg, ast_model.grid[0])).round(4)


@torch.no_grad()
def tsne_real_vs_synthetic(cfg: CFG, Xtr, ytr, Xsyn, ysyn, per_class: int = 120):
    ref = load_trained(cfg, "CNN1D", "none")
    sel_r = torch.cat([torch.nonzero(ytr == c).squeeze(1)[:per_class] for c in range(len(CLASSES))])
    sel_s = torch.cat([torch.nonzero(ysyn == c).squeeze(1)[:per_class] for c in range(len(CLASSES)) if (ysyn == c).any()])
    E = torch.cat([ref.embed(Xtr[sel_r].to(DEVICE)), ref.embed(Xsyn[sel_s].to(DEVICE))]).cpu().numpy()
    lab = np.concatenate([ytr[sel_r].numpy(), ysyn[sel_s].numpy()]); src = np.array(["real"] * len(sel_r) + ["sintético"] * len(sel_s))
    Z = TSNE(n_components=2, perplexity=min(30, max(5, len(E) // 10)), random_state=0, init="pca").fit_transform(E)
    return Z, lab, src


def tstr(cfg: CFG, Xsyn, ysyn, Xva, yva, Xte, yte, log=print) -> pd.DataFrame:
    counts = np.bincount(ysyn.numpy(), minlength=len(CLASSES))
    n_per = int(min(counts[counts > 0].min(), 800))
    sel = torch.cat([torch.nonzero(ysyn == c).squeeze(1)[:n_per] for c in range(len(CLASSES)) if (ysyn == c).any()])
    present = sorted(set(ysyn[sel].tolist())); mask = np.isin(yte.numpy(), present)
    rows = []
    for seed in cfg.seeds:
        m, _ = train_classifier(cfg, CNN1D(), Xsyn[sel], ysyn[sel], Xva, yva, cfg.epochs_cnn, cfg.lr_cnn, seed=seed, log=log)
        yp = predict(m, Xte)
        rows.append({"seed": seed, "clases sintéticas": len(present), "F1_macro TSTR (clases sintéticas)": f1_score(yte.numpy()[mask], yp[mask], average="macro", labels=present),
                     "acc TSTR (clases sintéticas)": accuracy_score(yte.numpy()[mask], yp[mask])})
    return pd.DataFrame(rows)
