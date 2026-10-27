from __future__ import annotations

import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from .augment import add_noise, classic_augment
from .config import CFG, CLASSES, DEVICE
from .models import build_model, model_spec

AUG_NAMES = ("none", "classic", "diffusion")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_loader(X, y, batch: int, shuffle: bool, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch, shuffle=shuffle, generator=g)


@torch.no_grad()
def predict(model, X, batch: int = 256) -> np.ndarray:
    model.eval(); out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch].to(DEVICE)).argmax(1).cpu())
    return torch.cat(out).numpy()


def metrics(y_true, y_pred) -> dict:
    return {"acc": accuracy_score(y_true, y_pred), "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "recall": recall_score(y_true, y_pred, average=None, labels=list(range(len(CLASSES)))).tolist()}


def train_classifier(cfg: CFG, model, Xtr, ytr, Xva, yva, epochs: int, lr: float, aug: str = "none", seed: int = 0, tag: str = "", log=print):
    seed_all(seed)
    model = model.to(DEVICE); model.specaug = (aug == "classic")
    loader = make_loader(Xtr, ytr, cfg.batch, True, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total = epochs * len(loader); warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    best, best_state, hist, t0 = -1.0, None, [], time.time()
    for ep in range(epochs):
        model.train(); run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            if aug == "classic":
                xb = classic_augment(xb)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step(); run += loss.item() * len(xb)
        f1v = f1_score(yva.numpy(), predict(model, Xva), average="macro")
        hist.append({"epoch": ep + 1, "loss": run / len(Xtr), "val_f1": f1v})
        if f1v > best:
            best, best_state = f1v, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if tag and ((ep + 1) % max(1, epochs // 5) == 0 or ep == 0):
            log(f"    [{tag}] época {ep+1}/{epochs} | loss {hist[-1]['loss']:.4f} | val F1 {f1v:.3f} | {time.time()-t0:.0f} s")
    model.load_state_dict(best_state)
    return model, {"best_val_f1": best, "train_sec": time.time() - t0, "history": hist}


def run_path(cfg: CFG, mname: str, aname: str, seed: int):
    return cfg.out / "runs" / f"{mname}_{aname}_s{seed}.json"


def weight_path(cfg: CFG, mname: str, aname: str, seed: int):
    return cfg.out / "weights" / f"{mname}_{aname}_s{seed}.pt"


def run_experiments(cfg: CFG, splits: dict, Xsyn, ysyn, models=("CNN1D", "CNN2D", "AST"), augs=AUG_NAMES, log=print) -> list:
    Xtr, ytr = splits["train"]; Xva, yva = splits["val"]; Xte, yte = splits["test"]
    datasets = {"none": (Xtr, ytr), "classic": (Xtr, ytr), "diffusion": (torch.cat([Xtr, Xsyn]), torch.cat([ytr, ysyn]))}
    spec = model_spec(cfg); records = []
    for mname in models:
        for aname in augs:
            for seed in cfg.seeds:
                path = run_path(cfg, mname, aname, seed)
                if path.exists():
                    records.append(json.loads(path.read_text())); log(f"caché {path.name}"); continue
                Xa, ya = datasets[aname]; epochs, lr = spec[mname]
                model, info = train_classifier(cfg, build_model(mname, cfg), Xa, ya, Xva, yva, epochs, lr, aug=aname, seed=seed, tag=f"{mname}/{aname}/s{seed}", log=log)
                yp = predict(model, Xte); mt = metrics(yte.numpy(), yp)
                f1_snr = {f"test_f1_snr{s}": f1_score(yte.numpy(), predict(model, add_noise(Xte, s)), average="macro") for s in cfg.snr_eval}
                rec = {"model": mname, "aug": aname, "seed": seed, "test_acc": mt["acc"], "test_f1_macro": mt["f1_macro"], **f1_snr,
                       **{f"recall_{c}": r for c, r in zip(CLASSES, mt["recall"])}, "best_val_f1": info["best_val_f1"],
                       "train_sec": info["train_sec"], "epochs": epochs, "n_train": len(Xa), "history": info["history"],
                       "confusion": confusion_matrix(yte.numpy(), yp, labels=list(range(len(CLASSES)))).tolist()}
                path.write_text(json.dumps(rec)); records.append(rec)
                if seed == cfg.seeds[0]:
                    torch.save(model.state_dict(), weight_path(cfg, mname, aname, seed))
                log(f"{mname:6s} {aname:9s} s{seed} | test F1 {mt['f1_macro']:.3f} ({', '.join(f'{s} dB {f1_snr[f'test_f1_snr{s}']:.3f}' for s in cfg.snr_eval)}) | val F1 {info['best_val_f1']:.3f} | {info['train_sec']:.0f} s")
                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()
    return records


def load_records(cfg: CFG) -> list:
    return [json.loads(p.read_text()) for p in sorted((cfg.out / "runs").glob("*.json"))]


def records_frame(records: list) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "confusion")} for r in records])


def load_trained(cfg: CFG, mname: str, aname: str, seed: int | None = None):
    seed = cfg.seeds[0] if seed is None else seed
    model = build_model(mname, cfg)
    model.load_state_dict(torch.load(weight_path(cfg, mname, aname, seed), map_location="cpu"))
    return model.to(DEVICE).eval()


def has_run(cfg: CFG, mname: str, aname: str) -> bool:
    return weight_path(cfg, mname, aname, cfg.seeds[0]).exists()
