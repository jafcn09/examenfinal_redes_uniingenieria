from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.signal import hilbert, welch

from .config import CFG, CLASSES, FOLDERS, BEARING, RPM_BY_LOAD, DATA_URL, DATA_DIRS


def download_cwru(root: Path) -> None:
    if all((root / d).exists() for d in DATA_DIRS):
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--no-checkout", DATA_URL, str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "sparse-checkout", "set", *DATA_DIRS], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "-q"], check=True)


def load_signal(cfg: CFG, cls: str, load: int):
    folder = cfg.root / FOLDERS[cls]
    path = next(p for p in folder.glob(f"*_{load}.mat"))
    m = sio.loadmat(path)
    de_key = next(k for k in m if k.endswith("DE_time"))
    rpm_key = next((k for k in m if k.endswith("RPM")), None)
    rpm = float(m[rpm_key].squeeze()) if rpm_key else RPM_BY_LOAD[load]
    return m[de_key].astype(np.float32).ravel(), rpm, path.name


def load_meta(cfg: CFG) -> pd.DataFrame:
    rows = []
    for c in CLASSES:
        for l in (0, 1, 2, 3):
            x, rpm, name = load_signal(cfg, c, l)
            rows.append({"clase": c, "carga_HP": l, "archivo": name, "muestras": len(x), "rpm": rpm})
    return pd.DataFrame(rows)


def make_windows(x: np.ndarray, win: int, stride: int) -> np.ndarray:
    n = (len(x) - win) // stride + 1
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def zscore(w: np.ndarray) -> np.ndarray:
    return (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-8)


def class_fraction(cfg: CFG, cls: str) -> float:
    return next((v for k, v in cfg.imbalance.items() if cls.endswith(k)), 1.0)


def build_split(cfg: CFG):
    rng = np.random.default_rng(0)
    parts = {k: ([], []) for k in ("train", "val", "test")}
    for ci, c in enumerate(CLASSES):
        tr_parts, va_parts = [], []
        for l in cfg.train_loads:
            x, _, _ = load_signal(cfg, c, l)
            cut = int(len(x) * (1 - cfg.val_frac))
            tr_parts.append(make_windows(x[:cut], cfg.win, cfg.train_stride))
            va_parts.append(make_windows(x[cut:], cfg.win, cfg.eval_stride))
        tr = np.concatenate(tr_parts)
        keep = rng.choice(len(tr), size=max(8, int(len(tr) * class_fraction(cfg, c))), replace=False)
        tr = tr[np.sort(keep)]
        parts["train"][0].append(tr); parts["train"][1].append(np.full(len(tr), ci))
        va = np.concatenate(va_parts)
        parts["val"][0].append(va); parts["val"][1].append(np.full(len(va), ci))
        for l in cfg.test_loads:
            x, _, _ = load_signal(cfg, c, l)
            te = make_windows(x, cfg.win, cfg.eval_stride)
            parts["test"][0].append(te); parts["test"][1].append(np.full(len(te), ci))
    pack = lambda X, y: (torch.from_numpy(zscore(np.concatenate(X))), torch.from_numpy(np.concatenate(y)).long())
    splits = {k: pack(*v) for k, v in parts.items()}
    if cfg.smoke:
        splits = {k: smoke_subsample(cfg, *v, n) for (k, v), n in zip(splits.items(), (48, 8, 12))}
    return splits


def smoke_subsample(cfg: CFG, X, y, n: int):
    g = torch.Generator().manual_seed(0)
    idx = torch.cat([torch.nonzero(y == c).squeeze(1)[torch.randperm(int((y == c).sum()), generator=g)[:max(6, int(n * class_fraction(cfg, CLASSES[c])))]]
                     for c in range(len(CLASSES))])
    return X[idx], y[idx]


def class_counts(splits: dict) -> pd.DataFrame:
    names = {"train": "train (0-2 HP, desbalanceado)", "val": "val (0-2 HP)", "test": "test (3 HP)"}
    return pd.DataFrame({names[k]: np.bincount(y.numpy(), minlength=len(CLASSES)) for k, (_, y) in splits.items()}, index=CLASSES)


def envelope_spectrum(x: np.ndarray, fs: int):
    env = np.abs(hilbert(x - x.mean()))
    f, p = welch(env - env.mean(), fs=fs, nperseg=min(len(env), 4096))
    return f, p


def fault_freqs(rpm: float) -> dict:
    fr = rpm / 60.0
    return {k: v * fr for k, v in BEARING.items()}
