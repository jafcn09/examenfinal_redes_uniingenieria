from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import yaml

CLASSES = ["Normal", "B007", "B014", "B021", "IR007", "IR014", "IR021", "OR007", "OR014", "OR021"]
FOLDERS = {
    "Normal": "Normal",
    "B007": "12k_Drive_End_Bearing_Fault_Data/B/007",
    "B014": "12k_Drive_End_Bearing_Fault_Data/B/014",
    "B021": "12k_Drive_End_Bearing_Fault_Data/B/021",
    "IR007": "12k_Drive_End_Bearing_Fault_Data/IR/007",
    "IR014": "12k_Drive_End_Bearing_Fault_Data/IR/014",
    "IR021": "12k_Drive_End_Bearing_Fault_Data/IR/021",
    "OR007": "12k_Drive_End_Bearing_Fault_Data/OR/007/@6",
    "OR014": "12k_Drive_End_Bearing_Fault_Data/OR/014",
    "OR021": "12k_Drive_End_Bearing_Fault_Data/OR/021/@6",
}
BEARING = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 2.3568, "FTF": 0.39828}
RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
DATA_URL = "https://github.com/s-whynot/CWRU-dataset.git"
DATA_DIRS = ["Normal", "12k_Drive_End_Bearing_Fault_Data"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HW = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else f"CPU x{os.cpu_count()}"


@dataclass
class CFG:
    root: Path = Path("data/CWRU")
    out: Path = Path("results")
    fs: int = 12000
    win: int = 2048
    train_stride: int = 512
    eval_stride: int = 2048
    train_loads: tuple = (0, 1, 2)
    test_loads: tuple = (3,)
    val_frac: float = 0.2
    imbalance: dict = field(default_factory=lambda: {"007": 0.3, "014": 0.1, "021": 0.03})
    seeds: tuple = (0, 1, 2)
    batch: int = 64
    epochs_cnn: int = 30
    epochs_ast: int = 8
    lr_cnn: float = 1e-3
    lr_ast: float = 3e-5
    ast_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    ast_frames: int = 96
    diff_epochs: int = 60
    diff_batch: int = 64
    diff_T: int = 1000
    ddim_steps: int = 50
    cfg_w: float = 2.0
    syn_cap: int = 0
    snr_eval: tuple = (10, 0)
    profile: str = "full"

    @property
    def smoke(self) -> bool:
        return self.profile == "smoke"

    def dirs(self) -> "CFG":
        for d in ["runs", "figures", "tables", "weights", "synthetic"]:
            (self.out / d).mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}


PROFILES = {
    "smoke": dict(seeds=(0,), epochs_cnn=2, epochs_ast=1, diff_epochs=2, ddim_steps=5, batch=32),
    "fast": dict(seeds=(0,), epochs_cnn=10, epochs_ast=2, diff_epochs=8, ddim_steps=20, syn_cap=300),
    "full": dict(),
}


def load_config(path: str | Path | None = None, profile: str | None = None, **overrides) -> CFG:
    raw = {}
    for p in ([Path("configs/base.yaml")] if path is None else [Path("configs/base.yaml"), Path(path)]):
        if p.exists():
            raw.update(yaml.safe_load(p.read_text()) or {})
    raw.update({k: v for k, v in overrides.items() if v is not None})
    profile = profile or os.environ.get("CWRU_PROFILE") or raw.pop("profile", None) or ("full" if DEVICE.type == "cuda" else "smoke")
    raw.pop("profile", None)
    for k in ("train_loads", "test_loads", "seeds", "snr_eval"):
        if k in raw:
            raw[k] = tuple(raw[k])
    for k in ("root", "out"):
        if k in raw:
            raw[k] = Path(raw[k])
    cfg = CFG(**raw)
    cfg.profile = profile
    for k, v in PROFILES[profile].items():
        setattr(cfg, k, v)
    if os.environ.get("CWRU_SYN_CAP"):
        cfg.syn_cap = int(os.environ["CWRU_SYN_CAP"])
    return cfg.dirs()
