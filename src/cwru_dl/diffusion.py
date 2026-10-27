from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG, CLASSES, DEVICE
from .train import make_loader, seed_all


class SinusoidalEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t.float()[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], 1)


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv1d(cin, cout, 3, 1, 1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv1d(cout, cout, 3, 1, 1)
        self.film = nn.Linear(emb, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        g, b = self.film(e)[:, :, None].chunk(2, 1)
        h = self.n2(h) * (1 + g) + b
        return self.c2(F.silu(h)) + self.skip(x)


class UNet1D(nn.Module):
    def __init__(self, n_cls: int = len(CLASSES), ch=(32, 64, 128, 256), emb: int = 128):
        super().__init__()
        self.t_emb = nn.Sequential(SinusoidalEmb(emb), nn.Linear(emb, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.y_emb = nn.Embedding(n_cls + 1, emb)
        self.inp = nn.Conv1d(1, ch[0], 7, 1, 3)
        self.down, self.pool = nn.ModuleList(), nn.ModuleList()
        for a, b in zip(ch[:-1], ch[1:]):
            self.down.append(ResBlock1D(a, a, emb)); self.pool.append(nn.Conv1d(a, b, 4, 4))
        self.mid1, self.mid2 = ResBlock1D(ch[-1], ch[-1], emb), ResBlock1D(ch[-1], ch[-1], emb)
        self.up, self.unpool = nn.ModuleList(), nn.ModuleList()
        for a, b in zip(ch[::-1][:-1], ch[::-1][1:]):
            self.unpool.append(nn.ConvTranspose1d(a, b, 4, 4)); self.up.append(ResBlock1D(2 * b, b, emb))
        self.out = nn.Sequential(nn.GroupNorm(8, ch[0]), nn.SiLU(), nn.Conv1d(ch[0], 1, 3, 1, 1))

    def forward(self, x, t, y):
        e = self.t_emb(t) + self.y_emb(y)
        h = self.inp(x); skips = []
        for blk, pool in zip(self.down, self.pool):
            h = blk(h, e); skips.append(h); h = pool(h)
        h = self.mid2(self.mid1(h, e), e)
        for blk, unpool in zip(self.up, self.unpool):
            h = unpool(h); h = blk(torch.cat([h, skips.pop()], 1), e)
        return self.out(h)


class Diffusion:
    def __init__(self, T: int, device):
        self.T = T
        beta = torch.linspace(1e-4, 0.02, T, device=device)
        self.abar = torch.cumprod(1 - beta, 0)

    def q_sample(self, x0, t, eps):
        a = self.abar[t][:, None, None]
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    @torch.no_grad()
    def ddim_sample(self, model, y, steps: int, w: float, null_id: int, L: int):
        model.eval(); B = len(y)
        x = torch.randn(B, 1, L, device=y.device)
        ts = torch.linspace(self.T - 1, 0, steps, device=y.device).long()
        yn = torch.full_like(y, null_id)
        for i, t in enumerate(ts):
            tb = torch.full((B,), int(t), device=y.device)
            eps = (1 + w) * model(x, tb, y) - w * model(x, tb, yn)
            a_t = self.abar[t]; a_prev = self.abar[ts[i + 1]] if i + 1 < steps else torch.tensor(1.0, device=y.device)
            x0 = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-6, 6)
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x.squeeze(1)


def train_diffusion(cfg: CFG, Xtr, ytr, seed: int = 0, log=print):
    seed_all(seed)
    net = UNet1D(len(CLASSES)).to(DEVICE); diff = Diffusion(cfg.diff_T, DEVICE)
    loader = make_loader(Xtr, ytr, 16 if cfg.smoke else cfg.diff_batch, True, seed)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.diff_epochs * len(loader))
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    hist, t0, step = [], time.time(), 0
    for ep in range(cfg.diff_epochs):
        net.train(); run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE).unsqueeze(1), yb.to(DEVICE)
            yb = torch.where(torch.rand(len(yb), device=DEVICE) < 0.1, torch.full_like(yb, len(CLASSES)), yb)
            t = torch.randint(0, cfg.diff_T, (len(xb),), device=DEVICE); eps = torch.randn_like(xb)
            loss = F.mse_loss(net(diff.q_sample(xb, t, eps), t, yb), eps)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); step += 1
            decay = min(0.999, (1 + step) / (10 + step))
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(decay).add_(v.detach(), alpha=1 - decay)
            run += loss.item() * len(xb)
        hist.append(run / len(Xtr))
        if (ep + 1) % max(1, cfg.diff_epochs // 8) == 0:
            log(f"  difusión época {ep+1}/{cfg.diff_epochs} | mse {hist[-1]:.4f} | {time.time()-t0:.0f} s")
    net.load_state_dict(ema)
    return net, diff, {"history": hist, "train_sec": time.time() - t0}


def synthesize_balance(cfg: CFG, net, diff, ytr, log=print):
    counts = np.bincount(ytr.numpy(), minlength=len(CLASSES)); target = counts.max()
    Xsyn, ysyn, t0 = [], [], time.time()
    for ci, c in enumerate(CLASSES):
        need = int(target - counts[ci]) if not cfg.syn_cap else int(min(target - counts[ci], cfg.syn_cap))
        if need <= 0:
            continue
        gen = []
        for i in range(0, need, 256):
            yb = torch.full((min(256, need - i),), ci, device=DEVICE)
            gen.append(diff.ddim_sample(net, yb, cfg.ddim_steps, cfg.cfg_w, len(CLASSES), cfg.win).cpu())
        g = torch.cat(gen); g = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-8)
        Xsyn.append(g); ysyn.append(torch.full((len(g),), ci))
        log(f"  {c}: {need} ventanas sintéticas")
    return torch.cat(Xsyn), torch.cat(ysyn), time.time() - t0


def run_diffusion_stage(cfg: CFG, splits: dict, log=print):
    syn_path, w_path, info_path = cfg.out / "synthetic" / "cwru_synthetic.pt", cfg.out / "weights" / "ddpm_unet1d.pt", cfg.out / "synthetic" / "ddpm_info.pt"
    if syn_path.exists() and info_path.exists():
        pack = torch.load(syn_path); info = torch.load(info_path)
        log(f"Sintéticas en caché: {tuple(pack['X'].shape)}")
        return pack["X"], pack["y"], info
    Xtr, ytr = splits["train"]
    net, diff, info = train_diffusion(cfg, Xtr, ytr, log=log)
    Xsyn, ysyn, info["sample_sec"] = synthesize_balance(cfg, net, diff, ytr, log=log)
    torch.save({"X": Xsyn, "y": ysyn}, syn_path); torch.save(net.state_dict(), w_path); torch.save(info, info_path)
    return Xsyn, ysyn, info
