import torch


def classic_augment(x: torch.Tensor) -> torch.Tensor:
    B, L = x.shape
    snr_db = torch.empty(B, 1, device=x.device).uniform_(10, 30)
    noise = torch.randn_like(x) * (x.std(dim=1, keepdim=True) / (10 ** (snr_db / 20)))
    shift = torch.randint(0, L, (B, 1), device=x.device)
    idx = (torch.arange(L, device=x.device)[None, :] + shift) % L
    x = torch.gather(x, 1, idx)
    return (x + noise) * torch.empty(B, 1, device=x.device).uniform_(0.8, 1.2)


def spec_augment(s: torch.Tensor, fmask: int, tmask: int, n: int = 2) -> torch.Tensor:
    B, _, Fd, Td = s.shape
    for _ in range(n):
        f0 = torch.randint(0, Fd - fmask, (B,), device=s.device); fw = torch.randint(0, fmask, (B,), device=s.device)
        t0 = torch.randint(0, Td - tmask, (B,), device=s.device); tw = torch.randint(0, tmask, (B,), device=s.device)
        fi = torch.arange(Fd, device=s.device)[None, :]; ti = torch.arange(Td, device=s.device)[None, :]
        fm = ((fi >= f0[:, None]) & (fi < (f0 + fw)[:, None])).float()[:, None, :, None]
        tm = ((ti >= t0[:, None]) & (ti < (t0 + tw)[:, None])).float()[:, None, None, :]
        s = s * (1 - fm) * (1 - tm)
    return s


def add_noise(X: torch.Tensor, snr_db: float, seed: int = 123) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(X.shape, generator=g) * (X.std(dim=1, keepdim=True) / (10 ** (snr_db / 20)))
    return X + noise
