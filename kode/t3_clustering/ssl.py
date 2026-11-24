"""Tugas 3 (b)–(c): representasi tanpa label.
- SimCLR gambar: augmentasi = distorsi yang ingin DIABAIKAN (rotasi, skala, kecerahan/kontras, blur, noise).
- SimCLR spektrum: augmentasi = pergeseran puncak, sisa baseline, spike, noise, skala.
- Joint (gaya CLIP): SimCLR gambar + SimCLR spektrum + kontras lintas-modal (gambar ↔ spektrum sampel yang sama)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..t1_klasifikasi.models import ImageEncoder
from ..common.device import DEVICE, USE_AMP
from ..common.paths import SEED, QUICK, epochs as n_epochs
from ..common.seed import set_seed


class SpecEncoder(nn.Module):
    def __init__(self, out=128):
        super().__init__()
        def blk(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                                 nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.net = nn.Sequential(blk(1, 32, 7), blk(32, 64, 5), blk(64, out, 3))
        self.out = out

    def forward(self, s):
        return self.net(s[:, None]).mean(2)


class ImgEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = ImageEncoder(width=32, out=256); self.out = 256

    def forward(self, x):
        return self.enc(x)[0]


def proj(d_in, d_out=64):
    return nn.Sequential(nn.Linear(d_in, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Linear(256, d_out))


# ------------------------------------------------------------------ augmentasi (di GPU)
def aug_img(x, gen):
    B = x.shape[0]; dev = x.device
    ang = torch.rand(B, generator=gen).to(dev) * 2 * math.pi
    sc = (0.75 + 0.5 * torch.rand(B, generator=gen)).to(dev)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    flip = torch.where(torch.rand(B, generator=gen) < 0.5, -1.0, 1.0).to(dev)
    theta = torch.stack([torch.stack([cos * flip, -sin, torch.zeros_like(cos)], 1), torch.stack([sin * flip, cos, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)          # rotasi bebas + skala + flip
    a = (0.7 + 0.6 * torch.rand(B, 1, 1, 1, generator=gen)).to(dev)                     # kontras
    b = (0.8 * (torch.rand(B, 1, 1, 1, generator=gen) - 0.5)).to(dev)                   # kecerahan
    x = x * a + b
    blur = (torch.rand(B, generator=gen) < 0.3).to(dev)
    xb = F.avg_pool2d(x, 3, 1, 1)
    x = torch.where(blur[:, None, None, None], xb, x)
    return x + 0.1 * torch.randn(x.shape, generator=gen).to(dev)


def aug_spec(s, gen):
    B, L = s.shape; dev = s.device
    shift = (torch.rand(B, generator=gen) * 8 - 4).to(dev)                              # geser puncak ±4 titik
    pos = torch.arange(L, device=dev)[None, :].float() - shift[:, None]
    i0 = pos.floor().clamp(0, L - 1).long(); i1 = (i0 + 1).clamp(max=L - 1); w = (pos - pos.floor()).clamp(0, 1)
    s = s.gather(1, i0) * (1 - w) + s.gather(1, i1) * w
    t = torch.linspace(-1, 1, L, device=dev)[None, :]
    c = (torch.rand(B, 3, generator=gen) - 0.5).to(dev) * torch.tensor([0.6, 0.6, 0.4], device=dev)
    s = s + c[:, :1] + c[:, 1:2] * t + c[:, 2:3] * t ** 2                                  # sisa baseline acak
    n_sp = torch.rand(B, L, generator=gen).to(dev) < 0.01
    s = s + n_sp * (2 + 3 * torch.rand(B, L, generator=gen).to(dev))                     # spike acak
    s = s * (0.8 + 0.4 * torch.rand(B, 1, generator=gen).to(dev))
    return s + 0.05 * torch.randn(s.shape, generator=gen).to(dev)


def nt_xent(z1, z2, tau=0.2):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    z = torch.cat([z1, z2]); n = z1.shape[0]
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e4)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def clip_loss(a, b, tau=0.2):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    lg = a @ b.t() / tau; t = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(lg, t) + F.cross_entropy(lg.t(), t))


def train_ssl(img=None, spec=None, mode="img", epochs=60, batch=512, lr=1e-3, seed=SEED, lam_clip=1.0, log=True):
    """mode: 'img' (SimCLR gambar), 'spec' (SimCLR spektrum), 'joint' (keduanya + kontras lintas-modal)."""
    set_seed(seed); gen = torch.Generator().manual_seed(seed)
    ei = ImgEnc().to(DEVICE) if mode in ("img", "joint") else None
    es = SpecEncoder().to(DEVICE) if mode in ("spec", "joint") else None
    pi = proj(256).to(DEVICE) if ei is not None else None
    ps = proj(128).to(DEVICE) if es is not None else None
    params = [p for m in (ei, es, pi, ps) if m is not None for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    N = len(img) if img is not None else len(spec)
    E = n_epochs(epochs); steps = E * (N // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ti = torch.from_numpy(img) if img is not None else None
    ts = torch.from_numpy(spec) if spec is not None else None
    hist = []
    for ep in range(E):
        for m in (ei, es, pi, ps):
            if m is not None: m.train()
        perm = torch.randperm(N, generator=gen); tot = 0.0; nb = 0
        for i in range(0, N - batch + 1, batch):
            b = perm[i:i + batch]
            with torch.autocast(DEVICE.type, enabled=USE_AMP):
                loss = 0.0
                if ei is not None:
                    x = ti[b].to(DEVICE)
                    h1, h2 = ei(aug_img(x, gen)), ei(aug_img(x, gen))
                    loss = loss + nt_xent(pi(h1).float(), pi(h2).float())
                if es is not None:
                    s = ts[b].to(DEVICE)
                    g1, g2 = es(aug_spec(s, gen)), es(aug_spec(s, gen))
                    loss = loss + nt_xent(ps(g1).float(), ps(g2).float())
                if mode == "joint":
                    loss = loss + lam_clip * clip_loss(pi(h1).float(), ps(g1).float())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += float(loss.item()); nb += 1
        hist.append({"epoch": ep + 1, "loss": tot / max(nb, 1)})
        if log and (ep % 5 == 0 or ep == E - 1):
            print(f"  [SSL {mode}] ep {ep+1}/{E} loss {tot/max(nb,1):.4f}", flush=True)
    return {"img": ei, "spec": es}, hist


@torch.no_grad()
def embed(enc, data, batch=2048):
    enc.eval(); out = []
    t = torch.from_numpy(data)
    for i in range(0, len(t), batch):
        with torch.autocast(DEVICE.type, enabled=USE_AMP):
            out.append(enc(t[i:i + batch].to(DEVICE)).float().cpu())
    return F.normalize(torch.cat(out), dim=1).numpy()
