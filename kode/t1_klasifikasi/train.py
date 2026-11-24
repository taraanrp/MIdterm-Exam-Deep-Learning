"""Tugas 1: training loop generik (AMP di GPU, modality dropout, augmentasi, early stopping pada macro-F1 terbobot)."""
import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import weighted_f1
from .models import MODELS
from ..common.device import DEVICE, USE_AMP
from ..common.paths import SEED, epochs as n_epochs
from ..common.seed import set_seed

DEFAULT = dict(model="concat (early/feature-level)", epochs=18, lr=2e-3, wd=1e-4, batch=256, label_smoothing=0.0,
               norm="global", aug_intensitas=False, bobot_lini_B=1.0, p_drop_sensor=0.15, p_drop_kanal=0.10,
               class_weight=True, seed=SEED)


class Arrays:
    """Semua array siap pakai (sudah dinormalisasi) untuk satu konfigurasi normalisasi."""

    def __init__(self, img, sen, line, norm, y=None):
        img = np.ascontiguousarray(img)
        if not img.flags.writeable:                                          # array dari cache mmap → salin ke RAM
            img = img.copy()
        self.img = torch.from_numpy(img)                                     # uint8, dinormalisasi per batch
        m, s = norm.img_stats(line)
        self.im_mean, self.im_std = torch.from_numpy(m), torch.from_numpy(s)
        x, miss = norm.sensor(sen)
        self.sen, self.miss = torch.from_numpy(x), torch.from_numpy(miss)
        self.line = torch.from_numpy(line)
        self.y = None if y is None else torch.from_numpy(y)

    def batch(self, idx, sensor_mode="asli"):
        idx = torch.as_tensor(idx)
        x = (self.img[idx].float() / 255.0 - self.im_mean[idx, None, None]) / self.im_std[idx, None, None]
        s, miss = self.sen[idx].clone(), self.miss[idx].clone()
        if sensor_mode == "hilang":
            s.zero_(); miss.fill_(1.0)
        elif sensor_mode == "satu_kanal_hilang":
            g = torch.Generator().manual_seed(0)
            ch = torch.randint(0, 3, (len(idx),), generator=g)
            s[torch.arange(len(idx)), ch] = 0.0; miss[torch.arange(len(idx)), ch] = 1.0
        return x[:, None].to(DEVICE, non_blocking=True), s.to(DEVICE, non_blocking=True), miss.to(DEVICE, non_blocking=True)


def _augment(x, s, miss, cfg, gen):
    """Augmentasi di GPU: flip/rot90 (cacat tidak bergantung orientasi), opsional jitter intensitas (domain shift),
    dan modality dropout (melatih model tetap berfungsi saat sensor tidak terekam)."""
    B = x.shape[0]
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1, generator=gen).item() < 0.5:
        x = x.flip(2)
    k = int(torch.randint(0, 4, (1,), generator=gen).item())
    x = torch.rot90(x, k, (2, 3))
    if cfg["aug_intensitas"]:
        a = (1 + 0.25 * (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1)).to(x.device)     # kontras
        b = (0.5 * (torch.rand(B, 1, 1, 1, generator=gen) * 2 - 1)).to(x.device)           # kecerahan (dalam satuan std)
        x = x * a + b
    if cfg["p_drop_sensor"] > 0:
        d = (torch.rand(B, generator=gen) < cfg["p_drop_sensor"]).to(x.device)
        s = s.masked_fill(d[:, None, None], 0.0); miss = torch.where(d[:, None], torch.ones_like(miss), miss)
    if cfg["p_drop_kanal"] > 0:
        d = (torch.rand(B, generator=gen) < cfg["p_drop_kanal"]).to(x.device)
        ch = torch.randint(0, 3, (B,), generator=gen).to(x.device)
        onehot = F.one_hot(ch, 3).bool() & d[:, None]
        s = s.masked_fill(onehot[:, :, None], 0.0); miss = torch.where(onehot, torch.ones_like(miss), miss)
    return x, s, miss


@torch.no_grad()
def predict(model, arr, idx, sensor_mode="asli", batch=1024):
    model.eval(); out = []
    for i in range(0, len(idx), batch):
        x, s, m = arr.batch(idx[i:i + batch], sensor_mode)
        with torch.autocast(DEVICE.type, enabled=USE_AMP):
            out.append(torch.softmax(model(x, s, m).float(), 1).cpu())
    return torch.cat(out).numpy()


def fit(arr, idx_tr, idx_va, cfg=None, y_override=None, keep_history=True, log=True):
    """Latih satu model. Mengembalikan (model terbaik, riwayat per epoch, metrik validasi terbaik)."""
    cfg = {**DEFAULT, **(cfg or {})}
    set_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    model = MODELS[cfg["model"]]().to(DEVICE)
    y_all = arr.y if y_override is None else torch.as_tensor(y_override)
    ytr = y_all[idx_tr]
    cw = None
    if cfg["class_weight"]:
        cnt = torch.bincount(ytr, minlength=7).float()
        cw = (cnt.sum() / (7 * cnt.clamp(min=1))).sqrt().to(DEVICE)          # bobot kelas √(1/frekuensi)
    sw_line = torch.where(arr.line == 1, torch.tensor(cfg["bobot_lini_B"]), torch.tensor(1.0))
    E = n_epochs(cfg["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = E * math.ceil(len(idx_tr) / cfg["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    yva = arr.y[idx_va].numpy(); lva = arr.line[idx_va].numpy()
    best, best_state, hist = -1, None, []
    for ep in range(E):
        model.train(); t0 = time.time()
        perm = idx_tr[torch.randperm(len(idx_tr), generator=gen).numpy()]
        tot, n = 0.0, 0
        for i in range(0, len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            x, s, m = arr.batch(b)
            x, s, m = _augment(x, s, m, cfg, gen)
            yb = y_all[b].to(DEVICE); w = sw_line[b].to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=USE_AMP):
                logits = model(x, s, m)
                loss = (F.cross_entropy(logits.float(), yb, weight=cw, label_smoothing=cfg["label_smoothing"], reduction="none") * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(b); n += len(b)
        pv = predict(model, arr, idx_va)
        vloss = float(F.nll_loss(torch.log(torch.from_numpy(pv) + 1e-8), torch.from_numpy(yva)).item())
        met = weighted_f1(yva, pv.argmax(1), lva)
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_loss": vloss, **met, "detik": time.time() - t0})
        if log:
            print(f"  [{cfg['model']}] ep {ep+1}/{E} loss {tot/n:.4f} val_loss {vloss:.4f} F1 {met['f1_macro']:.4f} "
                  f"F1w {met['f1_terbobot_uji']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if met["f1_terbobot_uji"] > best:
            best = met["f1_terbobot_uji"]; best_state = copy.deepcopy(model.state_dict()); best_ep = ep + 1
    model.load_state_dict(best_state)
    pv = predict(model, arr, idx_va)
    res = {**weighted_f1(yva, pv.argmax(1), lva), "epoch_terbaik": best_ep, "epoch_total": E}
    return model, hist, res, pv


def overfit_one_batch(arr, idx, model_name, steps=150, bs=64):
    """Sanity check: model harus bisa menghafal satu batch kecil (loss → ~0, akurasi → 100%)."""
    set_seed(SEED)
    model = MODELS[model_name]().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    b = idx[:bs]; x, s, m = arr.batch(b); y = arr.y[b].to(DEVICE)
    model.train(); log = []
    for t in range(steps):
        out = model(x, s, m); loss = F.cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 10 == 0 or t == steps - 1:
            log.append({"step": t, "loss": float(loss.item()), "akurasi": float((out.argmax(1) == y).float().mean().item())})
    return log
