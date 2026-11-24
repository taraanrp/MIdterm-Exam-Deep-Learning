"""Tugas 2: arsitektur sekuens (GRU, TCN) dan baseline non-DL (LightGBM)."""
import numpy as np
import torch
import torch.nn as nn


class GRUReg(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, drop=0.2):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, layers, batch_first=True, dropout=drop)
        self.drop = nn.Dropout(drop); self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(self.drop(h[:, -1])).squeeze(1)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, dil, drop):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(cin, cout, k, dilation=dil)
        self.act = nn.ReLU(); self.drop = nn.Dropout(drop)
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return self.drop(self.act(y)) + self.res(x)


class TCNReg(nn.Module):
    """Temporal Convolutional Network: konvolusi kausal berdilatasi (1, 2, 4, 8) → receptive field ±31 siklus."""

    def __init__(self, n_feat, ch=64, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(*[CausalConv(n_feat if i == 0 else ch, ch, 3, 2 ** i, drop) for i in range(4)])
        self.out = nn.Linear(ch, 1); self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.out(self.drop(h[:, :, -1])).squeeze(1)


ARCH = {"GRU": GRUReg, "TCN (1D-CNN dilatasi)": TCNReg}


def handcrafted(ws, rows, n_last=20):
    """Fitur baseline non-DL dari jendela: nilai terakhir, rata-rata 10 terakhir, kemiringan 20 terakhir per fitur."""
    X = ws.get(rows)                              # (B, W, F) termasuk kanal mask di akhir
    m = X[:, :, -1:]
    Xs = X[:, -n_last:, :-1]; ms = m[:, -n_last:, 0]
    last = X[:, -1, :-1]
    mean10 = (X[:, -10:, :-1] * m[:, -10:]).sum(1) / np.maximum(m[:, -10:].sum(1), 1)
    t = np.arange(n_last, dtype=np.float32)[None, :]
    tw = np.where(ms > 0, t, np.nan); tm = np.nanmean(tw, 1, keepdims=True)
    num = np.nansum((tw - tm)[:, :, None] * np.where(ms[:, :, None] > 0, Xs, np.nan), 1)
    den = np.nansum((tw - tm) ** 2, 1)[:, None] + 1e-6
    slope = num / den
    return np.concatenate([last, mean10, np.nan_to_num(slope)], 1)
