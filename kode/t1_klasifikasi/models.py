"""Tugas 1: encoder gambar (CNN), encoder sinyal (1D-CNN), dan tiga strategi fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASS = 7


def conv2(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class ImageEncoder(nn.Module):
    """CNN 4 blok: 64×64 → peta fitur 256×4×4. Keluaran: vektor (GAP, 256) dan 16 token (untuk cross-attention)."""

    def __init__(self, width=32, out=256):
        super().__init__()
        self.net = nn.Sequential(conv2(1, width), conv2(width, width * 2), conv2(width * 2, width * 4), conv2(width * 4, out))
        self.out = out

    def forward(self, x):
        f = self.net(x)                         # (B, 256, 4, 4)
        return f.mean((2, 3)), f.flatten(2).transpose(1, 2)


class SignalEncoder(nn.Module):
    """1D-CNN untuk sinyal (3 kanal × 128). Input ditambah 3 kanal mask (1 = kanal tidak terekam),
    sehingga model tahu kanal mana yang hilang. Keluaran: vektor (128) dan 16 token."""

    def __init__(self, out=128):
        super().__init__()
        def blk(cin, cout, k):
            return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                                 nn.Conv1d(cout, cout, k, padding=k // 2, bias=False), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
                                 nn.MaxPool1d(2))
        self.net = nn.Sequential(blk(6, 32, 7), blk(32, 64, 5), blk(64, out, 3))
        self.out = out

    def forward(self, s, miss):
        x = torch.cat([s, miss[:, :, None].expand(-1, -1, s.shape[2])], dim=1)
        f = self.net(x)                         # (B, 128, 16)
        return f.mean(2), f.transpose(1, 2)


def head(d_in):
    return nn.Sequential(nn.Dropout(0.3), nn.Linear(d_in, 128), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(128, N_CLASS))


class ImageOnly(nn.Module):
    def __init__(self):
        super().__init__(); self.img = ImageEncoder(); self.head = head(256)

    def forward(self, x, s, miss):
        return self.head(self.img(x)[0])


class SensorOnly(nn.Module):
    def __init__(self):
        super().__init__(); self.sig = SignalEncoder(); self.head = head(128)

    def forward(self, x, s, miss):
        return self.head(self.sig(s, miss)[0])


class ConcatFusion(nn.Module):
    """Fusion tingkat fitur (early/intermediate): vektor gambar dan sinyal digabung lalu satu kepala klasifikasi."""

    def __init__(self):
        super().__init__(); self.img = ImageEncoder(); self.sig = SignalEncoder(); self.head = head(256 + 128)

    def forward(self, x, s, miss):
        return self.head(torch.cat([self.img(x)[0], self.sig(s, miss)[0]], 1))


class LateFusion(nn.Module):
    """Late fusion: kepala terpisah per modalitas, logit digabung dengan bobot per kelas yang dipelajari.
    Jika SELURUH sensor hilang, keputusan otomatis hanya dari gambar."""

    def __init__(self):
        super().__init__()
        self.img = ImageEncoder(); self.sig = SignalEncoder()
        self.h_img, self.h_sig = head(256), head(128)
        self.gate = nn.Parameter(torch.zeros(N_CLASS))

    def forward(self, x, s, miss):
        li, ls = self.h_img(self.img(x)[0]), self.h_sig(self.sig(s, miss)[0])
        a = torch.sigmoid(self.gate)[None, :]
        all_missing = (miss.sum(1, keepdim=True) >= 3).float()
        return (1 - all_missing) * (a * li + (1 - a) * ls) + all_missing * li


class CrossAttentionFusion(nn.Module):
    """Cross-attention dua arah: token gambar meng-attend token sinyal dan sebaliknya, lalu digabung."""

    def __init__(self, d=128, heads=4):
        super().__init__()
        self.img = ImageEncoder(); self.sig = SignalEncoder()
        self.pi, self.ps = nn.Linear(256, d), nn.Linear(128, d)
        self.pos_i, self.pos_s = nn.Parameter(torch.zeros(1, 16, d)), nn.Parameter(torch.zeros(1, 16, d))
        self.att_is = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.att_si = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.1)
        self.ln_i, self.ln_s = nn.LayerNorm(d), nn.LayerNorm(d)
        self.head = head(2 * d)

    def forward(self, x, s, miss):
        ti = self.pi(self.img(x)[1]) + self.pos_i
        ts = self.ps(self.sig(s, miss)[1]) + self.pos_s
        ai = self.ln_i(ti + self.att_is(ti, ts, ts, need_weights=False)[0])
        as_ = self.ln_s(ts + self.att_si(ts, ti, ti, need_weights=False)[0])
        return self.head(torch.cat([ai.mean(1), as_.mean(1)], 1))


MODELS = {"concat (early/feature-level)": ConcatFusion, "late fusion": LateFusion, "cross-attention": CrossAttentionFusion,
          "image-only": ImageOnly, "sensor-only": SensorOnly}


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
