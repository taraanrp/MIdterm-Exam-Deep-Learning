"""Lokasi folder proyek. Semua path di kode lewat modul ini (tidak ada path hardcode)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("UTS_ROOT", Path(__file__).resolve().parents[2]))
DATA = Path(os.environ.get("UTS_DATA_DIR", ROOT / "data"))
KODE = ROOT / "kode"
OUT = KODE / "outputs"
RES, FIG, TAB, MOD, CACHE, LOG = (OUT / d for d in ["results", "figures", "tables", "models", "cache", "logs"])
CONFIGS = KODE / "configs"
for _d in (RES, FIG, TAB, MOD, CACHE, LOG, CONFIGS):
    _d.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("UTS_QUICK", "0") == "1"   # mode uji cepat: subset data + sedikit epoch
SEED = int(os.environ.get("UTS_SEED", "42"))
# Skala anggaran training (1.0 = setelan untuk GPU T4). Turunkan (mis. 0.5) kalau waktu GPU terbatas.
BUDGET = float(os.environ.get("UTS_BUDGET", "1.0"))


def data_file(name: str) -> Path:
    """Cari file data; menerima juga varian nama 'x_csv.gz' / 'x.csv'."""
    cands = [DATA / name, DATA / name.replace(".csv.gz", "_csv.gz"), DATA / name.replace(".csv.gz", ".csv")]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"File data '{name}' tidak ditemukan di {DATA}. Isi folder: {sorted(p.name for p in DATA.glob('*'))}")


def epochs(n: int) -> int:
    """Jumlah epoch sesuai mode: --quick → 1–2 epoch; selain itu n × BUDGET."""
    return 1 if QUICK else max(1, int(round(n * BUDGET)))
