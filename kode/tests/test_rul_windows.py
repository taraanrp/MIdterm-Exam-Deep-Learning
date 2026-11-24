"""Uji anti-kebocoran jendela RUL: jendela yang berakhir di siklus t hanya berisi siklus ≤ t dari unit yang sama."""
import numpy as np
import pandas as pd
from kode.t2_rul.data import WindowSet


def _toy():
    rows = []
    for u, L in [(1, 5), (2, 8)]:
        for c in range(1, L + 1):
            rows.append({"id_unit": u, "siklus": c, "rul": L - c, "f": u * 100 + c})
    return pd.DataFrame(rows)


def test_window_causal_and_padded():
    df = _toy(); ws = WindowSet(df[["f"]], df, W=4)
    r = np.flatnonzero((ws.unit_of == 2) & (ws.cycle == 3))
    win = ws.get(r)[0]                       # (4, 2): fitur + mask
    assert list(win[:, 0]) == [0, 201, 202, 203]       # padding kiri lalu siklus 1..3 unit 2 (bukan unit 1, bukan masa depan)
    assert list(win[:, 1]) == [0, 1, 1, 1]
    last = ws.last_rows()
    assert list(ws.cycle[last]) == [5, 8]
