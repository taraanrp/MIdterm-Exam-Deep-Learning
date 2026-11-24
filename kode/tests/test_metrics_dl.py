import numpy as np
from kode.common.metrics import asym_score, rmse, macro_f1


def test_asym_score():
    # d = -13 → e^1 − 1 ; d = +10 → e^1 − 1 ; d = 0 → 0
    assert abs(asym_score([20], [7]) - (np.e - 1)) < 1e-9
    assert abs(asym_score([20], [30]) - (np.e - 1)) < 1e-9
    assert asym_score([5, 5], [5, 5]) == 0
    # terlambat (overestimate) dihukum lebih berat daripada terlalu dini dengan selisih sama
    assert asym_score([50], [70]) > asym_score([50], [30])


def test_rmse_f1():
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    assert macro_f1([0, 1, 1], [0, 1, 1]) == 1.0
