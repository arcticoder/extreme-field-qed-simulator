import math
import numpy as np
from efqs.heisenberg_euler import delta_n_E, delta_n_B
from efqs.pair_production import schwinger_rate
from efqs.constants import E_s, B_c


def test_delta_n_scales_quadratic_E():
    dn1 = delta_n_E(0.1 * E_s)
    dn2 = delta_n_E(0.2 * E_s)
    # Parallel component should scale by ~4 when doubling E
    ratio = dn2[0] / dn1[0]
    assert 3.9 < ratio < 4.1


def test_delta_n_zero_at_zero_field():
    dn = delta_n_B(0.0)
    assert dn[0] == 0.0 and dn[1] == 0.0


def test_schwinger_rate_tiny_below_threshold():
    w = schwinger_rate(1e-2 * E_s)
    # Astronomically small; should be far below any lab-detectable rate
    # At E = 1e-2 E_s, the n=1 term yields ~4e-85 m^-3 s^-1
    assert w < 1e-80
