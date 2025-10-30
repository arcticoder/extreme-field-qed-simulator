"""
Schwinger pair production rate in a constant electric field (Sauter–Schwinger effect).

Rate density (pairs per unit volume per unit time) for uniform field E:
    w(E) = (e^2 E^2)/(4π^3 ħ^2 c) * sum_{n=1..∞} (1/n^2) exp(-n * π * E_s / E)
For E << E_s, the n=1 term dominates and the rate is astronomically small.

This module provides a truncated n=1 approximation and an optional series sum.
"""
from __future__ import annotations
import numpy as np
from .constants import e_charge, hbar, c, E_s


_def_prefactor = (e_charge**2) / (4.0 * np.pi**3 * hbar**2 * c)


def schwinger_rate(E0: float, n_terms: int = 1) -> float:
    """Return pair production rate density w(E) [m^-3 s^-1].
    E0: electric field amplitude [V/m].
    n_terms: number of terms to include in the series (n=1 is usually sufficient for E << E_s).
    """
    if E0 <= 0:
        return 0.0
    rates = [(_def_prefactor * E0**2) * (1.0 / (n**2)) * np.exp(-n * np.pi * (E_s / E0)) for n in range(1, n_terms + 1)]
    return float(np.sum(rates))


def expected_pairs(E0: float, volume: float, duration: float, n_terms: int = 1) -> float:
    """Return expected number of e+e- pairs for a uniform field over volume [m^3] and duration [s].
    """
    w = schwinger_rate(E0, n_terms=n_terms)
    return float(w * volume * duration)
