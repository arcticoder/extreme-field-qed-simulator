"""
Vacuum birefringence via Heisenberg–Euler effective Lagrangian (low-energy, weakly dispersive).

Implements leading-order refractive index shifts for polarizations parallel/perpendicular to an external field.

References:
- Heisenberg & Euler (1936)
- Dittrich & Gies (2000)
- Battesti & Rizzo (2013) Rep. Prog. Phys. 76 016401
"""
from __future__ import annotations
import numpy as np
from .constants import A_he, B_c, E_s, c, epsilon0, mu0


def _field_invariants(E: np.ndarray | float, B: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return invariants F = (B^2 - E^2/c^2)/2 and G = E·B/c.
    Inputs accept scalars or arrays; returns arrays broadcast to common shape.
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    F = 0.5 * (B**2 - (E**2) / c**2)
    # If E and B are magnitudes (no angle), assume orthogonal by default: G=0
    G = np.zeros_like(F)
    return F, G


def delta_n_B(B: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_n_parallel, delta_n_perp) for a probe in a static magnetic field B (Tesla).

    Leading-order results:
        n_parallel - 1 = (7/2) A (B/B_c)^2
        n_perp     - 1 = 2     A (B/B_c)^2
    where A = alpha/(45*pi).
    """
    B = np.asarray(B, dtype=float)
    x2 = (B / B_c) ** 2
    dn_par = 3.5 * A_he * x2
    dn_perp = 2.0 * A_he * x2
    return dn_par, dn_perp


def delta_n_E(E: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_n_parallel, delta_n_perp) for a probe in a static electric field E (V/m).

    By duality (E ↔ cB), use E/E_s in place of B/B_c:
        n_parallel - 1 = (7/2) A (E/E_s)^2
        n_perp     - 1 = 2     A (E/E_s)^2
    """
    E = np.asarray(E, dtype=float)
    x2 = (E / E_s) ** 2
    dn_par = 3.5 * A_he * x2
    dn_perp = 2.0 * A_he * x2
    return dn_par, dn_perp


def birefringent_indices(E: float = 0.0, B: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Combined Δn from coexisting static E and B, approximated by superposition of invariants.
    For most configurations with either E or B dominant, delta_n ~ O(A*(E/E_s)^2 + A*(B/B_c)^2).
    Returns (dn_parallel, dn_perp).
    """
    dnE = delta_n_E(E)
    dnB = delta_n_B(B)
    return dnE[0] + dnB[0], dnE[1] + dnB[1]
