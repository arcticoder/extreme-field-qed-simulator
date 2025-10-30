"""
Helper functions to turn Δn into observable phase and ellipticity for a probe.

We consider a linearly polarized probe at angle theta relative to the external-field principal axis.
The two orthogonal polarization components accumulate different phases leading to ellipticity.
"""
from __future__ import annotations
import numpy as np
from .heisenberg_euler import birefringent_indices


def phase_retardation(E0: float = 0.0, B0: float = 0.0, L: float = 1.0, lambda0: float = 532e-9) -> float:
    """Return differential phase δ = 2π L (n_parallel - n_perp) / λ for a path length L and probe wavelength λ.
    If a cavity is used with power build-up G, multiply the returned δ by G.
    """
    dn_par, dn_perp = birefringent_indices(E=E0, B=B0)
    delta_n = dn_par - dn_perp
    delta_phi = 2.0 * np.pi * L * delta_n / lambda0
    return float(delta_phi)


def induced_ellipticity(theta_deg: float, delta_phi: float) -> float:
    """Approximate induced ellipticity ψ ≈ (δ/2) sin(2θ) for small δ.
    theta_deg: initial linear polarization angle in degrees relative to fast axis.
    """
    theta = np.deg2rad(theta_deg)
    return 0.5 * delta_phi * np.sin(2.0 * theta)
