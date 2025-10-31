"""
Coupling efficiency metrics between EM fields and gravitational response.
"""
from __future__ import annotations
import numpy as np
from typing import Dict
from .constants import c, epsilon0, mu0


def total_energy(u: np.ndarray, dV: float) -> float:
    """Total EM energy from energy density field u and cell volume dV."""
    return float(np.sum(u) * dV)


def approximate_input_power_from_time_series(u_t: np.ndarray, dV: float, dt: float) -> float:
    """Crude estimate of average input power from dE/dt over time series u(x,t).
    Returns average positive dE/dt as an input power proxy.
    """
    E_t = np.sum(u_t, axis=(1, 2, 3)) * dV  # shape (T,)
    dE = np.gradient(E_t, dt)
    # Only count positive power injection
    P_in = np.mean(np.clip(dE, 0.0, None))
    return float(P_in)


def coupling_metrics(h_t: np.ndarray, P_t: np.ndarray, u_t: np.ndarray, dV: float, dt: float) -> Dict[str, float]:
    """Compute a set of scalar metrics capturing coupling efficiency.

    h_t: (T,3,3) strain time series
    P_t: (T,) gravitational wave power time series [W]
    u_t: (T,Nx,Ny,Nz) EM energy density [J/m^3]
    dV: cell volume [m^3]
    dt: time step [s]
    """
    h_rms = float(np.sqrt(np.mean(h_t**2)))
    h_max = float(np.max(np.abs(h_t)))
    P_avg = float(np.mean(P_t))
    E_tot_avg = float(np.mean(np.sum(u_t, axis=(1,2,3)) * dV))
    P_in = approximate_input_power_from_time_series(u_t, dV, dt)
    
    metrics = {
        "h_rms": h_rms,
        "h_max": h_max,
        "P_avg": P_avg,
        "E_em_avg": E_tot_avg,
        "P_in": P_in,
        "eff_Pgw_over_Pin": (P_avg / P_in) if P_in > 0 else 0.0,
        "h_rms_per_J": (h_rms / E_tot_avg) if E_tot_avg > 0 else 0.0,
    }
    return metrics
