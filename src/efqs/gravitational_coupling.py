"""
Linearized GR coupling of EM stress–energy to spacetime perturbations.

We provide practical estimators for:
- EM stress-energy tensor T^{mu nu} from E,B fields (SI units)
- Far-field gravitational wave strain via quadrupole formula
- Radiated GW power P_GW

Caveats:
- Uses weak-field, linearized gravity with Minkowski background.
- Far-field (radiation zone) estimates via standard quadrupole formulas.
- For near-field metric perturbations, only crude Poisson-like h00 is provided.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from .constants import c, epsilon0, mu0


@dataclass
class StressEnergy:
    T00: np.ndarray  # energy density [J/m^3]
    T0i: np.ndarray  # momentum density/c = S/c^2 [J/(m^3) * s/m]
    Tij: np.ndarray  # spatial stress tensor [Pa]


def stress_energy_from_fields(E: np.ndarray, B: np.ndarray, include_qed: bool = False) -> StressEnergy:
    """Compute EM stress–energy components from E,B fields.

    E,B shape: (..., 3) with SI units [V/m] and [T]. Returns arrays broadcast to input shape.
    include_qed: if True, apply Heisenberg-Euler QED corrections to energy density
    
    Formulas (SI):
      u = T00 = 1/2 (epsilon0 E^2 + B^2/mu0)
      S = (1/mu0) E x B  (Poynting) [W/m^2]
      T0i = S_i / c^2
      sigma_ij = epsilon0 E_i E_j + (1/mu0) B_i B_j - 1/2 delta_ij (epsilon0 E^2 + B^2/mu0)
    
    QED correction (Heisenberg-Euler):
      Delta u ≈ (2/45) (alpha/pi) (1/E_s^2) [(E^2 - c^2 B^2)^2 + 7(c E·B)^2]
    where E_s is the Schwinger critical field.
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    assert E.shape[-1] == 3 and B.shape[-1] == 3, "E,B must have last dim = 3"
    E2 = np.sum(E * E, axis=-1)
    B2 = np.sum(B * B, axis=-1)
    u = 0.5 * (epsilon0 * E2 + B2 / mu0)
    
    if include_qed:
        # Import locally to avoid circular dependency
        from .constants import alpha, E_s
        # Heisenberg-Euler correction to energy density
        # F = (B^2 - E^2/c^2)/2, G = E·B/c
        F = 0.5 * (B2 - E2 / c**2)
        EB_dot = np.sum(E * B, axis=-1)
        G = EB_dot / c
        # Delta u ≈ (2/45)(alpha/pi)(1/E_s^2) * [(2F)^2 + 7(2cG)^2]
        # Simplify: [(2F)^2 + 7(2cG)^2] = 4F^2 + 28c^2 G^2
        coeff = (2.0 / 45.0) * (alpha / np.pi) / (E_s**2)
        delta_u = coeff * (4.0 * F**2 + 28.0 * (c * G)**2)
        u += delta_u

    # Poynting vector
    S = (1.0 / mu0) * np.cross(E, B, axis=-1)
    T0i = S / (c**2)

    # Spatial stress (Maxwell stress tensor)
    # sigma_ij with shape (..., 3, 3)
    # Build outer products
    EE = epsilon0 * np.einsum('...i,...j->...ij', E, E)
    BB = (1.0 / mu0) * np.einsum('...i,...j->...ij', B, B)
    trace_term = 0.5 * (epsilon0 * E2 + B2 / mu0)
    I = np.eye(3)
    sigma = EE + BB - np.expand_dims(trace_term, axis=(-1,)) * I

    return StressEnergy(T00=u, T0i=T0i, Tij=sigma)


def quadrupole_moment(positions: np.ndarray, rho_energy: np.ndarray) -> np.ndarray:
    r"""Compute mass-energy quadrupole Q_ij = \int rho(x) (x_i x_j - 1/3 r^2 delta_ij) d^3x.
    Here rho(x) = u(x)/c^2 with u = energy density [J/m^3].

    positions: (N,3) coordinates [m]
    rho_energy: (N,) energy density [J/m^3]
    Returns Q with shape (3,3) [kg m^2].
    """
    pos = np.asarray(positions, dtype=float)
    u = np.asarray(rho_energy, dtype=float)
    assert pos.shape[1] == 3 and pos.shape[0] == u.shape[0]
    rho = u / (c**2)
    x = pos[:, 0]; y = pos[:, 1]; z = pos[:, 2]
    r2 = x*x + y*y + z*z
    Q = np.zeros((3, 3), dtype=float)
    # Using discrete sum approximation; assume unit cell volume dV passed via scaling of rho if needed
    Q[0, 0] = np.sum(rho * (x*x - r2/3.0))
    Q[1, 1] = np.sum(rho * (y*y - r2/3.0))
    Q[2, 2] = np.sum(rho * (z*z - r2/3.0))
    Q[0, 1] = Q[1, 0] = np.sum(rho * (x*y))
    Q[0, 2] = Q[2, 0] = np.sum(rho * (x*z))
    Q[1, 2] = Q[2, 1] = np.sum(rho * (y*z))
    return Q


def finite_difference(time_series: np.ndarray, dt: float, order: int) -> np.ndarray:
    """Compute time derivatives by central finite differences (order: 1,2,3)."""
    if order == 1:
        return np.gradient(time_series, dt, axis=0)
    if order == 2:
        return np.gradient(np.gradient(time_series, dt, axis=0), dt, axis=0)
    if order == 3:
        return np.gradient(np.gradient(np.gradient(time_series, dt, axis=0), dt, axis=0), dt, axis=0)
    raise ValueError("order must be 1,2,3")


def strain_far_field(Q_t: np.ndarray, dt: float, R: float, use_tt: bool = True, los: np.ndarray | None = None) -> np.ndarray:
    """Compute far-field strain tensor h_ij(t) ≈ (2G/(c^4 R)) d^2 Q_ij/dt^2.

    Q_t: (T, 3, 3) quadrupole sequence [kg m^2]
    dt: time step [s]
    R: observer distance [m]
    use_tt: if True, apply transverse-traceless (TT) projection
    los: line-of-sight unit vector (3,) for TT projection; defaults to [0,0,1]
    Returns h_t: (T, 3, 3) dimensionless strain tensor.
    """
    G = 6.67430e-11
    Qdd = finite_difference(Q_t, dt, order=2)
    pref = 2.0 * G / (c**4 * R)
    h = pref * Qdd
    
    if use_tt:
        if los is None:
            los = np.array([0.0, 0.0, 1.0])
        h = tt_project(h, los)
    
    return h


def tt_project(h: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Apply transverse-traceless (TT) projection to a strain tensor h_ij.
    
    For a wave propagating along unit vector n, the TT projection is:
      h^TT_ij = P_ik P_jl h_kl - 1/2 P_ij P_kl h_kl
    where P_ij = delta_ij - n_i n_j is the transverse projector.
    
    h: (..., 3, 3) strain tensor(s)
    n: (3,) line-of-sight unit vector
    Returns: (..., 3, 3) TT-projected strain
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)  # ensure unit vector
    
    # Transverse projector P_ij
    I = np.eye(3)
    P = I - np.outer(n, n)
    
    # Apply TT projection
    # h^TT_ij = P_ik P_jl h_kl - 1/2 P_ij (P_kl h_kl)
    # Using einsum for clarity
    h_TT = np.einsum('ik,jl,...kl->...ij', P, P, h)
    trace_term = np.einsum('kl,...kl->...', P, h)
    h_TT -= 0.5 * np.einsum('ij,...->...ij', P, trace_term)
    
    return h_TT


def radiated_power_from_quadrupole(Q_t: np.ndarray, dt: float) -> np.ndarray:
    """Instantaneous GW power via quadrupole formula:
      P(t) = (G/(5 c^5)) <...Q^{(3)}_ij Q^{(3)}_ij...>
    Approximate by contracting the 3rd derivative tensor with itself.

    Returns P(t) [W] as array of length T.
    """
    G = 6.67430e-11
    Q3 = finite_difference(Q_t, dt, order=3)
    # Contract over i,j
    contracted = np.einsum('tij,tij->t', Q3, Q3)
    P = (G / (5.0 * c**5)) * contracted
    return P


def h00_static_poisson(u: np.ndarray, dV: float, positions: np.ndarray, probe_pos: np.ndarray) -> float:
    """Crude near-field static potential-like perturbation h00 from energy density u(r):
      ∇^2 h00 ≈ -16π G T00 / c^4, with T00 = u.
    Solve Green's function integral: h00(r) ≈ (4G/(c^4)) ∫ u(r')/|r-r'| d^3r'.
    """
    G = 6.67430e-11
    r = np.asarray(probe_pos)
    R = np.linalg.norm(positions - r, axis=1) + 1e-12
    val = (4.0 * G / (c**4)) * np.sum(u / R * dV)
    return float(val)
