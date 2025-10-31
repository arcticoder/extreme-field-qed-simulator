"""
Anomalous (parameterized new-physics) coupling module.

Implements κ-parameterized modifications to the effective stress-energy tensor:
    T^{μν}_eff = T^{μν}_EM + κ F^{μν}[A, φ, E, B, ...]

This allows us to:
1. Quantify how large κ must be for observable effects
2. Use experimental null results to constrain κ
3. Systematically explore new-physics scenarios without inventing specific models

The functional F can be chosen from various ansätze motivated by dimensional analysis,
gauge theory, or phenomenological considerations.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any
from .constants import c, epsilon0, mu0, hbar


@dataclass
class AnomalousCoupling:
    """Container for anomalous coupling configuration and results."""
    name: str
    kappa: float  # Coupling strength parameter
    functional: Callable  # F^{μν}[fields] callable
    description: str = ""
    units: str = ""  # Physical units of κ


def vector_potential_squared_ansatz(A: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """Ansatz: F ∝ |A|² η^{μν} (isotropic energy density from vector potential).
    
    Motivation: Some beyond-SM theories couple gravity to vector potential directly.
    Dimensional analysis: [κ] = [Energy]/[A²] = J·s²/(kg·m⁴) in SI.
    
    A: (N,3) vector potential [V·s/m] or [T·m]
    kappa: coupling strength
    Returns: (N,) anomalous energy density contribution [J/m³]
    """
    A = np.asarray(A, dtype=float)
    A2 = np.sum(A * A, axis=-1)
    # Return isotropic energy density contribution
    return kappa * A2


def spatial_gradient_ansatz(A: np.ndarray, positions: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """Ansatz: F ∝ ∇·(|A|²) (spatially localized coupling).
    
    Motivation: Coupling concentrated where vector potential gradients are large
    (e.g., near sources, boundaries).
    
    A: (N,3) vector potential
    positions: (N,3) grid positions [m]
    kappa: coupling strength
    Returns: (N,) anomalous energy density [J/m³]
    """
    A = np.asarray(A, dtype=float)
    pos = np.asarray(positions, dtype=float)
    
    A2 = np.sum(A * A, axis=-1)
    
    # Compute gradient using finite differences
    # For simplicity on irregular grids, use nearest-neighbor approximation
    # In production, use proper spectral or FD stencils
    grad_A2 = np.gradient(A2, axis=0)  # Simplified 1D gradient as placeholder
    
    # Divergence ~ sum of spatial derivatives
    # This is a crude approximation; full implementation needs proper grid geometry
    return kappa * np.abs(grad_A2)


def field_invariant_ansatz(E: np.ndarray, B: np.ndarray, kappa: float = 1.0, 
                           use_F: bool = True, use_G: bool = False) -> np.ndarray:
    """Ansatz: F ∝ field invariants F² or G² or mixed.
    
    Motivation: Lorentz-invariant couplings respecting Maxwell gauge symmetry.
    
    F = (B² - E²/c²)/2  (first invariant)
    G = E·B/c          (second invariant, pseudoscalar)
    
    E: (N,3) electric field [V/m]
    B: (N,3) magnetic field [T]
    kappa: coupling strength
    use_F, use_G: which invariants to include
    Returns: (N,) anomalous energy density [J/m³]
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    
    E2 = np.sum(E * E, axis=-1)
    B2 = np.sum(B * B, axis=-1)
    
    result = np.zeros_like(E2)
    
    if use_F:
        F = 0.5 * (B2 - E2 / c**2)
        result += kappa * F**2
    
    if use_G:
        EB_dot = np.sum(E * B, axis=-1)
        G = EB_dot / c
        result += kappa * G**2
    
    return result


def photon_number_coupling_ansatz(E: np.ndarray, B: np.ndarray, omega: float, kappa: float = 1.0) -> np.ndarray:
    """Ansatz: F ∝ photon number density n_γ = u/(ħω).
    
    Motivation: Some quantum gravity proposals suggest coupling scales with
    coherent photon number in a mode.
    
    E, B: field arrays
    omega: characteristic angular frequency [rad/s]
    kappa: coupling strength [units: J·s]
    Returns: (N,) anomalous contribution [J/m³]
    """
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    
    E2 = np.sum(E * E, axis=-1)
    B2 = np.sum(B * B, axis=-1)
    u = 0.5 * (epsilon0 * E2 + B2 / mu0)
    
    # Photon number density
    n_gamma = u / (hbar * omega + 1e-30)  # avoid division by zero
    
    return kappa * n_gamma


def apply_anomalous_coupling(T00_EM: np.ndarray, coupling: AnomalousCoupling, 
                             fields: Dict[str, np.ndarray]) -> np.ndarray:
    """Apply anomalous coupling to EM stress-energy.
    
    T00_eff = T00_EM + coupling.functional(fields, kappa=coupling.kappa)
    
    T00_EM: (N,) electromagnetic energy density [J/m³]
    coupling: AnomalousCoupling instance
    fields: dict containing field arrays ('E', 'B', 'A', 'positions', etc.)
    Returns: (N,) effective energy density [J/m³]
    """
    # Call the functional with provided fields and kappa
    anomalous_contribution = coupling.functional(**fields, kappa=coupling.kappa)
    
    return T00_EM + anomalous_contribution


def required_kappa_for_target_strain(h_target: float, h_EM: float, 
                                     anomalous_fraction: float = 0.1) -> float:
    """Estimate κ required to boost strain from h_EM to h_target.
    
    Assumes linear scaling: h ∝ T00 ∝ (T00_EM + κ F)
    If anomalous contribution is anomalous_fraction of total T00_eff,
    then κ can be estimated.
    
    h_target: desired strain amplitude
    h_EM: strain from pure EM stress-energy
    anomalous_fraction: fraction of T00_eff from anomalous term
    Returns: estimated κ (units depend on chosen functional)
    """
    if h_EM >= h_target:
        return 0.0  # Already exceeds target
    
    # h ∝ T00, so h_target/h_EM = T00_eff/T00_EM
    boost_factor = h_target / (h_EM + 1e-50)
    
    # If T00_eff = T00_EM + κ F, and κ F = anomalous_fraction * T00_eff
    # Then T00_eff = T00_EM + anomalous_fraction * T00_eff
    # => T00_eff * (1 - anomalous_fraction) = T00_EM
    # => T00_eff = T00_EM / (1 - anomalous_fraction)
    
    # We want T00_eff / T00_EM = boost_factor
    # So anomalous_fraction = 1 - 1/boost_factor
    required_anomalous_fraction = 1.0 - 1.0 / boost_factor
    
    # This gives us the fractional contribution; actual κ depends on F magnitude
    # Caller must normalize by characteristic F value
    return required_anomalous_fraction


def compute_kappa_constraints(h_EM: np.ndarray, P_EM: np.ndarray,
                              detection_thresholds: Dict[str, float],
                              F_characteristic: float,
                              T00_EM_avg: float) -> Dict[str, float]:
    """Compute κ upper bounds from null detection across multiple thresholds.
    
    h_EM: strain time series from pure EM
    P_EM: GW power from pure EM
    detection_thresholds: dict of {name: threshold_h} for different detectors
    F_characteristic: typical magnitude of anomalous functional F
    T00_EM_avg: average EM energy density [J/m³]
    
    Returns: dict of {detector_name: κ_upper_bound}
    """
    h_rms_EM = float(np.sqrt(np.mean(h_EM**2)))
    
    constraints = {}
    
    for detector_name, h_threshold in detection_thresholds.items():
        if h_rms_EM >= h_threshold:
            # Already detectable with EM alone
            constraints[detector_name] = 0.0
        else:
            # Compute required boost
            boost = h_threshold / h_rms_EM
            
            # Required anomalous T00 contribution
            delta_T00_required = T00_EM_avg * (boost - 1.0)
            
            # Estimate κ: if δT00 = κ F_char, then κ = δT00 / F_char
            kappa_required = delta_T00_required / (F_characteristic + 1e-50)
            
            constraints[detector_name] = kappa_required
    
    return constraints


# Predefined coupling configurations for common scenarios
PREDEFINED_COUPLINGS = {
    "vector_potential_squared": AnomalousCoupling(
        name="VectorPotentialSquared",
        kappa=1.0,
        functional=vector_potential_squared_ansatz,
        description="Isotropic energy density from |A|²",
        units="J·s²/(kg·m⁴)"
    ),
    "field_invariant_F2": AnomalousCoupling(
        name="FieldInvariantF2",
        kappa=1.0,
        functional=lambda E, B, kappa: field_invariant_ansatz(E, B, kappa, use_F=True, use_G=False),
        description="Coupling to first Lorentz invariant F²",
        units="[dimensionless or J·m³]"
    ),
    "photon_number": AnomalousCoupling(
        name="PhotonNumberCoupling",
        kappa=1.0,
        functional=photon_number_coupling_ansatz,
        description="Coupling to coherent photon number density",
        units="J·s"
    ),
}
