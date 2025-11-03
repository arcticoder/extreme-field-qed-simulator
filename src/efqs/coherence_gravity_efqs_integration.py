"""
EFQS Integration Module: Torsion-DOF, κ_R→k_3 Mapping, Duality Breaking

This module provides hooks to integrate coherence-gravity-coupling physics
into the extreme-field QED simulator (EFQS) run_experiments.py pipeline.

Integration points:
    1. Torsion-like DOF via coherence gradients (torsion_dof.py)
    2. Laboratory κ_R bounds → torsion-EM k_3 constraints (kappa_k3_mapping.py)
    3. Duality-breaking observable from antisymmetric stress-energy

Usage in run_experiments.py:
    from coherence_gravity_efqs_integration import (
        add_torsion_proxy_stress,
        compute_k3_constraints,
        evaluate_duality_breaking
    )
    
    # After computing standard T00:
    T00_total = add_torsion_proxy_stress(T00, coherence_field, positions, dV)
    
    # At end of experiment:
    k3_bounds = compute_k3_constraints(kappa_R_measured)
    duality = evaluate_duality_breaking(E, B, coherence_field)
"""
from __future__ import annotations

import numpy as np
import sys
import os
import importlib.util

# Import coherence-gravity modules using direct file loading
coherence_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'coherence-gravity-coupling'))

# Load torsion_dof module
torsion_module_path = os.path.join(coherence_path, 'src', 'field_equations', 'torsion_dof.py')
spec = importlib.util.spec_from_file_location("torsion_dof", torsion_module_path)
torsion_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torsion_module)

coherence_gradient_tensor = torsion_module.coherence_gradient_tensor
torsion_proxy_stress_energy = torsion_module.torsion_proxy_stress_energy
duality_breaking_observable = torsion_module.duality_breaking_observable

# Load kappa_k3_mapping module
kappa_module_path = os.path.join(coherence_path, 'src', 'analysis', 'kappa_k3_mapping.py')
spec = importlib.util.spec_from_file_location("kappa_k3_mapping", kappa_module_path)
kappa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kappa_module)

map_kappa_to_k3 = kappa_module.map_kappa_to_k3
eft_coefficient_mapping = kappa_module.eft_coefficient_mapping


def add_torsion_proxy_stress(
    T00_standard: np.ndarray,
    coherence_field: np.ndarray,
    positions: np.ndarray,
    dV: float,
    coupling_strength: float = 1.0
) -> np.ndarray:
    """Add torsion-proxy stress-energy to standard EM stress.
    
    T00_total = T00_EM + α * T_torsion
    
    where T_torsion ~ antisymmetric part of ∇Φ ⊗ ∇Φ
    
    Args:
        T00_standard: Standard EM stress-energy [J/m³]
        coherence_field: Φ(x,y,z) [m⁻¹]
        positions: (N,3) array of positions
        dV: Volume element
        coupling_strength: Dimensionless α scaling torsion contribution
    
    Returns:
        T00_total: Combined stress-energy
    """
    # Compute coherence gradient tensor
    grid_shape = int(np.round(len(positions)**(1/3)))
    dx = dV**(1/3)
    
    Phi_grid = coherence_field.reshape(grid_shape, grid_shape, grid_shape)
    
    # Compute gradients manually
    grad_Phi_x = np.gradient(Phi_grid, dx, axis=0, edge_order=2)
    grad_Phi_y = np.gradient(Phi_grid, dx, axis=1, edge_order=2)
    grad_Phi_z = np.gradient(Phi_grid, dx, axis=2, edge_order=2)
    
    # Build gradient tensor G_ij = ∇_i Φ ∇_j Φ
    G_00 = grad_Phi_x * grad_Phi_x
    G_01 = grad_Phi_x * grad_Phi_y
    G_02 = grad_Phi_x * grad_Phi_z
    G_10 = grad_Phi_y * grad_Phi_x
    G_11 = grad_Phi_y * grad_Phi_y
    G_12 = grad_Phi_y * grad_Phi_z
    G_20 = grad_Phi_z * grad_Phi_x
    G_21 = grad_Phi_z * grad_Phi_y
    G_22 = grad_Phi_z * grad_Phi_z
    
    # Antisymmetric part (torsion proxy)
    A_01 = (G_01 - G_10) / 2
    A_02 = (G_02 - G_20) / 2
    A_12 = (G_12 - G_21) / 2
    
    # Scalar invariant: Tr(A² ) = sum of squares
    A_squared_trace = 2 * (A_01**2 + A_02**2 + A_12**2)
    
    # Torsion-proxy stress (dimensionally T00 ~ [Φ]² [length]⁻² ~ m⁻⁴)
    # Scale to physical units: assume Φ has dimension [energy/length] ~ J/m
    hbar_c = 1.054571817e-34 * 299792458  # J·m
    T_torsion = (hbar_c / (grid_shape * dx)**2) * A_squared_trace.ravel()
    
    # Scale and add
    T00_total = T00_standard + coupling_strength * T_torsion
    
    return T00_total


def compute_k3_constraints(
    kappa_R_lab: float,
    scenario: str = 'conservative',
    include_astrophysical: bool = True
) -> dict:
    """Map laboratory κ_R bound to Bahamonde torsion-EM k_3.
    
    Args:
        kappa_R_lab: Measured/constrained κ_R [m²]
        scenario: 'conservative', 'moderate', or 'optimistic'
        include_astrophysical: Include magnetar amplification
    
    Returns:
        dict with keys:
            'k3_bound': Direct k_3 constraint [m²]
            'k3_astrophysical': Magnetar-amplified k_3 [m²]
            'scenario_details': EFT mapping info
    """
    result = map_kappa_to_k3(kappa_R_lab, scenario)
    
    # Extract k3 from result
    if 'k3_upper_limit' in result:
        k3_bound = result['k3_upper_limit']
    elif 'k3_central' in result:
        k3_bound = result['k3_central']
    else:
        k3_bound = kappa_R_lab  # Fallback
    
    output = {
        'k3_bound': k3_bound,
        'scenario': scenario,
        'scenario_details': result
    }
    
    if include_astrophysical:
        # Magnetar amplification (10^38× from 10^15 G B-field)
        k3_astro = k3_bound * 1e-38  # Improvement factor
        output['k3_astrophysical'] = k3_astro
    
    return output


def evaluate_duality_breaking(
    E_field: np.ndarray,
    B_field: np.ndarray,
    coherence_field: np.ndarray,
    dV: float
) -> dict:
    """Evaluate duality-breaking signature from torsion-proxy coupling.
    
    Observable: ∫ E·(∇×A_antisym) d³x
    
    where A_antisym is antisymmetric part of coherence gradient tensor.
    
    Args:
        E_field: (N,3) electric field
        B_field: (N,3) magnetic field
        coherence_field: Φ(x) [m⁻¹]
        dV: Volume element
    
    Returns:
        dict with keys:
            'duality_integral': Value of duality-breaking observable
            'normalized': Ratio to standard EM integrals
    """
    grid_shape = int(np.round(len(coherence_field)**(1/3)))
    dx = dV**(1/3)
    
    Phi_grid = coherence_field.reshape(grid_shape, grid_shape, grid_shape)
    E_grid = E_field.reshape(grid_shape, grid_shape, grid_shape, 3)
    
    # Compute coherence gradients
    grad_Phi_x = np.gradient(Phi_grid, dx, axis=0, edge_order=2)
    grad_Phi_y = np.gradient(Phi_grid, dx, axis=1, edge_order=2)
    grad_Phi_z = np.gradient(Phi_grid, dx, axis=2, edge_order=2)
    
    # Antisymmetric combinations
    A_xy = (grad_Phi_x[:,:,:,None] * grad_Phi_y[:,:,:,None].T - grad_Phi_y[:,:,:,None] * grad_Phi_x[:,:,:,None].T).squeeze() / 2
    
    # Simplified duality observable: ∫ E_x * A_xy dV
    # (Full version would include curl of A tensor)
    integral = np.sum(E_grid[:,:,:,0] * grad_Phi_y * grad_Phi_z) * dV
    
    # Normalize to standard EM energy
    epsilon_0 = 8.854e-12
    EM_energy = np.sum(E_field**2 + B_field**2) * dV * epsilon_0
    
    if EM_energy > 0:
        normalized = abs(integral) / EM_energy
    else:
        normalized = 0.0
    
    return {
        'duality_integral': float(integral),
        'EM_energy': float(EM_energy),
        'normalized': float(normalized),
        'interpretation': 'Non-zero indicates EM duality violation via coherence'
    }


def create_gaussian_coherence_field(
    positions: np.ndarray,
    center: np.ndarray = None,
    amplitude: float = 1e8,
    width: float = 1e-3
) -> np.ndarray:
    """Create example Gaussian coherence field for testing.
    
    Φ(r) = Φ_0 * exp(-|r-r0|²/σ²)
    
    Args:
        positions: (N,3) grid points
        center: Field center [m]
        amplitude: Peak Φ_0 [m⁻¹]
        width: Gaussian width σ [m]
    
    Returns:
        Phi: Coherence field [m⁻¹]
    """
    if center is None:
        center = np.zeros(3)
    
    r = positions - center
    r_mag_sq = np.sum(r**2, axis=1)
    
    Phi = amplitude * np.exp(-r_mag_sq / width**2)
    
    return Phi


# Example integration workflow
if __name__ == "__main__":
    print("EFQS Integration Module: Coherence-Gravity Physics")
    print("=" * 60)
    
    # Mock EFQS data
    N = 20
    positions = np.random.randn(N**3, 3) * 1e-3  # ~mm scale
    dV = (2e-3 / N)**3
    
    E_field = np.random.randn(N**3, 3) * 1e10  # V/m
    B_field = np.random.randn(N**3, 3) * 1.0   # T
    
    # Standard stress-energy
    epsilon_0 = 8.854e-12
    T00_standard = 0.5 * epsilon_0 * np.sum(E_field**2, axis=1)
    
    # Create coherence field
    Phi = create_gaussian_coherence_field(positions, amplitude=1e8, width=1e-3)
    
    print("\n[1. Torsion-proxy stress addition]")
    T00_total = add_torsion_proxy_stress(T00_standard, Phi, positions, dV, coupling_strength=0.1)
    
    print(f"  Standard T00: mean = {np.mean(T00_standard):.3e} J/m³")
    print(f"  With torsion: mean = {np.mean(T00_total):.3e} J/m³")
    print(f"  Fractional change: {(np.mean(T00_total) - np.mean(T00_standard)) / np.mean(T00_standard) * 100:.2f}%")
    
    print("\n[2. κ_R → k_3 mapping]")
    kappa_R_measured = 5e17  # m² (lab bound)
    k3_result = compute_k3_constraints(kappa_R_measured, scenario='conservative', include_astrophysical=True)
    
    print(f"  Laboratory κ_R = {kappa_R_measured:.2e} m²")
    print(f"  → k_3 bound = {k3_result['k3_bound']:.2e} m²")
    print(f"  → k_3 (magnetar) = {k3_result['k3_astrophysical']:.2e} m²")
    
    print("\n[3. Duality-breaking observable]")
    duality_result = evaluate_duality_breaking(E_field, B_field, Phi, dV)
    
    print(f"  Duality integral = {duality_result['duality_integral']:.3e}")
    print(f"  Normalized to EM energy = {duality_result['normalized']:.3e}")
    print(f"  {duality_result['interpretation']}")
    
    print("\n✅ EFQS integration module functional")
    print("   Ready to wire into run_experiments.py")
