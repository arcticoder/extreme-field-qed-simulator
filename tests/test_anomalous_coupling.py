"""
Integration tests for anomalous_coupling module.

Tests:
1. κ-scaling verification (h ∝ κ for linear ansätze)
2. Field normalization (dimensionless vs. dimensioned)
3. Ansatz symmetries (parity, Lorentz invariance)
4. Constraint computation accuracy
5. Predefined ansätze availability
"""

import pytest
import numpy as np
from src.efqs.anomalous_coupling import (
    vector_potential_squared_ansatz,
    field_invariant_ansatz,
    photon_number_coupling_ansatz,
    axion_like_ansatz,
    dilaton_like_ansatz,
    chern_simons_like_ansatz,
    spatial_gradient_ansatz,
    required_kappa_for_target_strain,
    compute_kappa_constraints,
    PREDEFINED_COUPLINGS
)
from src.efqs.constants import c, epsilon0, mu0, hbar


class TestAnsatzFunctions:
    """Test individual ansatz functional forms."""
    
    def test_vector_potential_squared(self):
        """Test A² ansatz scaling and units."""
        A = np.array([1e-6, 0, 0])  # Magnetic vector potential (T·m)
        kappa = 1.0
        
        result = vector_potential_squared_ansatz(A, kappa)
        
        # Should return stress-energy-like tensor (diagonal components)
        assert result.shape == (3, 3)
        assert np.allclose(result, result.T)  # Symmetric
        
        # Should scale with A²
        result_2x = vector_potential_squared_ansatz(2*A, kappa)
        assert np.allclose(result_2x, 4 * result, rtol=1e-10)
        
        # Should scale linearly with κ
        result_kappa2 = vector_potential_squared_ansatz(A, 2*kappa)
        assert np.allclose(result_kappa2, 2 * result, rtol=1e-10)
    
    def test_field_invariant(self):
        """Test F² = (B² - E²/c²) ansatz."""
        E = np.array([1e8, 0, 0])  # V/m
        B = np.array([0, 0.1, 0])  # T
        kappa = 1e-20
        
        result = field_invariant_ansatz(E, B, kappa)
        
        # Should be diagonal (isotropic)
        assert result.shape == (3, 3)
        
        # Parity even: should not change under E → -E, B → B
        result_flip_E = field_invariant_ansatz(-E, B, kappa)
        assert np.allclose(result, result_flip_E, rtol=1e-10)
    
    def test_photon_number(self):
        """Test photon number ansatz scaling."""
        E = np.array([1e10, 0, 0])  # V/m
        B = E[0] / c * np.array([0, 1, 0])  # Plane wave: |E| = c|B|
        omega = 2.4e15  # 800 nm → ω ~ 2.4e15 rad/s
        kappa = 1e-30
        
        result = photon_number_coupling_ansatz(E, B, omega, kappa)
        
        assert result.shape == (3, 3)
        
        # Should scale with E² (energy density u ∝ E²)
        result_2x = photon_number_coupling_ansatz(2*E, 2*B, omega, kappa)
        assert np.allclose(result_2x, 4 * result, rtol=1e-10)
        
        # Should scale inversely with ω (n_γ = u/(ℏω))
        result_high_freq = photon_number_coupling_ansatz(E, B, 2*omega, kappa)
        assert np.allclose(result_high_freq, 0.5 * result, rtol=1e-10)
    
    def test_axion_like_parity_violation(self):
        """Test axion-like E·B ansatz violates parity."""
        E = np.array([1e8, 0, 0])  # V/m
        B = np.array([0, 0.5, 0])  # T
        kappa = 1e-15
        
        result = axion_like_ansatz(E, B, kappa)
        
        # E·B is pseudoscalar: changes sign under parity (r → -r, E → -E, B → B)
        result_parity = axion_like_ansatz(-E, B, kappa)
        assert np.allclose(result, -result_parity, rtol=1e-10)
    
    def test_dilaton_like_trace(self):
        """Test dilaton-like T^μ_μ trace ansatz."""
        E = np.array([1e9, 0, 0])  # V/m
        B = np.array([0, 0, 0.1])  # T
        kappa = 1.0  # Dimensionless for trace coupling
        
        result = dilaton_like_ansatz(E, B, kappa)
        
        # Should return stress-energy contribution
        assert result.shape == (3, 3)
        
        # Should scale with E⁴ (QED trace correction ∝ (E²)²)
        result_2x = dilaton_like_ansatz(2*E, B, kappa)
        # Expect ~16× for quartic term
        ratio = np.linalg.norm(result_2x) / np.linalg.norm(result)
        assert 10 < ratio < 20  # Approximately 16
    
    def test_chern_simons_like(self):
        """Test Chern-Simons A·B Lorentz violation."""
        A = np.array([1e-8, 0, 0])  # T·m
        E = np.array([1e7, 0, 0])  # V/m
        B = np.array([0, 0.1, 0])  # T
        kappa = 1e-20
        
        result = chern_simons_like_ansatz(E, B, A, kappa)
        
        # Parity violating: A·B pseudoscalar
        result_parity = chern_simons_like_ansatz(E, B, -A, kappa)
        assert np.allclose(result, -result_parity, rtol=1e-10)
    
    def test_spatial_gradient(self):
        """Test spatial gradient ansatz."""
        A = np.array([1e-6, 0, 0])
        positions = np.array([[0, 0, 0]])  # Single point
        kappa = 1e-25
        
        result = spatial_gradient_ansatz(A, positions, kappa)
        
        # Should return stress-energy tensor
        assert result.shape == (3, 3)


class TestKappaScaling:
    """Test κ-dependent scaling in constraint computation."""
    
    def test_kappa_required_inverse_scaling(self):
        """Test κ_required ∝ (h_threshold / h_baseline)."""
        h_baseline1 = 1e-22
        h_baseline2 = 4e-22  # 4× larger
        h_threshold = 1e-21
        
        kappa1 = required_kappa_for_target_strain(h_threshold, h_baseline1)
        kappa2 = required_kappa_for_target_strain(h_threshold, h_baseline2)
        
        # κ_required ∝ (h_threshold / h_baseline)
        # 4× h_baseline should give 1/4 κ_required
        ratio = kappa1 / kappa2
        assert np.isclose(ratio, 4.0, rtol=0.1)


class TestPredefinedCouplings:
    """Test registry of predefined ansätze."""
    
    def test_all_ansatze_available(self):
        """Verify all advertised ansätze are in registry."""
        expected_ansatze = [
            'vector_potential_squared',
            'field_invariant_F2',
            'photon_number',
            'axion_like',
            'dilaton_like',
            'chern_simons_like',
            'spatial_gradient'
        ]
        
        for ansatz in expected_ansatze:
            assert ansatz in PREDEFINED_COUPLINGS, f"{ansatz} missing from registry"
    
    def test_ansatz_metadata(self):
        """Check that each ansatz has description and function."""
        for name, info in PREDEFINED_COUPLINGS.items():
            assert 'description' in info
            assert 'function' in info
            assert callable(info['function'])
            assert isinstance(info['description'], str)


class TestConstraintComputation:
    """Test κ-constraint calculation for different scenarios."""
    
    def test_zero_baseline_infinite_kappa(self):
        """If h_baseline = 0, any κ allows detection (edge case)."""
        h_baseline = 0.0
        h_threshold = 1e-22
        
        kappa = required_kappa_for_target_strain(h_threshold, h_baseline)
        # Should return inf or very large number (no constraint)
        assert np.isinf(kappa) or kappa > 1e100
    
    def test_baseline_above_threshold(self):
        """If h_baseline > threshold, already detectable."""
        h_baseline = 1e-20
        h_threshold = 1e-21
        
        kappa = required_kappa_for_target_strain(h_threshold, h_baseline)
        # Should return 0 or very small (already detectable)
        assert kappa <= 1.0
    
    def test_realistic_constraint(self):
        """Realistic scenario: weak baseline, need κ boost."""
        h_baseline = 1e-58  # Typical from simulations
        h_threshold = 1e-22  # LIGO sensitivity
        
        kappa = required_kappa_for_target_strain(h_threshold, h_baseline)
        
        # Should be very large (10^30+) but finite
        assert 1e20 < kappa < 1e60
        assert np.isfinite(kappa)


class TestNumericalStability:
    """Test numerical stability for extreme field values."""
    
    def test_near_schwinger_field(self):
        """Test ansätze near Schwinger critical field."""
        E_schwinger = 1.3e18  # V/m
        E = np.array([0.1 * E_schwinger, 0, 0])
        B = E[0] / c * np.array([0, 1, 0])
        kappa = 1e-30
        
        # Should not overflow or underflow
        result = field_invariant_ansatz(E, B, kappa)
        assert np.all(np.isfinite(result))
        assert not np.any(np.isnan(result))
    
    def test_very_weak_fields(self):
        """Test with laboratory-scale weak fields."""
        E = np.array([1e3, 0, 0])  # 1 kV/m
        B = np.array([0, 1e-6, 0])  # 1 μT
        kappa = 1e10
        
        result = field_invariant_ansatz(E, B, kappa)
        assert np.all(np.isfinite(result))
    
    def test_high_frequency_photons(self):
        """Test photon number ansatz at gamma-ray frequencies."""
        E = np.array([1e7, 0, 0])
        B = E[0] / c * np.array([0, 1, 0])
        omega_gamma = 1e20  # ~100 keV
        kappa = 1e-25
        
        result = photon_number_coupling_ansatz(E, B, omega_gamma, kappa)
        assert np.all(np.isfinite(result))


class TestSymmetries:
    """Test expected physical symmetries of ansätze."""
    
    def test_lorentz_scalar_rotation_invariance(self):
        """F² ansatz should be rotationally invariant."""
        E = np.array([1e8, 1e8, 0])  # Diagonal field
        B = np.array([0, 0, 0.5])
        kappa = 1e-20
        
        result1 = field_invariant_ansatz(E, B, kappa)
        
        # Rotate 90° around z: (Ex, Ey) → (-Ey, Ex)
        E_rot = np.array([-1e8, 1e8, 0])
        result2 = field_invariant_ansatz(E_rot, B, kappa)
        
        # F² = B² - E²/c² should be invariant under rotation
        # Stress-energy components will rotate, but trace should match
        assert np.isclose(np.trace(result1), np.trace(result2), rtol=1e-10)
    
    def test_time_reversal(self):
        """Check time-reversal properties."""
        E = np.array([1e7, 0, 0])
        B = np.array([0, 0.3, 0])
        kappa = 1e-18
        
        # Under time reversal: E → E, B → -B
        result_forward = field_invariant_ansatz(E, B, kappa)
        result_reversed = field_invariant_ansatz(E, -B, kappa)
        
        # F² is time-reversal even
        assert np.allclose(result_forward, result_reversed, rtol=1e-10)
        
        # Axion-like E·B is time-reversal odd
        result_ax_fwd = axion_like_ansatz(E, B, kappa)
        result_ax_rev = axion_like_ansatz(E, -B, kappa)
        assert np.allclose(result_ax_fwd, -result_ax_rev, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
