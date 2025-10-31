"""Tests for coupling metrics and energy loss dynamics."""
import numpy as np
import pytest
from efqs.metrics import coupling_metrics, total_energy, approximate_input_power_from_time_series
from efqs.pair_production import energy_loss_power, schwinger_rate
from efqs.constants import E_s, me, c, hbar


def test_total_energy_positive():
    """Total energy should be positive for positive energy density."""
    u = np.array([[[1e12, 2e12], [3e12, 4e12]]])  # (1,2,2) grid
    dV = 1e-6
    E = total_energy(u, dV)
    assert E > 0.0
    # sum = 1e12 + 2e12 + 3e12 + 4e12 = 10e12 = 1e13
    # E = sum * dV = 1e13 * 1e-6 = 1e7
    assert np.isclose(E, 1e7)


def test_coupling_metrics_shape_and_keys():
    """Coupling metrics should return expected keys and sensible values."""
    # Simple mock strain and power time series
    h_t = np.random.randn(100, 3, 3) * 1e-22
    P_t = np.abs(np.random.randn(100)) * 1e20
    u_t = np.ones((100, 10, 10, 10)) * 1e12
    dV = 1e-6
    dt = 1e-15
    
    m = coupling_metrics(h_t, P_t, u_t, dV, dt)
    
    assert 'h_rms' in m
    assert 'h_max' in m
    assert 'P_avg' in m
    assert 'E_em_avg' in m
    assert 'P_in' in m
    assert 'eff_Pgw_over_Pin' in m
    assert 'h_rms_per_J' in m
    
    assert m['h_rms'] > 0.0
    assert m['h_max'] >= m['h_rms']
    assert m['P_avg'] > 0.0
    assert m['E_em_avg'] > 0.0


def test_pair_loss_reduces_energy():
    """Pair production loss should reduce total EM energy over time."""
    # Setup: field near Schwinger threshold
    E0 = 0.1 * E_s  # 10% of critical field
    volume = 1e-12  # 1 cubic micron
    dt = 1e-15  # 1 fs
    
    # Initial energy density (EM energy ~ epsilon0 E^2 / 2)
    from efqs.constants import epsilon0
    u_initial = 0.5 * epsilon0 * E0**2
    E_initial = u_initial * volume
    
    # Loss power
    P_loss = energy_loss_power(E0, volume, n_terms=2)
    
    # Energy after one time step with loss
    E_after = E_initial - P_loss * dt
    
    # Should have less energy (or at least non-increasing)
    assert E_after <= E_initial
    
    # For fields well below Schwinger, loss should be tiny
    E0_weak = 1e13  # V/m << E_s
    P_loss_weak = energy_loss_power(E0_weak, volume, n_terms=1)
    # Loss should be negligible
    assert P_loss_weak < 1e-10  # practically zero


def test_approximate_input_power():
    """Input power estimate should scale with energy and inverse time."""
    # Constant energy density: gradient ~ 0, clipped to 0
    u_const = np.ones((100, 5, 5, 5)) * 1e12
    dV = 1e-6
    dt = 1e-15
    P_const = approximate_input_power_from_time_series(u_const, dV, dt)
    # For constant u, derivative ~ 0, clipped to 0
    assert P_const == 0.0
    
    # Increasing energy density
    u_inc = np.linspace(1e12, 2e12, 100).reshape(100, 1, 1, 1) * np.ones((100, 5, 5, 5))
    P_inc = approximate_input_power_from_time_series(u_inc, dV, dt)
    # Should be positive due to positive gradient
    assert P_inc > 0.0


def test_schwinger_rate_scaling():
    """Schwinger rate should scale exponentially with E/E_s."""
    w1 = schwinger_rate(0.01 * E_s)
    w2 = schwinger_rate(0.1 * E_s)
    w3 = schwinger_rate(0.5 * E_s)
    
    # Should increase as field approaches critical
    assert w2 > w1
    assert w3 > w2
    
    # Very weak field should have negligible rate
    assert w1 < 1e-30


def test_metrics_energy_conservation_check():
    """If no losses, efficiency should reflect pure GW radiation."""
    # Mock data: constant energy, small GW power
    steps = 50
    h_t = np.random.randn(steps, 3, 3) * 1e-25
    P_gw = np.ones(steps) * 1e15  # 1 PW GW output
    u_t = np.ones((steps, 8, 8, 8)) * 1e10  # constant 1e10 J/m^3
    dV = 1e-6
    dt = 1e-15
    
    m = coupling_metrics(h_t, P_gw, u_t, dV, dt)
    
    # Total EM energy
    E_em = m['E_em_avg']
    # Efficiency should be small (GW << EM)
    assert m['eff_Pgw_over_Pin'] < 1.0
    assert m['eff_Pgw_over_Pin'] >= 0.0
