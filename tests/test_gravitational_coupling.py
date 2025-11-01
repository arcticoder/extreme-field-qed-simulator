"""
Unit tests for gravitational coupling module (quadrupole approximation).
"""
import numpy as np
from efqs.gravitational_coupling import (
    em_energy_density,
    compute_quadrupole,
    spectral_derivative,
    compute_h_and_power,
    run_pipeline,
)


def test_em_energy_density_zero_fields():
    """Energy density should be zero for zero fields."""
    E = np.zeros((10, 3))
    B = np.zeros((10, 3))
    u = em_energy_density(E, B)
    assert np.allclose(u, 0.0)


def test_quadrupole_zero_energy():
    """Quadrupole should be zero for zero energy density."""
    grid = {"x": np.linspace(-1, 1, 5), "y": np.linspace(-1, 1, 5), "z": np.linspace(-1, 1, 5)}
    nt, nx, ny, nz = 3, 5, 5, 5
    u_em = np.zeros((nt, nx, ny, nz))
    Q = compute_quadrupole(grid, u_em)
    assert Q.shape == (nt, 3, 3)
    assert np.allclose(Q, 0.0)


def test_spectral_derivative_sinusoid():
    """Derivative of sin(ωt) should be approximately ω·cos(ωt)."""
    nt = 128
    t = np.linspace(0, 2*np.pi, nt, endpoint=False)
    omega = 5.0
    ts = np.sin(omega * t)[:, None, None]  # shape (nt, 1, 1)
    dt = t[1] - t[0]
    
    deriv1 = spectral_derivative(ts, dt, order=1)
    expected = omega * np.cos(omega * t)[:, None, None]
    
    # Check RMS error instead of pointwise (more robust for spectral methods)
    rms_error = np.sqrt(np.mean((deriv1 - expected)**2))
    assert rms_error < 0.5, f"RMS error {rms_error} too large"


def test_run_pipeline_shapes():
    """Check that run_pipeline returns correct shapes."""
    nt, nx, ny, nz = 8, 3, 3, 2
    x = np.linspace(-0.1, 0.1, nx)
    y = np.linspace(-0.1, 0.1, ny)
    z = np.linspace(-0.1, 0.1, nz)
    grid = {"x": x, "y": y, "z": z}

    dt = 1e-3
    E = np.random.rand(nt, nx, ny, nz, 3) * 1e3
    B = np.random.rand(nt, nx, ny, nz, 3) * 1e-6

    Q, h, P = run_pipeline(grid, E, B, dt, R=10.0)

    assert Q.shape == (nt, 3, 3), f"Expected Q shape (8,3,3), got {Q.shape}"
    assert h.shape == (nt, 3, 3), f"Expected h shape (8,3,3), got {h.shape}"
    assert isinstance(P, float), f"Expected P to be float, got {type(P)}"
    assert P >= 0.0, "Power should be non-negative"


def test_run_pipeline_zero_fields():
    """Pipeline should return zero strain and power for zero fields."""
    nt, nx, ny, nz = 5, 3, 3, 2
    grid = {"x": np.linspace(-1, 1, nx), "y": np.linspace(-1, 1, ny), "z": np.linspace(-1, 1, nz)}

    E = np.zeros((nt, nx, ny, nz, 3))
    B = np.zeros((nt, nx, ny, nz, 3))

    Q, h, P = run_pipeline(grid, E, B, dt=0.01, R=10.0)

    assert np.allclose(Q, 0.0), "Quadrupole should be zero for zero fields"
    assert np.allclose(h, 0.0), "Strain should be zero for zero fields"
    assert np.allclose(P, 0.0), "Power should be zero for zero fields"
