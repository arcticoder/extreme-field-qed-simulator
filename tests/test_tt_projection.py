"""Unit tests for TT projection identities in gravitational_coupling.py"""
import numpy as np
import pytest
from efqs.gravitational_coupling import strain_far_field


def test_tt_projection_traceless():
    """TT-projected strain should be traceless."""
    # Create a simple quadrupole moment time series with non-zero trace
    nt = 10
    Q_t = np.zeros((nt, 3, 3))
    for t in range(nt):
        Q_t[t] = np.array([
            [1.0, 0.1, 0.0],
            [0.1, 2.0, 0.0],
            [0.0, 0.0, 3.0]
        ]) * np.sin(2 * np.pi * t / nt)
    
    dt = 1e-15
    R = 10.0
    h_tt = strain_far_field(Q_t, dt, R, use_tt=True, use_spectral=True)
    
    # Check trace at each timestep
    for t in range(nt):
        trace = np.trace(h_tt[t])
        assert abs(trace) < 1e-30, f"TT projection should be traceless, got trace={trace}"


def test_tt_projection_transverse():
    """TT-projected strain should be transverse to line of sight."""
    nt = 10
    Q_t = np.random.randn(nt, 3, 3)
    Q_t = (Q_t + Q_t.transpose(0, 2, 1)) / 2  # symmetrize
    
    dt = 1e-15
    R = 10.0
    
    # Test different line-of-sight directions
    for los in [np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([1, 1, 1])]:
        h_tt = strain_far_field(Q_t, dt, R, use_tt=True, line_of_sight=los)
        n = los / np.linalg.norm(los)
        
        # Check h_ij n^j = 0 (transverse condition)
        for t in range(nt):
            h_times_n = h_tt[t] @ n
            assert np.allclose(h_times_n, 0, atol=1e-28), \
                f"TT strain should be transverse, got h·n = {h_times_n}"


def test_tt_projection_symmetric():
    """TT-projected strain should remain symmetric."""
    nt = 10
    Q_t = np.random.randn(nt, 3, 3)
    Q_t = (Q_t + Q_t.transpose(0, 2, 1)) / 2  # start with symmetric Q
    
    dt = 1e-15
    R = 10.0
    h_tt = strain_far_field(Q_t, dt, R, use_tt=True)
    
    for t in range(nt):
        assert np.allclose(h_tt[t], h_tt[t].T), \
            f"TT strain should be symmetric at t={t}"


def test_line_of_sight_normalization():
    """Line of sight vector should be automatically normalized."""
    nt = 5
    Q_t = np.random.randn(nt, 3, 3)
    Q_t = (Q_t + Q_t.transpose(0, 2, 1)) / 2
    
    dt = 1e-15
    R = 10.0
    
    # Same direction, different magnitudes should give same result
    los1 = np.array([1, 1, 1])
    los2 = np.array([5, 5, 5])
    
    h1 = strain_far_field(Q_t, dt, R, use_tt=True, line_of_sight=los1)
    h2 = strain_far_field(Q_t, dt, R, use_tt=True, line_of_sight=los2)
    
    assert np.allclose(h1, h2), \
        "TT projection should be independent of line-of-sight magnitude"


def test_tt_vs_no_tt():
    """TT projection should produce traceless result while non-TT may have trace."""
    nt = 5
    # Create Q with definite non-zero trace structure
    Q_t = np.zeros((nt, 3, 3))
    for t in range(nt):
        Q_t[t] = np.diag([1.0, 2.0, 3.0]) * (1 + 0.1 * t)
    
    dt = 1e-15
    R = 10.0
    
    h_no_tt = strain_far_field(Q_t, dt, R, use_tt=False)
    h_tt = strain_far_field(Q_t, dt, R, use_tt=True)
    
    # Non-TT version may have trace; TT version should be traceless
    for t in range(nt):
        assert abs(np.trace(h_tt[t])) < 1e-30, \
            f"TT strain should be traceless at t={t}"
        # At least one timestep should have non-negligible trace in non-TT version
    has_trace = any(abs(np.trace(h_no_tt[t])) > 1e-30 for t in range(nt))
    assert has_trace, "Non-TT version should have non-zero trace for this test case"


def test_projection_operator_idempotent():
    """Applying TT projection twice should give same result as once."""
    nt = 5
    Q_t = np.random.randn(nt, 3, 3)
    Q_t = (Q_t + Q_t.transpose(0, 2, 1)) / 2
    
    dt = 1e-15
    R = 10.0
    
    # First projection
    h1 = strain_far_field(Q_t, dt, R, use_tt=True)
    
    # "Apply projection again" by treating h1 as if it were Q (hack for testing)
    # Actually we need to verify idempotency mathematically: P(P(h)) = P(h)
    # For this we manually apply projection
    n = np.array([0, 0, 1.0])
    P = np.eye(3) - np.outer(n, n)
    
    for t in range(nt):
        Ph = P @ h1[t] @ P
        h_twice = Ph - 0.5 * P * np.trace(Ph)
        # Should equal h1[t] since it's already TT
        assert np.allclose(h_twice, h1[t], atol=1e-25), \
            f"TT projection should be idempotent at t={t}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
