import numpy as np
from efqs.em_sources import make_grid, rotating_quadrupole_energy
from efqs.gravitational_coupling import quadrupole_moment, strain_far_field, radiated_power_from_quadrupole


def test_radiated_power_zero_for_static():
    grid = make_grid(L=0.02, N=9)
    dt = 1e-6
    steps = 10
    Q_list = []
    for n in range(steps):
        t = n * dt
        u = rotating_quadrupole_energy(grid, U0=1e6, R0=0.005, omega=0.0, t=t)
        Q = quadrupole_moment(grid.positions_flat, u.reshape(-1) * grid.dV)
        Q_list.append(Q)
    Q_t = np.stack(Q_list, axis=0)
    P = radiated_power_from_quadrupole(Q_t, dt)
    assert np.allclose(P, 0.0, atol=1e-25)


def test_strain_scales_with_distance():
    # Construct a simple oscillatory Q_t
    T = 100
    dt = 1e-6
    t = np.arange(T) * dt
    Q0 = np.zeros((3,3)); Q0[0,0] = 1.0
    Q_t = np.array([np.sin(2*np.pi*10*t_k) * Q0 for t_k in t])
    h1 = strain_far_field(Q_t, dt, R=1.0)
    h2 = strain_far_field(Q_t, dt, R=10.0)
    ratio = np.mean(np.abs(h1)) / np.mean(np.abs(h2))
    assert 9.5 < ratio < 10.5
