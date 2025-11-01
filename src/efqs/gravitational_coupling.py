"""
Compute weak-field gravitational emission (quadrupole approximation)
from EM stress-energy outputs produced by the EFQS code.

Expect inputs:
 - grid: dict with 'x','y','z' 1D arrays or meshgrid-like shapes
 - E: numpy array shape (nt, nx, ny, nz, 3)
 - B: numpy array shape (nt, nx, ny, nz, 3)
 - dt: time step (s)
Outputs:
 - Q_tt: quadrupole moment tensor Q_ij(t)
 - h_tt: metric perturbation components h_ij(t) at distance R
 - P_GW: time-averaged radiated GW power (scalar)
"""
from __future__ import annotations

import numpy as np

# physical constants (SI)
G = 6.67430e-11
c = 299792458.0
eps0 = 8.8541878128e-12
mu0 = 1.0 / (eps0 * c * c)


def em_energy_density(E: np.ndarray, B: np.ndarray) -> np.ndarray:
        """u_EM = 0.5*(eps0*E^2 + B^2/mu0)

        E, B arrays shape (..., 3)
        Returns array with shape matching E[..., 0].
        """
        E2 = np.sum(E * E, axis=-1)
        B2 = np.sum(B * B, axis=-1)
        return 0.5 * (eps0 * E2 + B2 / mu0)


def compute_quadrupole(grid: dict, u_em: np.ndarray) -> np.ndarray:
        """
        Compute Q_ij(t) = ∫ rho_eff(x,t) (x_i x_j - 1/3 δ_ij r^2) dV
        where rho_eff = u_em / c^2.

        Inputs:
            grid: dict with 'x','y','z' 1D arrays
            u_em: numpy array shape (nt, nx, ny, nz)
        Returns:
            Q: numpy array shape (nt, 3, 3)
        """
        x = np.asarray(grid["x"])  # (nx,)
        y = np.asarray(grid["y"])  # (ny,)
        z = np.asarray(grid["z"])  # (nz,)

        # Estimate cell volume from coordinate gradients (assume regular grid)
        dx = np.gradient(x)
        dy = np.gradient(y)
        dz = np.gradient(z)
        dV = float(dx[0] * dy[0] * dz[0])

        # Spatial coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        coords = np.stack([X, Y, Z], axis=-1)  # (nx, ny, nz, 3)
        r2 = X * X + Y * Y + Z * Z

        nt = int(u_em.shape[0])
        Q = np.zeros((nt, 3, 3), dtype=float)
        rho = u_em / (c * c)

        # Precompute trace part factor tensor (nx,ny,nz,3,3)
        eye = np.eye(3, dtype=float)
        trace_term = (r2[..., None, None] / 3.0) * eye  # broadcast

        for t in range(nt):
                # integrand: rho * (x_i x_j - 1/3 δ_ij r^2)
                # coords[..., None, :] * coords[..., :, None] -> (nx,ny,nz,3,3)
                quad_kernel = coords[..., None, :] * coords[..., :, None] - trace_term
                I = rho[t, ...][..., None, None] * quad_kernel
                Q[t, :, :] = np.sum(I, axis=(0, 1, 2)) * dV
        return Q


def spectral_derivative(time_series: np.ndarray, dt: float, order: int = 1, pad_factor: float = 2.0) -> np.ndarray:
        """Compute time derivative using FFT differentiation.

        time_series shape (nt, ...), returns same shape.
        """
        nt = int(time_series.shape[0])
        nfft = int(2 ** np.ceil(np.log2(max(1, int(nt * pad_factor)))))
        freqs = np.fft.rfftfreq(nfft, d=dt)
        Tpad = np.fft.rfft(time_series, n=nfft, axis=0)
        fac = (1j * 2 * np.pi * freqs) ** order
        # Multiply along frequency axis
        Tpad *= fac[(slice(None),) + (None,) * (Tpad.ndim - 1)]
        deriv = np.fft.irfft(Tpad, n=nfft, axis=0)[:nt, ...]
        return deriv.real


def compute_h_and_power(Q: np.ndarray, dt: float, R: float = 10.0) -> tuple[np.ndarray, float]:
        """
        Given Q_ij(t), compute
            h_ij(t) = (2G/(c^4 R)) * d^2 Q_ij / dt^2 (TT projection omitted)
        and estimate radiated power via
            P = (G/(5 c^5)) < d^3Q/dt^3 : d^3Q/dt^3 > (time average)
        """
        # second and third derivatives
        Qdd = spectral_derivative(Q, dt, order=2)  # (nt,3,3)
        Qddd = spectral_derivative(Q, dt, order=3)
        pref = 2.0 * G / (c ** 4 * R)
        h = pref * Qdd  # Not TT-projected; order-of-magnitude estimate

        # instantaneous radiated power integrand: contract Qddd_ij Qddd_ij
        inst = np.sum(Qddd * Qddd, axis=(1, 2))
        P = float((G / (5.0 * c ** 5)) * np.mean(inst))
        return h, P


def run_pipeline(grid: dict, E: np.ndarray, B: np.ndarray, dt: float, R: float = 10.0):
        """
        Inputs:
            E,B: arrays shape (nt, nx, ny, nz, 3)
            dt: timestep [s]
            R: observer distance [m]
        Returns: Q(t), h(t), P_GW
        """
        u = em_energy_density(E, B)  # (nt,nx,ny,nz)
        Q = compute_quadrupole(grid, u)
        h, P = compute_h_and_power(Q, dt, R=R)
        return Q, h, P
