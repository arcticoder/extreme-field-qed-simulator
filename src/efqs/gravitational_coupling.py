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


# === API wrappers for script compatibility ===


def quadrupole_moment(positions: np.ndarray, energy_elements_J: np.ndarray) -> np.ndarray:
        """Wrapper around compute_quadrupole for position-based API.

        Inputs:
            positions: array shape (N, 3) of (x,y,z) coordinates OR dict with 'x','y','z' keys
            energy_elements_J: array shape (N,) in Joules (already u * dV from caller)
        Returns:
            Q: numpy array shape (3, 3) for single snapshot
        """
        if isinstance(positions, dict):
                # Dict-based grid interface
                return compute_quadrupole(positions, energy_elements_J)

        # Array-based interface: positions (N,3), energy_elements (N,)
        # Compute Q_ij = sum_k (u_k / c^2) * (x_i^k x_j^k - 1/3 delta_ij r_k^2)
        rho = energy_elements_J / (c ** 2)  # mass-equivalent
        coords = positions  # (N, 3)
        r2 = np.sum(coords ** 2, axis=1)  # (N,)
        Q = np.zeros((3, 3), dtype=float)
        eye = np.eye(3)
        for k in range(len(rho)):
                Q += rho[k] * (
                        np.outer(coords[k], coords[k]) - (r2[k] / 3.0) * eye
                )
        return Q


def strain_far_field(
        Q_t: np.ndarray, 
        dt: float, 
        R: float, 
        use_tt: bool = True, 
        use_spectral: bool = True,
        line_of_sight: np.ndarray | None = None
) -> np.ndarray:
        """Compute far-field strain h_ij from quadrupole Q_ij(t).

        Inputs:
            Q_t: shape (nt, 3, 3)
            dt: timestep [s]
            R: distance [m]
            use_tt: if True, apply TT projection
            use_spectral: if True, use spectral derivatives
            line_of_sight: 3-vector pointing from source to observer (default: [0,0,1] = z-axis)
                          Will be automatically normalized if provided.
        Returns:
            h: shape (nt, 3, 3)
        """
        if use_spectral:
                Qdd = spectral_derivative(Q_t, dt, order=2)
        else:
                # Fallback finite difference 2nd deriv (centered)
                Qdd = np.zeros_like(Q_t)
                Qdd[1:-1] = (Q_t[2:] - 2 * Q_t[1:-1] + Q_t[:-2]) / (dt ** 2)
        pref = 2.0 * G / (c ** 4 * R)
        h = pref * Qdd

        if use_tt:
                # TT projection: for a given line-of-sight n,
                # P_ij = delta_ij - n_i n_j, h_TT = P h P - 1/2 P Tr(Ph)
                if line_of_sight is None:
                        n = np.array([0.0, 0.0, 1.0])
                else:
                        n = np.asarray(line_of_sight, dtype=float)
                        n = n / np.linalg.norm(n)  # normalize
                P = np.eye(3) - np.outer(n, n)
                for t in range(h.shape[0]):
                        Ph = P @ h[t] @ P
                        trace_Ph = np.trace(Ph)
                        h[t] = Ph - 0.5 * P * trace_Ph
        return h


def radiated_power_from_quadrupole(Q_t: np.ndarray, dt: float, use_spectral: bool = True) -> np.ndarray:
        """Compute instantaneous radiated GW power from quadrupole.

        Inputs:
            Q_t: shape (nt, 3, 3)
            dt: timestep
            use_spectral: if True, use spectral 3rd derivative
        Returns:
            P_t: array shape (nt,) [W] (instantaneous power at each time)
        """
        if use_spectral:
                Qddd = spectral_derivative(Q_t, dt, order=3)
        else:
                # Fallback finite-difference 3rd deriv
                Qddd = np.zeros_like(Q_t)
                # 4-point centered: f'''(t) ≈ [-f(t-2h) + 2f(t-h) - 2f(t+h) + f(t+2h)]/(2h^3)
                Qddd[2:-2] = (
                        -Q_t[:-4] + 2 * Q_t[1:-3] - 2 * Q_t[3:-1] + Q_t[4:]
                ) / (2 * dt ** 3)
        inst = np.sum(Qddd * Qddd, axis=(1, 2))
        P_t = (G / (5.0 * c ** 5)) * inst
        return P_t


def dominant_frequency(
        series: np.ndarray, dt: float, component: int | tuple | None = None
) -> dict[str, float]:
        """Extract dominant frequency and bandwidth from time series via FFT.

        Inputs:
            series: shape (nt,) or (nt, ...) for h_+, h_x, or matrix components
            dt: timestep [s]
            component: if series is multidimensional, index to extract (e.g., (0,0) for h_xx)
        Returns:
            dict with keys: 'peak_freq' [Hz], 'peak_amplitude', 'bandwidth_3dB' [Hz]
        """
        if component is not None:
                if isinstance(component, tuple):
                        s = series[(slice(None),) + component]
                else:
                        s = series[:, component]
        else:
                # Assume 1D or take first component
                s = series.ravel() if series.ndim == 1 else series[:, 0, 0]

        nt = len(s)
        freqs = np.fft.rfftfreq(nt, d=dt)
        fft_s = np.fft.rfft(s)
        psd = np.abs(fft_s) ** 2

        idx_peak = int(np.argmax(psd[1:])) + 1  # skip DC
        peak_freq = float(freqs[idx_peak])
        peak_amp = float(np.sqrt(psd[idx_peak]))

        # -3 dB bandwidth: find indices where psd drops to half peak
        half_power = psd[idx_peak] / 2.0
        left_idx = idx_peak
        while left_idx > 1 and psd[left_idx] > half_power:
                left_idx -= 1
        right_idx = idx_peak
        while right_idx < len(psd) - 1 and psd[right_idx] > half_power:
                right_idx += 1
        bw_3dB = float(freqs[right_idx] - freqs[left_idx])

        return {"peak_freq_Hz": peak_freq, "peak_amplitude": peak_amp, "bandwidth_Hz": bw_3dB}


def stress_energy_from_fields(
        E: np.ndarray, B: np.ndarray, include_qed: bool = False
) -> np.ndarray:
        """Compute EM stress-energy T^{00} = (ε_0 E^2 + B^2/μ_0)/2.

        Inputs:
            E, B: arrays shape (..., 3)
            include_qed: if True, apply Heisenberg–Euler corrections (placeholder: not implemented)
        Returns:
            T00: array with shape matching E[..., 0]
        """
        if include_qed:
                # Placeholder for QED corrections; for now just return classical
                pass
        return em_energy_density(E, B)
