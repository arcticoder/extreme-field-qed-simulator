"""
Simple EM source models producing time-dependent energy-density distributions for GW estimates.

These are not full Maxwell solvers; they provide analytic/toy field patterns amenable to quadrupole evaluation.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from .constants import c, epsilon0, mu0


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float

    @property
    def coords(self) -> np.ndarray:
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        return np.stack([X, Y, Z], axis=-1)

    @property
    def positions_flat(self) -> np.ndarray:
        return self.coords.reshape(-1, 3)

    @property
    def dV(self) -> float:
        return self.dx * self.dy * self.dz


def make_grid(L: float = 0.1, N: int = 21) -> Grid:
    x = np.linspace(-L/2, L/2, N)
    return Grid(x=x, y=x, z=x, dx=x[1]-x[0], dy=x[1]-x[0], dz=x[1]-x[0])


def interfering_pulses_energy(grid: Grid, E0: float, wavelength: float, omega: float, t: float) -> np.ndarray:
    """Two counter-propagating linearly polarized plane waves along x produce a standing wave.
    E(x,t) = 2 E0 cos(k x) cos(ω t) ê_y, B(x,t) = (2 E0/c) sin(k x) sin(ω t) ê_z (phase-shifted to conserve energy flow)
    Returns energy density u(x,y,z,t) [J/m^3] over grid.
    """
    k = 2.0 * np.pi / wavelength
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    Ey = 2.0 * E0 * np.cos(k * X) * np.cos(omega * t)
    Bz = 2.0 * (E0 / c) * np.sin(k * X) * np.sin(omega * t)
    u = 0.5 * (epsilon0 * Ey**2 + (Bz**2) / mu0)
    return u


def rotating_quadrupole_energy(grid: Grid, U0: float, R0: float, omega: float, t: float) -> np.ndarray:
    """Toy rotating quadrupole: put four Gaussian hotspots at radius R0, rotating at ω.
    Approximates a ring-lattice intensity pattern. U0 is peak energy density [J/m^3].
    """
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    theta = omega * t
    centers = []
    for base in [0.0, 0.5*np.pi, np.pi, 1.5*np.pi]:
        a = base + theta
        cx, cy = R0 * np.cos(a), R0 * np.sin(a)
        centers.append((cx, cy))
    sigma2 = (0.2*R0 + 1e-6)**2
    u = np.zeros_like(X)
    for (cx, cy) in centers:
        r2 = (X - cx)**2 + (Y - cy)**2 + Z**2
        u += U0 * np.exp(-r2 / (2.0 * sigma2))
    return u
