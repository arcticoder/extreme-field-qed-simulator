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


def gaussian_beam_energy(grid: Grid, P0: float, w0: float, wavelength: float, z_focus: float = 0.0, t: float = 0.0, omega: float = 0.0) -> np.ndarray:
    """Focused Gaussian beam energy density.
    
    Models a TEM00 Gaussian beam focused at z=z_focus with waist w0 and peak power P0.
    Energy density derived from intensity profile assuming CW or time-averaged fields.
    
    P0: peak power [W]
    w0: beam waist at focus [m]
    wavelength: wavelength [m]
    z_focus: z-position of beam focus [m]
    t: time [s] (for optional oscillation)
    omega: angular frequency [rad/s] for time modulation
    
    Returns u(x,y,z) [J/m^3] (time-averaged or instantaneous depending on omega).
    """
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    
    # Rayleigh range
    z_R = np.pi * w0**2 / wavelength
    
    # Distance from focus
    z_rel = Z - z_focus
    
    # Beam radius at position z
    w_z = w0 * np.sqrt(1.0 + (z_rel / z_R)**2)
    
    # Radial coordinate
    r2 = X**2 + Y**2
    
    # Intensity profile (assuming paraxial Gaussian beam)
    # I(r,z) = I0 (w0/w(z))^2 exp(-2 r^2 / w(z)^2)
    # where I0 = 2 P0 / (pi w0^2)
    I0 = 2.0 * P0 / (np.pi * w0**2)
    I = I0 * (w0 / w_z)**2 * np.exp(-2.0 * r2 / w_z**2)
    
    # Energy density u = I / c (time-averaged)
    u = I / c
    
    # Optional time modulation (e.g., for pulsed beam)
    if omega > 0:
        u *= np.abs(np.cos(omega * t))**2
    
    return u


def rotating_capacitor_energy(grid: Grid, V0: float, d: float, R: float, omega: float, t: float) -> np.ndarray:
    """Toy model for a rotating capacitor (Biefeld-Brown-like geometry).
    
    Models two parallel disks of radius R separated by distance d with voltage V0,
    rotating about the z-axis at angular frequency omega.
    The electric field is approximated as uniform between the plates.
    
    V0: voltage [V]
    d: plate separation [m]
    R: disk radius [m]
    omega: rotation angular frequency [rad/s]
    t: time [s]
    
    Returns u(x,y,z,t) [J/m^3] energy density.
    """
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    
    # Rotation angle
    theta = omega * t
    
    # Rotate coordinates (plates rotate in x-y plane)
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)
    
    # Plates at z = -d/2 and z = +d/2
    # Electric field E = V0/d in z-direction between plates
    E0 = V0 / d
    
    # Check if point is between plates and within disk radius
    r_xy = np.sqrt(X_rot**2 + Y_rot**2)
    between_plates = (Z >= -d/2) & (Z <= d/2)
    within_disk = r_xy <= R
    active = between_plates & within_disk
    
    # Energy density u = 1/2 epsilon0 E^2 within active region
    u = np.zeros_like(X)
    u[active] = 0.5 * epsilon0 * E0**2
    
    return u


def plasma_ring_energy(grid: Grid, I0: float, R: float, r_minor: float, omega: float, t: float, direction: str = 'cw') -> np.ndarray:
    """Counter-rotating plasma ring (toroidal current distribution).
    
    Models a toroidal plasma ring of major radius R and minor radius r_minor,
    carrying current I0, rotating at angular frequency omega.
    For counter-rotating configurations, call twice with direction='cw' and 'ccw'.
    
    The energy density is approximated from the magnetic field energy of a toroidal current loop.
    For simplicity, we model the energy density from the poloidal magnetic field:
    B_pol ~ μ0 I / (2π r) where r is distance from the ring center in the poloidal plane.
    
    I0: peak current [A]
    R: major radius of torus [m]
    r_minor: minor radius (thickness) of torus [m]
    omega: rotation angular frequency [rad/s]
    t: time [s]
    direction: 'cw' (clockwise) or 'ccw' (counter-clockwise) rotation
    
    Returns u(x,y,z,t) [J/m^3] energy density.
    """
    X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    
    # Rotation angle
    theta = omega * t
    if direction == 'ccw':
        theta = -theta
    
    # Ring center rotates in x-y plane
    ring_x = R * np.cos(theta)
    ring_y = R * np.sin(theta)
    ring_z = 0.0
    
    # Distance from ring center (toroidal coordinates simplified)
    # Approximate: distance in poloidal plane (perpendicular to ring)
    dx = X - ring_x
    dy = Y - ring_y
    dz = Z - ring_z
    
    # Cylindrical radius from ring centerline
    r_cyl = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Inside the torus minor radius?
    inside = r_cyl <= r_minor
    
    # Magnetic field magnitude (simplified toroidal approximation)
    # B ~ μ0 I / (2π r_cyl) for r_cyl < r_minor
    # Energy density u = B^2 / (2 μ0)
    B = np.zeros_like(X)
    epsilon = 1e-9  # avoid division by zero
    B[inside] = mu0 * I0 / (2.0 * np.pi * (r_cyl[inside] + epsilon))
    
    u = B**2 / (2.0 * mu0)
    
    # Smooth falloff to avoid sharp edges (Gaussian envelope)
    envelope = np.exp(-((r_cyl - r_minor/2)**2) / (2.0 * (r_minor/3)**2))
    u *= envelope
    
    return u

