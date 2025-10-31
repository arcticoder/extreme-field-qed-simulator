"""
Canonical source geometries for systematic GW coupling experiments.

Implements realistic EM source configurations:
- Colliding/interfering laser pulses (focused beams)
- High-Q optical cavity modes
- Plasma toroids and counter-rotating ring currents
- Rotating asymmetric capacitors (Biefeld-Brown geometry)

Each geometry provides E(x,t), B(x,t) fields on a regular grid,
suitable for stress-energy and quadrupole computations.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from .constants import c, epsilon0, mu0, E_s


@dataclass
class SourceGeometry:
    """Container for source configuration and field evaluator."""
    name: str
    description: str
    field_evaluator: callable  # (grid, t, params) -> (E, B)
    characteristic_scale: float  # Typical length scale [m]
    characteristic_time: float  # Typical time scale [s]


def colliding_gaussian_pulses(grid_positions: np.ndarray, t: float,
                              E0: float, waist: float, wavelength: float,
                              pulse_duration: float, collision_delay: float = 0.0,
                              polarization: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """Two focused Gaussian pulses counter-propagating along z-axis.
    
    Creates intense hotspot at collision (z=0, t=collision_delay).
    Ideal for QED-regime stress-energy concentration.
    
    Parameters:
    -----------
    grid_positions : (N, 3) array of coordinates [m]
    t : time [s]
    E0 : peak electric field [V/m]
    waist : beam waist at focus [m]
    wavelength : wavelength [m]
    pulse_duration : temporal FWHM [s]
    collision_delay : time of collision at z=0 [s]
    polarization : 'linear', 'circular', or 'orthogonal'
    
    Returns:
    --------
    E, B : (N, 3) arrays of fields [V/m], [T]
    """
    pos = np.asarray(grid_positions, dtype=float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
    k = 2.0 * np.pi / wavelength
    omega = 2.0 * np.pi * c / wavelength
    z_R = np.pi * waist**2 / wavelength  # Rayleigh range
    
    # Transverse coordinate
    r_perp = np.sqrt(x**2 + y**2)
    
    # Pulse 1: propagating +z (from z=-∞)
    z1_rel = z
    w1 = waist * np.sqrt(1.0 + (z1_rel / z_R)**2)
    R1 = z1_rel * (1.0 + (z_R / (z1_rel + 1e-12))**2)
    gouy1 = np.arctan(z1_rel / z_R)
    
    envelope1 = (waist / w1) * np.exp(-(r_perp / w1)**2)
    phase1 = k * z1_rel - omega * (t - collision_delay) + k * r_perp**2 / (2.0 * R1) - gouy1
    temporal1 = np.exp(-((t - collision_delay - z1_rel/c) / pulse_duration)**2)
    
    E1_magnitude = E0 * envelope1 * temporal1 * np.cos(phase1)
    
    # Pulse 2: propagating -z (from z=+∞)
    z2_rel = -z
    w2 = waist * np.sqrt(1.0 + (z2_rel / z_R)**2)
    R2 = z2_rel * (1.0 + (z_R / (z2_rel + 1e-12))**2)
    gouy2 = np.arctan(z2_rel / z_R)
    
    envelope2 = (waist / w2) * np.exp(-(r_perp / w2)**2)
    phase2 = -k * z2_rel - omega * (t - collision_delay) + k * r_perp**2 / (2.0 * R2) - gouy2
    temporal2 = np.exp(-((t - collision_delay + z2_rel/c) / pulse_duration)**2)
    
    E2_magnitude = E0 * envelope2 * temporal2 * np.cos(phase2)
    
    # Combine fields depending on polarization
    E = np.zeros((len(pos), 3))
    B = np.zeros((len(pos), 3))
    
    if polarization == 'linear':
        # Both polarized along x
        E[:, 0] = E1_magnitude + E2_magnitude
        # B fields orthogonal to E and k
        B[:, 1] = (E1_magnitude - E2_magnitude) / c
    
    elif polarization == 'orthogonal':
        # Pulse 1 along x, pulse 2 along y
        E[:, 0] = E1_magnitude
        E[:, 1] = E2_magnitude
        B[:, 1] = E1_magnitude / c
        B[:, 0] = -E2_magnitude / c
    
    elif polarization == 'circular':
        # Circular polarization (complex superposition)
        E[:, 0] = E1_magnitude + E2_magnitude * np.cos(np.pi/2)
        E[:, 1] = E2_magnitude * np.sin(np.pi/2)
        B[:, 1] = E1_magnitude / c
        B[:, 0] = -E2_magnitude / c
    
    return E, B


def cavity_standing_wave(grid_positions: np.ndarray, t: float,
                         mode_numbers: Tuple[int, int, int],
                         cavity_length: float, Q_factor: float,
                         stored_energy: float, modulation_freq: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """High-Q optical cavity standing wave mode.
    
    Creates spatially coherent EM energy distribution with potentially
    huge photon numbers. Time-modulation via cavity detuning or rotating mirrors
    creates quadrupole time-derivatives.
    
    Parameters:
    -----------
    mode_numbers : (nx, ny, nz) mode integers
    cavity_length : cavity dimension [m]
    Q_factor : quality factor (determines stored energy lifetime)
    stored_energy : total energy in cavity [J]
    modulation_freq : external modulation frequency [Hz] for quadrupole variation
    
    Returns:
    --------
    E, B : field arrays
    """
    pos = np.asarray(grid_positions, dtype=float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
    nx, ny, nz = mode_numbers
    L = cavity_length
    
    # Mode frequencies (box normalization)
    omega_mode = c * np.pi * np.sqrt((nx/L)**2 + (ny/L)**2 + (nz/L)**2)
    
    # Standing wave pattern
    E_mode_shape = (np.sin(nx * np.pi * (x + L/2) / L) *
                    np.sin(ny * np.pi * (y + L/2) / L) *
                    np.sin(nz * np.pi * (z + L/2) / L))
    
    # Cavity decay
    decay = np.exp(-omega_mode * t / (2.0 * Q_factor))
    
    # External modulation
    if modulation_freq > 0:
        modulation = 1.0 + 0.1 * np.cos(2.0 * np.pi * modulation_freq * t)
    else:
        modulation = 1.0
    
    # Normalize to stored energy
    # E² ~ energy / (epsilon0 * volume)
    volume = L**3
    E0 = np.sqrt(2.0 * stored_energy / (epsilon0 * volume))
    
    # Time oscillation
    E_amplitude = E0 * E_mode_shape * decay * modulation * np.cos(omega_mode * t)
    
    # Assign to components (TEM-like mode, E along one direction)
    E = np.zeros((len(pos), 3))
    B = np.zeros((len(pos), 3))
    
    E[:, 0] = E_amplitude  # Polarized along x
    
    # B perpendicular, phase-shifted by 90°
    B[:, 1] = -(E0 / c) * E_mode_shape * decay * modulation * np.sin(omega_mode * t)
    
    return E, B


def plasma_toroid_currents(grid_positions: np.ndarray, t: float,
                           major_radius: float, minor_radius: float,
                           current: float, rotation_freq: float,
                           direction: str = 'cw') -> Tuple[np.ndarray, np.ndarray]:
    """Counter-rotating toroidal plasma currents (tokamak-like geometry).
    
    Generates time-varying magnetic quadrupole from rotating current distribution.
    Can model two counter-rotating streams to enhance quadrupole asymmetry.
    
    Parameters:
    -----------
    major_radius : R, torus major radius [m]
    minor_radius : a, torus minor (tube) radius [m]
    current : total toroidal current [A]
    rotation_freq : rotation frequency [Hz] (for rigid-body rotation)
    direction : 'cw' or 'ccw' rotation
    
    Returns:
    --------
    E, B : field arrays (E mostly zero, B from current distribution)
    """
    pos = np.asarray(grid_positions, dtype=float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
    omega_rot = 2.0 * np.pi * rotation_freq
    theta_rot = omega_rot * t
    if direction == 'ccw':
        theta_rot = -theta_rot
    
    # Torus centerline rotates in x-y plane
    # Current flows toroidally (φ direction in cylindrical coords)
    
    # For simplicity, model as discrete current loop at major radius
    # Biot-Savart for circular loop rotated
    
    # Loop center position
    x_center = major_radius * np.cos(theta_rot)
    y_center = major_radius * np.sin(theta_rot)
    z_center = 0.0
    
    # Distance from loop center
    dx = x - x_center
    dy = y - y_center
    dz = z - z_center
    r_vec = np.stack([dx, dy, dz], axis=-1)
    r_mag = np.linalg.norm(r_vec, axis=-1, keepdims=True) + 1e-12
    
    # Current element direction (tangent to toroid, φ-direction)
    # For loop in x-y plane, current flows perpendicular to radial
    current_dir = np.stack([-np.sin(theta_rot), np.cos(theta_rot), np.zeros_like(x)], axis=-1)
    
    # Biot-Savart: B = (μ0 I / 4π) ∫ dl × r / r³
    # Approximate as concentrated loop
    cross_product = np.cross(current_dir, r_vec)
    
    B = (mu0 * current / (4.0 * np.pi)) * cross_product / r_mag**3
    
    # Smooth with Gaussian envelope in minor radius
    envelope = np.exp(-(r_mag.squeeze() - major_radius)**2 / (2.0 * minor_radius**2))
    B *= envelope[:, np.newaxis]
    
    # E field from time-varying B (Faraday's law, approximate)
    # For rotating current, E ~ ω R B (order of magnitude)
    E = omega_rot * major_radius * np.cross(np.array([0., 0., 1.]), B)
    
    return E, B


def rotating_asymmetric_capacitor(grid_positions: np.ndarray, t: float,
                                  radius: float, separation: float,
                                  voltage: float, rotation_freq: float,
                                  num_sectors: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Rotating asymmetric capacitor (Biefeld-Brown inspired geometry).
    
    Parallel-plate capacitor with asymmetric charge/voltage distribution,
    rotating about central axis. Creates time-varying electric quadrupole.
    
    Parameters:
    -----------
    radius : plate radius [m]
    separation : plate separation [m]
    voltage : applied voltage [V]
    rotation_freq : rotation frequency [Hz]
    num_sectors : number of asymmetric sectors (default 4 for quadrupole)
    
    Returns:
    --------
    E, B : field arrays
    """
    pos = np.asarray(grid_positions, dtype=float)
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    
    omega_rot = 2.0 * np.pi * rotation_freq
    theta_rot = omega_rot * t
    
    # Rotate coordinates
    x_rot = x * np.cos(theta_rot) + y * np.sin(theta_rot)
    y_rot = -x * np.sin(theta_rot) + y * np.cos(theta_rot)
    
    # Cylindrical coords
    r_cyl = np.sqrt(x_rot**2 + y_rot**2)
    phi = np.arctan2(y_rot, x_rot)
    
    # Asymmetric potential pattern (sectoral)
    # e.g., ± voltage in alternating sectors
    sector_angle = 2.0 * np.pi / num_sectors
    sector_index = np.floor(phi / sector_angle).astype(int)
    voltage_pattern = voltage * ((-1) ** sector_index)
    
    # Electric field between plates (z-direction, uniform approximation)
    E = np.zeros((len(pos), 3))
    between_plates = (z >= -separation/2) & (z <= separation/2) & (r_cyl <= radius)
    
    E[between_plates, 2] = voltage_pattern[between_plates] / separation
    
    # Magnetic field from time-varying E (displacement current)
    # ∇ × B = μ0 ε0 ∂E/∂t
    # B ~ (μ0 ε0 / r) ∂E/∂t ~ (1/c²r) ω E (order of magnitude)
    
    B = np.zeros((len(pos), 3))
    # Azimuthal B field from radial E time-variation
    # Crude approximation: B_φ ~ (r/c²) ω E_z
    B[between_plates, 0] = -(y_rot[between_plates] / (c**2)) * omega_rot * E[between_plates, 2]
    B[between_plates, 1] = (x_rot[between_plates] / (c**2)) * omega_rot * E[between_plates, 2]
    
    return E, B


# Registry of predefined geometries
GEOMETRIES = {
    "colliding_pulses": SourceGeometry(
        name="CollidingGaussianPulses",
        description="Counter-propagating focused laser pulses creating QED hotspot",
        field_evaluator=colliding_gaussian_pulses,
        characteristic_scale=1e-6,  # μm focal spot
        characteristic_time=1e-15  # fs pulse duration
    ),
    "cavity_mode": SourceGeometry(
        name="CavityStandingWave",
        description="High-Q optical cavity with large stored photon number",
        field_evaluator=cavity_standing_wave,
        characteristic_scale=0.1,  # cm cavity
        characteristic_time=1e-9  # ns cavity lifetime / modulation
    ),
    "plasma_toroid": SourceGeometry(
        name="PlasmaToroidCurrents",
        description="Rotating toroidal plasma current distribution",
        field_evaluator=plasma_toroid_currents,
        characteristic_scale=0.1,  # 10 cm major radius
        characteristic_time=1e-6  # μs rotation period
    ),
    "rotating_capacitor": SourceGeometry(
        name="RotatingAsymmetricCapacitor",
        description="Biefeld-Brown rotating asymmetric capacitor geometry",
        field_evaluator=rotating_asymmetric_capacitor,
        characteristic_scale=0.1,  # 10 cm diameter
        characteristic_time=1e-3  # ms rotation period
    ),
}
