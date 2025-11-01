"""
Demonstration of gravitational coupling module.

This script shows how to use the quadrupole approximation to compute
gravitational wave strain h(t) and radiated power P_GW from EM field configurations.
"""
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from efqs.gravitational_coupling import run_pipeline


def demo_oscillating_dipole():
    """
    Demo: Simple oscillating electric dipole field.
    
    Creates a time-varying E field oscillating at 10 Hz, computes resulting
    gravitational wave strain and radiated power.
    """
    print("=" * 70)
    print("Demo: Oscillating Electric Dipole → Gravitational Waves")
    print("=" * 70)
    
    # Setup grid
    nt, nx, ny, nz = 64, 8, 8, 6
    x = np.linspace(-0.1, 0.1, nx)  # meters
    y = np.linspace(-0.1, 0.1, ny)
    z = np.linspace(-0.1, 0.1, nz)
    grid = {"x": x, "y": y, "z": z}
    
    # Time parameters
    dt = 1e-3  # 1 ms time step
    t = np.arange(nt) * dt
    freq = 10.0  # Hz
    
    # Create oscillating E field (dipole-like pattern)
    E = np.zeros((nt, nx, ny, nz, 3), dtype=float)
    B = np.zeros_like(E)
    
    # E_x varies sinusoidally in time and linearly in space (dipole)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    for ti in range(nt):
        amplitude = 1e6 * np.sin(2 * np.pi * freq * t[ti])  # V/m
        E[ti, :, :, :, 0] = amplitude * X / 0.1  # Linear gradient
    
    print(f"\nConfiguration:")
    print(f"  Grid: {nx}×{ny}×{nz} cells, Δx ≈ {(x[1]-x[0])*1e3:.1f} mm")
    print(f"  Time: {nt} steps, Δt = {dt*1e3:.1f} ms, f = {freq} Hz")
    print(f"  E field: peak amplitude {np.max(np.abs(E)):.2e} V/m")
    
    # Run gravitational coupling pipeline
    print(f"\nComputing quadrupole moment Q(t), strain h(t), and power P_GW...")
    R = 10.0  # Observer distance in meters
    Q, h, P = run_pipeline(grid, E, B, dt, R=R)
    
    # Analyze results
    h_rms = np.sqrt(np.mean(h**2))
    h_max = np.max(np.abs(h))
    Q_rms = np.sqrt(np.mean(Q**2))
    
    print(f"\nResults:")
    print(f"  Quadrupole Q_rms: {Q_rms:.3e} kg·m²")
    print(f"  Strain h_rms: {h_rms:.3e} (dimensionless)")
    print(f"  Strain h_max: {h_max:.3e}")
    print(f"  Radiated power P_GW: {P:.3e} W")
    print(f"  Observer distance: {R} m")
    
    # Estimate characteristic frequency
    h_fft = np.fft.rfft(h[:, 0, 0], axis=0)
    freqs = np.fft.rfftfreq(nt, d=dt)
    peak_idx = np.argmax(np.abs(h_fft))
    peak_freq = freqs[peak_idx]
    print(f"  Dominant h frequency: {peak_freq:.1f} Hz (expected ~{2*freq} Hz for quadrupole)")
    
    print("\n✓ Pipeline executed successfully\n")
    return Q, h, P


def demo_rotating_charge_distribution():
    """
    Demo: Rotating charge distribution (simplified model).
    
    Models a simple rotating quadrupole pattern to show time-varying
    gravitational wave emission.
    """
    print("=" * 70)
    print("Demo: Rotating Quadrupole → Gravitational Waves")
    print("=" * 70)
    
    # Smaller grid for faster computation
    nt, nx, ny, nz = 32, 6, 6, 4
    x = np.linspace(-0.05, 0.05, nx)
    y = np.linspace(-0.05, 0.05, ny)
    z = np.linspace(-0.05, 0.05, nz)
    grid = {"x": x, "y": y, "z": z}
    
    dt = 5e-4
    t = np.arange(nt) * dt
    omega = 20.0  # rad/s
    
    E = np.zeros((nt, nx, ny, nz, 3), dtype=float)
    B = np.zeros_like(E)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    
    # Create rotating quadrupole pattern
    for ti in range(nt):
        theta = omega * t[ti]
        # E field rotates in x-y plane
        E[ti, :, :, :, 0] = 5e5 * (X * np.cos(theta) - Y * np.sin(theta))
        E[ti, :, :, :, 1] = 5e5 * (X * np.sin(theta) + Y * np.cos(theta))
    
    print(f"\nConfiguration:")
    print(f"  Grid: {nx}×{ny}×{nz} cells")
    print(f"  Rotation frequency: {omega/(2*np.pi):.1f} Hz")
    print(f"  Time steps: {nt}, dt = {dt*1e3:.2f} ms")
    
    R = 5.0
    Q, h, P = run_pipeline(grid, E, B, dt, R=R)
    
    h_rms = np.sqrt(np.mean(h**2))
    print(f"\nResults:")
    print(f"  Strain h_rms: {h_rms:.3e}")
    print(f"  Radiated power P_GW: {P:.3e} W")
    print(f"  Observer distance: {R} m")
    
    print("\n✓ Pipeline executed successfully\n")
    return Q, h, P


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Gravitational Coupling Module Demonstration")
    print("=" * 70 + "\n")
    
    # Run demos
    Q1, h1, P1 = demo_oscillating_dipole()
    Q2, h2, P2 = demo_rotating_charge_distribution()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nBoth demonstrations completed successfully.")
    print("\nKey takeaways:")
    print("  • EM energy density u = ½(ε₀E² + B²/μ₀)")
    print("  • Quadrupole Q_ij(t) computed from ∫ ρ(x) x_i x_j d³x")
    print("  • Strain h_ij ≈ (2G/c⁴R) d²Q_ij/dt² (order of magnitude)")
    print("  • Power P_GW = (G/5c⁵) ⟨d³Q_ij/dt³ · d³Q_ij/dt³⟩")
    print("  • Spectral derivatives (FFT) used for stability")
    print("\nFor production use:")
    print("  • Increase grid resolution for better accuracy")
    print("  • Use longer time series (nt >> 100) for clean derivatives")
    print("  • Ensure dt resolves highest modulation frequency")
    print("  • Add TT projection for precise waveform analysis")
    print("=" * 70 + "\n")
