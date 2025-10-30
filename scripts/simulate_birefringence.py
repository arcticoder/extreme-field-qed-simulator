#!/usr/bin/env python3
from __future__ import annotations
import argparse
from efqs.vacuum_birefringence import phase_retardation, induced_ellipticity


def main():
    p = argparse.ArgumentParser(description="Vacuum birefringence: phase and ellipticity estimate")
    p.add_argument("--E0", type=float, default=0.0, help="Static electric field [V/m] (effective)")
    p.add_argument("--B0", type=float, default=0.0, help="Static magnetic field [T] (effective)")
    p.add_argument("--L", type=float, default=1.0, help="Path length through field [m]")
    p.add_argument("--lambda0", type=float, default=532e-9, help="Probe wavelength [m]")
    p.add_argument("--theta", type=float, default=45.0, help="Probe polarization angle [deg]")
    p.add_argument("--cavity-gain", type=float, default=1.0, help="Phase build-up factor (power cavity)")
    args = p.parse_args()

    dphi = phase_retardation(E0=args.E0, B0=args.B0, L=args.L, lambda0=args.lambda0)
    dphi_eff = dphi * args.cavity_gain
    psi = induced_ellipticity(args.theta, dphi_eff)

    print("Vacuum birefringence estimate (Heisenberg–Euler, leading order):")
    print(f"  E0 = {args.E0:.3e} V/m, B0 = {args.B0:.3e} T, L = {args.L} m, λ = {args.lambda0:.3e} m")
    print(f"  Phase retardation δ (single pass): {dphi:.3e} rad")
    if args.cavity_gain != 1.0:
        print(f"  Cavity-enhanced δ: {dphi_eff:.3e} rad (gain={args.cavity_gain:.3e})")
    print(f"  Induced ellipticity ψ ≈ {(psi):.3e} rad at θ={args.theta}°")


if __name__ == "__main__":
    main()
