#!/usr/bin/env python3
from __future__ import annotations
import argparse
from efqs.pair_production import expected_pairs


def main():
    p = argparse.ArgumentParser(description="Schwinger pair production estimate (uniform field)")
    p.add_argument("--E0", type=float, required=True, help="Electric field amplitude [V/m]")
    p.add_argument("--volume", type=float, required=True, help="Interaction volume [m^3]")
    p.add_argument("--duration", type=float, required=True, help="Pulse duration [s]")
    p.add_argument("--n-terms", type=int, default=1, help="Series terms in Schwinger sum (default: 1)")
    args = p.parse_args()

    N = expected_pairs(args.E0, volume=args.volume, duration=args.duration, n_terms=args.n_terms)
    print("Schwinger pair production estimate (uniform field):")
    print(f"  E0 = {args.E0:.3e} V/m, V = {args.volume:.3e} m^3, Δt = {args.duration:.3e} s, n_terms={args.n_terms}")
    print(f"  Expected pairs: {N:.3e}")
    if N < 1e-6:
        print("  Note: For E << E_S, production is exponentially suppressed; values ≪ 1 are expected.")


if __name__ == "__main__":
    main()
