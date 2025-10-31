#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Allow running without installation
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from efqs.em_sources import make_grid, interfering_pulses_energy, rotating_quadrupole_energy
from efqs.gravitational_coupling import quadrupole_moment, strain_far_field, radiated_power_from_quadrupole


def load_config(path: str) -> Dict[str, Any]:
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise SystemExit(f"Failed to load config {path}: {e}")


def main():
    p = argparse.ArgumentParser(description="Linearized GR coupling from EM energy densities")
    p.add_argument("--config", required=True, help="Path to JSON/YAML config file")
    p.add_argument("--plot", action="store_true", help="Plot h_ij(t) and P_GW(t)")
    args = p.parse_args()

    cfg = load_config(args.config)
    enable_gravity = bool(cfg.get("enable_gravity", True))
    obs_R = float(cfg.get("observer_distance_m", 10.0))

    grid = make_grid(L=float(cfg.get("box_size_m", 0.1)), N=int(cfg.get("grid_points", 21)))
    dt = float(cfg.get("dt_s", 1e-12))
    steps = int(cfg.get("steps", 100))

    # Build time series of quadrupole from chosen source
    src = cfg.get("source", {"type": "interfering_pulses"})
    Q_list = []
    for n in range(steps):
        t = n * dt
        if src["type"] == "interfering_pulses":
            E0 = float(src.get("E0_Vpm", 1e15))
            wavelength = float(src.get("wavelength_m", 800e-9))
            omega = 2.0 * np.pi * float(src.get("frequency_Hz", 3.75e14))
            u = interfering_pulses_energy(grid, E0=E0, wavelength=wavelength, omega=omega, t=t)
        elif src["type"] == "rotating_quadrupole":
            U0 = float(src.get("U0_Jpm3", 1e12))
            R0 = float(src.get("R0_m", 0.02))
            omega = 2.0 * np.pi * float(src.get("frequency_Hz", 1e5))
            u = rotating_quadrupole_energy(grid, U0=U0, R0=R0, omega=omega, t=t)
        else:
            raise SystemExit(f"Unknown source type: {src['type']}")
        # Flatten for quadrupole; include cell volume via scaling of rho: multiply by dV when summing later
        pos = grid.positions_flat
        u_flat = u.reshape(-1)
        # quadrupole currently sums rho*..., where rho = u/c^2; include dV by scaling rho
        Q = quadrupole_moment(pos, u_flat * grid.dV)
        Q_list.append(Q)

    Q_t = np.stack(Q_list, axis=0)

    if enable_gravity:
        h_t = strain_far_field(Q_t, dt=dt, R=obs_R)
        P_t = radiated_power_from_quadrupole(Q_t, dt=dt)
        print(f"Computed strain h_ij(t) with shape {h_t.shape} and GW power time series with shape {P_t.shape}.")
        print(f"RMS strain magnitude ~ {np.sqrt(np.mean(h_t**2)):.3e}")
        print(f"Average P_GW ~ {np.mean(P_t):.3e} W")
        if args.plot:
            t = np.arange(steps) * dt
            # Plot trace-free diagonal components as a proxy
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
            axes[0].plot(t, h_t[:,0,0], label='h_xx')
            axes[0].plot(t, h_t[:,1,1], label='h_yy')
            axes[0].plot(t, h_t[:,2,2], label='h_zz')
            axes[0].set_ylabel('h (dimensionless)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(t, P_t, label='P_GW')
            axes[1].set_ylabel('Power [W]')
            axes[1].set_xlabel('Time [s]')
            axes[1].grid(True, alpha=0.3)
            plt.show()
    else:
        print("Gravity coupling disabled; computed quadrupole only.")


if __name__ == "__main__":
    main()
