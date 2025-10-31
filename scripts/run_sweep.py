#!/usr/bin/env python3
"""
Parameter sweep runner for gravitational coupling simulations.

Reads a sweep configuration and runs the simulation across a parameter grid,
saving results to a CSV or JSON file for analysis.
"""
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any, List
import numpy as np
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from efqs.em_sources import make_grid, interfering_pulses_energy, rotating_quadrupole_energy, gaussian_beam_energy
from efqs.gravitational_coupling import quadrupole_moment, strain_far_field, radiated_power_from_quadrupole
from efqs.cli_utils import load_config, save_results_json


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def run_single_sim(base_cfg: Dict[str, Any], param_values: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single simulation with the given parameter overrides."""
    cfg = base_cfg.copy()
    for key, val in param_values.items():
        # Navigate nested dicts if key contains '.'
        if '.' in key:
            parts = key.split('.')
            target = cfg
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = val
        else:
            cfg[key] = val
    
    grid = make_grid(L=float(cfg.get("box_size_m", 0.1)), N=int(cfg.get("grid_points", 21)))
    dt = float(cfg.get("dt_s", 1e-12))
    steps = int(cfg.get("steps", 100))
    obs_R = float(cfg.get("observer_distance_m", 10.0))
    use_tt = bool(cfg.get("use_tt_projection", True))
    
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
        elif src["type"] == "gaussian_beam":
            P0 = float(src.get("P0_W", 1e12))
            w0 = float(src.get("w0_m", 1e-3))
            wavelength = float(src.get("wavelength_m", 800e-9))
            omega = 2.0 * np.pi * float(src.get("frequency_Hz", 0.0))
            u = gaussian_beam_energy(grid, P0=P0, w0=w0, wavelength=wavelength, omega=omega, t=t)
        else:
            raise ValueError(f"Unknown source type: {src['type']}")
        
        pos = grid.positions_flat
        u_flat = u.reshape(-1)
        Q = quadrupole_moment(pos, u_flat * grid.dV)
        Q_list.append(Q)
    
    Q_t = np.stack(Q_list, axis=0)
    h_t = strain_far_field(Q_t, dt=dt, R=obs_R, use_tt=use_tt)
    P_t = radiated_power_from_quadrupole(Q_t, dt=dt)
    
    results = {
        "h_rms": float(np.sqrt(np.mean(h_t**2))),
        "h_max": float(np.max(np.abs(h_t))),
        "P_avg": float(np.mean(P_t)),
        "P_max": float(np.max(P_t)),
    }
    results.update(param_values)
    return results


def main():
    p = argparse.ArgumentParser(description="Run parameter sweeps for gravity coupling")
    p.add_argument("--sweep-config", required=True, help="Path to JSON sweep config file")
    p.add_argument("--output", default="sweep_results.json", help="Output file for results")
    args = p.parse_args()
    
    sweep_cfg = load_config(args.sweep_config)
    base_cfg = sweep_cfg.get("base_config", {})
    sweep_params = sweep_cfg.get("sweep_parameters", {})
    
    # Build parameter grid
    param_names = list(sweep_params.keys())
    param_ranges = [sweep_params[name] for name in param_names]
    
    results = []
    total = np.prod([len(r) for r in param_ranges])
    count = 0
    
    import itertools
    for param_combo in itertools.product(*param_ranges):
        param_values = dict(zip(param_names, param_combo))
        print(f"[{count+1}/{int(total)}] Running: {param_values}")
        result = run_single_sim(base_cfg, param_values)
        results.append(result)
        count += 1
    
    # Save results
    save_results_json(results, args.output)
    print(f"\nSaved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
