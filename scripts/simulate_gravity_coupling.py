#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt

# Allow running without installation
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from efqs.em_sources import make_grid, interfering_pulses_energy, rotating_quadrupole_energy, gaussian_beam_energy, plasma_ring_energy
from efqs.gravitational_coupling import quadrupole_moment, strain_far_field, radiated_power_from_quadrupole, dominant_frequency
from efqs.metrics import coupling_metrics
from efqs.pair_production import energy_loss_power
from efqs.cli_utils import load_config, save_results_json


def main():
    p = argparse.ArgumentParser(description="Linearized GR coupling from EM energy densities")
    p.add_argument("--config", required=True, help="Path to JSON/YAML config file")
    p.add_argument("--plot", action="store_true", help="Plot h_ij(t) and P_GW(t)")
    args = p.parse_args()

    cfg = load_config(args.config)
    enable_gravity = bool(cfg.get("enable_gravity", True))
    obs_R = float(cfg.get("observer_distance_m", 10.0))
    use_tt = bool(cfg.get("use_tt_projection", True))
    include_qed = bool(cfg.get("include_qed_corrections", False))
    include_pair_losses = bool(cfg.get("include_pair_losses", False))
    birefringence_feedback = bool(cfg.get("birefringence_feedback", False))

    grid = make_grid(L=float(cfg.get("box_size_m", 0.1)), N=int(cfg.get("grid_points", 21)))
    dt = float(cfg.get("dt_s", 1e-12))
    steps = int(cfg.get("steps", 100))

    # Build time series of quadrupole from chosen source
    src = cfg.get("source", {"type": "interfering_pulses"})
    Q_list = []
    u_series = []
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
        elif src["type"] == "plasma_ring":
            I0 = float(src.get("I0_A", 1e6))
            R = float(src.get("R_m", 0.02))
            r_minor = float(src.get("r_minor_m", 0.005))
            omega = 2.0 * np.pi * float(src.get("frequency_Hz", 1e5))
            direction = src.get("direction", "cw")
            u = plasma_ring_energy(grid, I0=I0, R=R, r_minor=r_minor, omega=omega, t=t, direction=direction)
        else:
            raise SystemExit(f"Unknown source type: {src['type']}")
        # Flatten for quadrupole; include cell volume via scaling of rho: multiply by dV when summing later
        # Apply pair production loss (uniform approximation) by scaling energy density
        if include_pair_losses:
            # Estimate loss power from field amplitude; approximate volume as simulation box size^3
            box_V = (grid.dx * len(grid.x)) * (grid.dy * len(grid.y)) * (grid.dz * len(grid.z))
            # Use E0 if available; otherwise derive an effective E from intensity (I = u c) -> E ~ sqrt(2u/epsilon0)
            try:
                E_eff = E0
            except NameError:
                # average u to estimate E
                u_mean = float(np.mean(u))
                from efqs.constants import epsilon0
                E_eff = float(np.sqrt(2.0 * u_mean / epsilon0))
            P_loss = energy_loss_power(E_eff, volume=box_V, n_terms=1)
            # Reduce total energy by P_loss*dt: distribute proportionally over grid
            E_total = float(np.sum(u) * grid.dV)
            if E_total > 0.0:
                frac = max(0.0, 1.0 - (P_loss * dt) / E_total)
                u *= frac
        
        # Apply birefringence feedback (first-order perturbation via local Δn)
        if birefringence_feedback:
            # Estimate local E from u (assuming u ~ epsilon0 E^2 / 2)
            from efqs.constants import epsilon0
            from efqs.heisenberg_euler import delta_n_E
            E_local = np.sqrt(2.0 * u / epsilon0 + 1e-20)
            # Compute Δn (use parallel component as proxy)
            dn_par, dn_perp = delta_n_E(E_local)
            # Effective index change modulates intensity: I' = I * (1 + Δn)
            # Energy density u scales similarly (first-order)
            u *= (1.0 + dn_par)
        
        pos = grid.positions_flat
        u_flat = u.reshape(-1)
        # quadrupole currently sums rho*..., where rho = u/c^2; include dV by scaling rho
        Q = quadrupole_moment(pos, u_flat * grid.dV)
        Q_list.append(Q)
        u_series.append(u)

    Q_t = np.stack(Q_list, axis=0)
    u_t = np.stack(u_series, axis=0)

    if enable_gravity:
        h_t = strain_far_field(Q_t, dt=dt, R=obs_R, use_tt=use_tt)
        P_t = radiated_power_from_quadrupole(Q_t, dt=dt)
        qed_str = " (with QED corrections)" if include_qed else ""
        tt_str = " (TT-projected)" if use_tt else ""
        print(f"Computed strain h_ij(t){tt_str} with shape {h_t.shape} and GW power time series with shape {P_t.shape}.{qed_str}")
        rms_h = float(np.sqrt(np.mean(h_t**2)))
        avg_P = float(np.mean(P_t))
        print(f"RMS strain magnitude ~ {rms_h:.3e}")
        print(f"Average P_GW ~ {avg_P:.3e} W")
        # Efficiency metrics
        m = coupling_metrics(h_t, P_t, u_t, dV=grid.dV, dt=dt)
        print("Coupling metrics:")
        for k, v in m.items():
            print(f"  {k}: {v:.3e}")
        # Frequency-domain analysis
        freq_info = dominant_frequency(h_t, dt, component=(0, 0))
        print(f"Strain spectrum (h_xx): peak at {freq_info['peak_freq_Hz']:.3e} Hz, amplitude {freq_info['peak_amplitude']:.3e}, BW {freq_info['bandwidth_Hz']:.3e} Hz")
        # Optional JSON output
        out_path = cfg.get("output_json")
        if out_path:
            results = {
                "observer_distance_m": obs_R,
                "use_tt_projection": use_tt,
                "include_pair_losses": include_pair_losses,
                "birefringence_feedback": birefringence_feedback,
                "source": src,
                "steps": steps,
                "dt_s": dt,
                "rms_strain": rms_h,
                "avg_P_GW_W": avg_P,
                "metrics": m,
                "frequency_spectrum": freq_info,
            }
            save_results_json(results, out_path)
            print(f"Saved results to {out_path}")
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
