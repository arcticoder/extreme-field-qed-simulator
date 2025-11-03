#!/usr/bin/env python3
"""
Full pipeline experiment runner for extreme-field QED + gravitational coupling.

Reads YAML experiment definitions, computes EM fields with optional QED corrections,
evaluates gravitational coupling (classical + anomalous), and saves comprehensive results.

Usage:
    python run_experiments.py --config configs/experiments.yaml --experiment experiment_1
    python run_experiments.py --config configs/experiments.yaml --sweep parameter_sweep
"""
from __future__ import annotations
import argparse
import yaml
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from efqs.source_geometries import GEOMETRIES
from efqs.gravitational_coupling import (
    stress_energy_from_fields, quadrupole_moment,
    strain_far_field, radiated_power_from_quadrupole,
    dominant_frequency
)
from efqs.anomalous_coupling import (
    PREDEFINED_COUPLINGS, apply_anomalous_coupling,
    compute_kappa_constraints
)
from efqs.metrics import coupling_metrics
from efqs.constants import c, epsilon0


def load_experiment_config(config_path: str, experiment_name: str) -> Dict[str, Any]:
    """Load specific experiment from YAML config file."""
    with open(config_path, 'r') as f:
        docs = list(yaml.safe_load_all(f))
    
    for doc in docs:
        if doc and experiment_name in doc:
            config = doc[experiment_name]
            config['name'] = experiment_name  # Add name to config
            
            # Fix YAML scientific notation parsing bug (1.0e14 -> str instead of float)
            # Convert all numeric-looking strings in parameters to floats
            if 'geometry' in config and 'parameters' in config['geometry']:
                params = config['geometry']['parameters']
                for key, value in params.items():
                    if isinstance(value, str):
                        try:
                            params[key] = float(value)
                        except ValueError:
                            pass  # Keep as string if not convertible
            
            return config
    
    raise ValueError(f"Experiment '{experiment_name}' not found in {config_path}")


def create_grid(config: Dict) -> tuple:
    """Create spatial grid from config."""
    # Support both formats: box_size+grid_points or explicit min/max/n
    if 'box_size' in config:
        box_size = config['box_size']
        grid_points = config['grid_points']
        
        if isinstance(grid_points, list):
            nx, ny, nz = grid_points
        else:
            nx = ny = nz = grid_points
        
        x = np.linspace(-box_size/2, box_size/2, nx)
        y = np.linspace(-box_size/2, box_size/2, ny)
        z = np.linspace(-box_size/2, box_size/2, nz)
    else:
        # Explicit min/max/n format
        x = np.linspace(config['x_min'], config['x_max'], config['nx'])
        y = np.linspace(config['y_min'], config['y_max'], config['ny'])
        z = np.linspace(config['z_min'], config['z_max'], config['nz'])
        nx, ny, nz = config['nx'], config['ny'], config['nz']
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    
    dV = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    
    return positions, dV, (nx, ny, nz)


def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute single experiment and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{config['description']}")
    print(f"{'='*60}\n")
    
    # Setup grid
    print("Setting up grid...")
    positions, dV, grid_shape = create_grid(config['grid'])
    N_grid = len(positions)
    
    # Setup time array
    time_config = config['time_evolution']
    if 'dt' in time_config:
        dt = time_config['dt']
        t_array = np.arange(time_config['t_start'], time_config['t_end'], dt)
    else:
        # Compute dt from t_start, t_end, nt
        nt = time_config['nt']
        t_array = np.linspace(time_config['t_start'], time_config['t_end'], nt)
        dt = t_array[1] - t_array[0]
    
    N_time = len(t_array)
    
    print(f"Grid: {grid_shape}, {N_grid} points")
    print(f"Time: {N_time} steps, dt = {dt:.3e} s")
    
    # Get geometry evaluator
    geom_type = config['geometry']['type']
    geom_params = config['geometry']['parameters']
    geometry = GEOMETRIES[geom_type]
    field_func = geometry.field_evaluator
    
    print(f"Geometry: {geometry.name}")
    
    # Initialize storage
    Q_t = np.zeros((N_time, 3, 3))
    T00_EM_avg = np.zeros(N_time)
    
    # Optional: store full field snapshots
    save_fields = config['output'].get('save_fields', False)
    if save_fields:
        # Store only snapshots, not full 4D array (memory limit)
        snapshot_indices = np.linspace(0, N_time-1, min(20, N_time), dtype=int)
        E_snapshots = []
        B_snapshots = []
    
    # Time evolution
    print("\nComputing EM fields and stress-energy...")
    for i, t in enumerate(t_array):
        if i % max(1, N_time // 10) == 0:
            print(f"  Step {i}/{N_time} (t = {t:.3e} s)")
        
        # Evaluate fields
        E, B = field_func(positions, t, **geom_params)
        
        # Compute stress-energy (returns T00 as numpy array)
        include_qed = config['physics'].get('include_qed_stress_energy', False)
        T00 = stress_energy_from_fields(E, B, include_qed=include_qed)
        
        T00_EM_avg[i] = np.mean(T00)
        
        # Quadrupole moment
        Q_t[i] = quadrupole_moment(positions, T00 * dV)
        
        # Store snapshots if requested
        if save_fields and i in snapshot_indices:
            E_snapshots.append(E.reshape(grid_shape + (3,)))
            B_snapshots.append(B.reshape(grid_shape + (3,)))
    
    print("Done computing fields.\n")
    
    # Gravitational coupling
    results = {'config': config}
    
    if config.get('gravitational', {}).get('enabled', True):  # Default enabled if section exists
        print("Computing gravitational coupling...")
        
        grav_config = config.get('gravitational', {})
        observer_distances = grav_config.get('observer_distances', [1.0, 10.0, 100.0])
        use_spectral = grav_config.get('use_spectral_derivatives', True)
        use_tt = grav_config.get('apply_TT_projection', True)
        
        results['gravitational'] = {}
        
        for R in observer_distances:
            print(f"  Observer distance R = {R} m")
            
            h_t = strain_far_field(Q_t, dt, R, use_tt=use_tt, use_spectral=use_spectral)
            P_t = radiated_power_from_quadrupole(Q_t, dt, use_spectral=use_spectral)
            
            h_rms = float(np.sqrt(np.mean(h_t**2)))
            P_avg = float(np.mean(P_t))
            
            freq_info = dominant_frequency(h_t, dt)
            
            print(f"    h_rms = {h_rms:.3e}")
            print(f"    P_avg = {P_avg:.3e} W")
            print(f"    Peak freq = {freq_info['peak_freq_Hz']:.3e} Hz")
            print(f"    Peak amplitude = {freq_info['peak_amplitude']:.3e}")
            print(f"    -3 dB bandwidth = {freq_info['bandwidth_Hz']:.3e} Hz")
            
            results['gravitational'][f'R_{R}m'] = {
                'h_rms': h_rms,
                'h_max': float(np.max(np.abs(h_t))),
                'P_avg': P_avg,
                'P_peak': float(np.max(P_t)),
                'frequency_spectrum': freq_info,
                'h_timeseries': h_t if config['output'].get('save_strain', False) else None,
                'P_timeseries': P_t if config['output'].get('save_strain', False) else None,
            }
        
        print()
    
    # Anomalous coupling analysis
    anom_config = config.get('anomalous_coupling', {})
    if anom_config and anom_config.get('enabled', True):  # Default enabled if section exists
        print("Analyzing anomalous coupling scenarios...")
        
        ansatz_names = anom_config.get('ansatze', anom_config.get('ansatz', []))
        kappa_values = anom_config.get('kappa_scan', [])
        thresholds = config.get('detection_thresholds', {})
        
        results['anomalous'] = {}
        
        for ansatz_name in ansatz_names:
            print(f"\n  Ansatz: {ansatz_name}")
            
            if ansatz_name not in PREDEFINED_COUPLINGS:
                print(f"    Warning: {ansatz_name} not in predefined couplings, skipping")
                continue
            
            # For anomalous coupling, we need characteristic F value
            # Recompute fields at t=0 for reference
            t_ref = 0.0
            E_ref, B_ref = field_func(positions, t_ref, **geom_params)
            
            # Dummy field dict for coupling functional
            # (This is simplified; full implementation needs proper field passing)
            fields_dict = {'E': E_ref, 'B': B_ref, 'omega': 2.0*np.pi*c/800e-9}
            
            # Get baseline h_rms from gravitational results
            R_primary = observer_distances[0]
            h_EM = results['gravitational'][f'R_{R_primary}m']['h_rms']
            
            # Compute required κ for each threshold
            coupling = PREDEFINED_COUPLINGS[ansatz_name]
            
            # Estimate characteristic F (depends on ansatz)
            if 'field_invariant' in ansatz_name.lower():
                E2 = np.sum(E_ref**2, axis=-1)
                B2 = np.sum(B_ref**2, axis=-1)
                F_char = float(np.mean(0.25 * (B2 - E2/c**2)**2))
            elif 'photon' in ansatz_name.lower():
                E2 = np.sum(E_ref**2, axis=-1)
                B2 = np.sum(B_ref**2, axis=-1)
                u_ref = 0.5 * (epsilon0 * E2 + B2 / (4*np.pi*1e-7))
                F_char = float(np.mean(u_ref))
            else:
                F_char = 1.0  # Placeholder
            
            kappa_constraints = compute_kappa_constraints(
                results['gravitational'][f'R_{R_primary}m']['h_timeseries'] 
                    if results['gravitational'][f'R_{R_primary}m']['h_timeseries'] is not None
                    else np.ones(10) * h_EM,
                results['gravitational'][f'R_{R_primary}m']['P_timeseries']
                    if results['gravitational'][f'R_{R_primary}m']['P_timeseries'] is not None
                    else np.ones(10) * results['gravitational'][f'R_{R_primary}m']['P_avg'],
                thresholds,
                F_char,
                float(np.mean(T00_EM_avg))
            )
            
            results['anomalous'][ansatz_name] = {
                'kappa_constraints': kappa_constraints,
                'F_characteristic': F_char,
            }
            
            for detector, kappa_req in kappa_constraints.items():
                print(f"    {detector}: κ_required = {kappa_req:.3e}")
    
    print("\nExperiment complete.\n")
    return results


def save_results_hdf5(results: Dict[str, Any], output_path: str):
    """Save results to HDF5 file."""
    print(f"Saving results to {output_path}...")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        f.attrs['experiment_name'] = results['config']['name']
        f.attrs['description'] = results['config']['description']
        
        # Gravitational results
        if 'gravitational' in results:
            grav_group = f.create_group('gravitational')
            for key, data in results['gravitational'].items():
                obs_group = grav_group.create_group(key)
                for metric_name, value in data.items():
                    if value is not None and not isinstance(value, dict):
                        if isinstance(value, np.ndarray):
                            obs_group.create_dataset(metric_name, data=value)
                        else:
                            obs_group.attrs[metric_name] = value
                    elif isinstance(value, dict):
                        # Handle nested dicts (like frequency_spectrum)
                        for k, v in value.items():
                            obs_group.attrs[f"{metric_name}_{k}"] = v
        
        # Anomalous coupling results
        if 'anomalous' in results:
            anom_group = f.create_group('anomalous')
            for ansatz_name, data in results['anomalous'].items():
                ansatz_group = anom_group.create_group(ansatz_name)
                ansatz_group.attrs['F_characteristic'] = data['F_characteristic']
                
                const_group = ansatz_group.create_group('kappa_constraints')
                for detector, kappa in data['kappa_constraints'].items():
                    const_group.attrs[detector] = kappa
    
    print("Results saved.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run extreme-field QED + gravitational coupling experiments"
    )
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--experiment', help='Experiment name to run')
    parser.add_argument('--sweep', help='Parameter sweep name to run')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("Listing available experiments in", args.config)
        with open(args.config, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        for doc in docs:
            if doc:
                for key in doc.keys():
                    print(f"  - {key}")
        return
    
    if not args.experiment and not args.sweep:
        print("Error: Must specify either --experiment or --sweep")
        return
    
    if args.experiment:
        config = load_experiment_config(args.config, args.experiment)
        results = run_experiment(config)
        
        output_file = config.get('output', {}).get('hdf5_path', config.get('output', {}).get('hdf5_file', f'results/{args.experiment}.h5'))
        save_results_hdf5(results, output_file)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Name: {config['name']}")
        if 'gravitational' in results:
            for key, data in results['gravitational'].items():
                print(f"\n{key}:")
                print(f"  h_rms = {data['h_rms']:.3e}")
                print(f"  P_avg = {data['P_avg']:.3e} W")
        
        if 'anomalous' in results:
            print(f"\nAnomалous Coupling Constraints:")
            for ansatz, data in results['anomalous'].items():
                print(f"\n  {ansatz}:")
                for detector, kappa in data['kappa_constraints'].items():
                    print(f"    {detector}: κ < {kappa:.3e}")
    
    elif args.sweep:
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP: {args.sweep}")
        print(f"{'='*60}\n")
        
        sweep_config = load_experiment_config(args.config, args.sweep)
        sweep_results = run_parameter_sweep(sweep_config)
        
        # Save consolidated summary
        output_dir = Path(sweep_config.get('output', {}).get('sweep_dir', 'results/sweeps'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_csv = output_dir / f"{args.sweep}_summary.csv"
        sweep_results['summary'].to_csv(summary_csv, index=False)
        print(f"\nSummary saved to {summary_csv}")
        
        # Quick visualization
        if sweep_results['summary'].shape[0] > 1:
            plot_sweep_summary(sweep_results['summary'], 
                             sweep_config.get('sweep_parameter', 'parameter'),
                             output_dir / f"{args.sweep}_plots.png")


def run_parameter_sweep(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute parameter sweep over a grid of values.
    
    config: sweep configuration with 'sweep_parameter' and 'sweep_values'
    Returns: dict with 'runs' (list of results) and 'summary' (pandas DataFrame)
    """
    import pandas as pd
    
    sweep_param = config['sweep_parameter']
    sweep_values = config['sweep_values']
    
    # Convert to float if needed (YAML scientific notation bug)
    sweep_values = [float(v) if isinstance(v, str) else v for v in sweep_values]
    
    base_config = config.copy()
    
    print(f"Sweeping {sweep_param} over {len(sweep_values)} values")
    print(f"Range: {min(sweep_values):.3e} to {max(sweep_values):.3e}\n")
    
    all_runs = []
    summary_rows = []
    
    for i, value in enumerate(sweep_values):
        print(f"[{i+1}/{len(sweep_values)}] {sweep_param} = {value:.3e}")
        
        # Deep copy and update parameter
        run_config = update_sweep_parameter(base_config, sweep_param, value)
        run_config['name'] = f"{config['name']}_sweep_{i}"
        
        # Run experiment
        try:
            results = run_experiment(run_config)
            all_runs.append(results)
            
            # Extract summary metrics
            row = {'sweep_param': sweep_param, 'sweep_value': value}
            
            if 'gravitational' in results:
                for obs_key, obs_data in results['gravitational'].items():
                    row[f'{obs_key}_h_rms'] = obs_data['h_rms']
                    row[f'{obs_key}_h_max'] = obs_data['h_max']
                    row[f'{obs_key}_P_avg'] = obs_data['P_avg']
                    row[f'{obs_key}_freq_peak'] = obs_data.get('frequency_spectrum_peak_freq_Hz', 0.0)
            
            if 'anomalous' in results:
                for ansatz, ansatz_data in results['anomalous'].items():
                    for detector, kappa in ansatz_data['kappa_constraints'].items():
                        row[f'kappa_{ansatz}_{detector}'] = kappa
            
            summary_rows.append(row)
            
            # Save individual HDF5
            output_dir = Path(run_config.get('output', {}).get('sweep_dir', 'results/sweeps'))
            output_dir.mkdir(parents=True, exist_ok=True)
            hdf5_path = output_dir / f"{run_config['name']}.h5"
            save_results_hdf5(results, hdf5_path)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            summary_rows.append({'sweep_param': sweep_param, 'sweep_value': value, 'error': str(e)})
    
    summary_df = pd.DataFrame(summary_rows)
    
    return {'runs': all_runs, 'summary': summary_df, 'config': config}


def update_sweep_parameter(config: Dict, param_path: str, value: float) -> Dict:
    """Update a nested parameter in config dict.
    
    param_path: dot-separated path like 'geometry.parameters.E0'
    value: new value to set
    """
    import copy
    new_config = copy.deepcopy(config)
    
    keys = param_path.split('.')
    target = new_config
    
    for key in keys[:-1]:
        target = target[key]
    
    target[keys[-1]] = value
    
    return new_config


def plot_sweep_summary(summary_df, sweep_param: str, save_path: str):
    """Generate quick visualization of sweep results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = summary_df['sweep_value'].values
    
    # h_rms
    ax = axes[0, 0]
    plotted = False
    for col in summary_df.columns:
        if 'h_rms' in col:
            ax.loglog(x, summary_df[col].values, 'o-', label=col)
            plotted = True
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('h_rms')
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # P_avg
    ax = axes[0, 1]
    plotted = False
    for col in summary_df.columns:
        if 'P_avg' in col:
            ax.loglog(x, summary_df[col].values, 's-', label=col)
            plotted = True
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('P_avg [W]')
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # κ constraints (pick one detector, all ansätze)
    ax = axes[1, 0]
    plotted = False
    for col in summary_df.columns:
        if 'kappa_' in col and 'LIGO' in col:
            ax.loglog(x, summary_df[col].values, '^-', label=col.replace('kappa_', '').replace('_LIGO', ''))
            plotted = True
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('κ_required (LIGO)')
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Peak frequency
    ax = axes[1, 1]
    plotted = False
    for col in summary_df.columns:
        if 'freq_peak' in col:
            ax.semilogx(x, summary_df[col].values, 'd-', label=col)
            plotted = True
    ax.set_xlabel(sweep_param)
    ax.set_ylabel('Peak Frequency [Hz]')
    if plotted:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sweep plots saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run extreme-field QED + gravitational coupling experiments"
    )
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--experiment', help='Experiment name to run')
    parser.add_argument('--sweep', help='Parameter sweep name to run')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    
    args = parser.parse_args()
    
    if args.list:
        print("Listing available experiments in", args.config)
        with open(args.config, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        for doc in docs:
            if doc:
                for key in doc.keys():
                    print(f"  - {key}")
        return
    
    if not args.experiment and not args.sweep:
        print("Error: Must specify either --experiment or --sweep")
        return
    
    if args.experiment:
        config = load_experiment_config(args.config, args.experiment)
        results = run_experiment(config)
        
        output_file = config.get('output', {}).get('hdf5_path', config.get('output', {}).get('hdf5_file', f'results/{args.experiment}.h5'))
        save_results_hdf5(results, output_file)
        
        print(f"\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Name: {config['name']}")
        if 'gravitational' in results:
            for key, data in results['gravitational'].items():
                print(f"\n{key}:")
                print(f"  h_rms = {data['h_rms']:.3e}")
                print(f"  P_avg = {data['P_avg']:.3e} W")
        
        if 'anomalous' in results:
            print(f"\nAnomалous Coupling Constraints:")
            for ansatz, data in results['anomalous'].items():
                print(f"\n  {ansatz}:")
                for detector, kappa in data['kappa_constraints'].items():
                    print(f"    {detector}: κ < {kappa:.3e}")
    
    elif args.sweep:
        print(f"\n{'='*60}")
        print(f"PARAMETER SWEEP: {args.sweep}")
        print(f"{'='*60}\n")
        
        sweep_config = load_experiment_config(args.config, args.sweep)
        sweep_results = run_parameter_sweep(sweep_config)
        
        # Save consolidated summary
        output_dir = Path(sweep_config.get('output', {}).get('sweep_dir', 'results/sweeps'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_csv = output_dir / f"{args.sweep}_summary.csv"
        sweep_results['summary'].to_csv(summary_csv, index=False)
        print(f"\nSummary saved to {summary_csv}")
        
        # Quick visualization
        if sweep_results['summary'].shape[0] > 1:
            plot_sweep_summary(sweep_results['summary'], 
                             sweep_config.get('sweep_parameter', 'parameter'),
                             output_dir / f"{args.sweep}_plots.png")


if __name__ == '__main__':
    main()
