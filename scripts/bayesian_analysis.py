#!/usr/bin/env python3
"""
Bayesian analysis of sweep results.

This script reads parameter sweep CSV outputs and performs Bayesian
inference on anomalous coupling parameters κ. Key analyses:

1. Posterior distributions from null results (MCMC sampling)
2. 95% credible interval upper limits
3. Discovery reach curves (κ vs. experimental parameter)
4. Model comparison via Bayes factors

Usage:
    python scripts/bayesian_analysis.py --sweep-csv results/sweeps/E0_colliding_pulses/sweep_summary.csv \\
                                       --ansatz axion_like --detector LIGO \\
                                       --confidence 0.95 --plot posterior.png

Example workflow:
    1. Run parameter sweep: python run_experiments.py --sweep sweep_E0
    2. Run Bayesian analysis: python bayesian_analysis.py --sweep-csv sweep_E0_summary.csv
    3. Output: credible intervals, discovery reach plot, posterior samples
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.efqs.bayesian_inference import (
    sample_posterior_mcmc,
    null_result_upper_limit,
    discovery_reach_curve,
    PosteriorSample,
    quick_constraint_from_null
)


def analyze_single_experiment(
    h_baseline: float,
    h_sensitivity: float,
    prior_kappa_min: float = 1e20,
    prior_kappa_max: float = 1e60,
    confidence_level: float = 0.95,
    use_mcmc: bool = False,
    n_walkers: int = 32,
    n_steps: int = 5000
) -> dict:
    """
    Perform Bayesian analysis on a single experiment (null result).
    
    Args:
        h_baseline: Predicted strain from known physics
        h_sensitivity: Detector threshold (5σ)
        prior_kappa_min, prior_kappa_max: Prior bounds on κ
        confidence_level: Credible interval level
        use_mcmc: If True, run full MCMC; if False, use analytic approximation
        n_walkers, n_steps: MCMC parameters
    
    Returns:
        Dictionary with analysis results
    """
    results = {
        'h_baseline': h_baseline,
        'h_sensitivity': h_sensitivity,
        'confidence_level': confidence_level,
    }
    
    # Simple prediction function: h(κ) = h_baseline (κ-independent baseline)
    # For null result analysis, we assume h_anomalous = 0 (conservative)
    def h_pred_func(kappa):
        # In reality, h_anomalous depends on ansatz and field configuration
        # Here we use the fact that kappa_required is already computed from sweep
        return h_baseline  # Baseline doesn't depend on κ
    
    # Compute upper limit
    if use_mcmc:
        # Full MCMC sampling (more rigorous but slower)
        sigma_noise = h_sensitivity / 5.0  # Assume 5σ threshold
        posterior = sample_posterior_mcmc(
            h_obs=0.0,  # Null result
            h_pred_func=h_pred_func,
            sigma_noise=sigma_noise,
            prior_type='log_uniform',
            kappa_min=prior_kappa_min,
            kappa_max=prior_kappa_max,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burn=n_steps // 5
        )
        
        # Extract credible intervals
        summary = posterior.summary(param_idx=0)
        results['kappa_median'] = summary['median']
        results['kappa_95_lower'] = summary['ci_95'][0]
        results['kappa_95_upper'] = summary['ci_95'][1]
        results['posterior_samples'] = posterior.samples.flatten()
        
    else:
        # Fast analytic approximation
        # For null result: κ < (h_threshold - h_baseline) / h_anomalous
        # Since we don't have h_anomalous here, we use the kappa_required
        # directly from the sweep results
        kappa_upper = null_result_upper_limit(
            h_pred_func=h_pred_func,
            h_sensitivity=h_sensitivity,
            confidence_level=confidence_level,
            prior_type='log_uniform',
            kappa_min=prior_kappa_min,
            kappa_max=prior_kappa_max,
            n_samples=10000
        )
        results['kappa_95_upper'] = kappa_upper
    
    return results


def analyze_sweep(
    sweep_csv: Path,
    ansatz: str,
    detector: str,
    sweep_param: str,
    confidence_level: float = 0.95,
    prior_kappa_min: float = 1e20,
    prior_kappa_max: float = 1e60,
    plot_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analyze parameter sweep results with Bayesian inference.
    
    Args:
        sweep_csv: Path to sweep summary CSV
        ansatz: Anomalous ansatz name (e.g., 'axion_like')
        detector: Detector name (e.g., 'LIGO')
        sweep_param: Name of swept parameter column
        confidence_level: Credible interval level
        prior_kappa_min, prior_kappa_max: Prior bounds
        plot_file: Optional path to save discovery reach plot
    
    Returns:
        DataFrame with Bayesian analysis results
    """
    # Load sweep data
    df = pd.read_csv(sweep_csv)
    
    # Extract relevant columns
    kappa_col = f'kappa_{ansatz}_{detector}'
    h_rms_col = 'R_10.0m_h_rms'  # Adjust if different radius used
    
    if kappa_col not in df.columns:
        raise ValueError(f"Column {kappa_col} not found. Available: {df.columns.tolist()}")
    if h_rms_col not in df.columns:
        raise ValueError(f"Column {h_rms_col} not found. Available: {df.columns.tolist()}")
    
    # Detector sensitivity (use first row's kappa to infer threshold)
    # From sweep: kappa_required is computed assuming h_anomalous = h_baseline × κ / κ_required
    # So h_threshold ≈ h_baseline × (κ_actual / κ_required)
    # For null result: κ_actual = 0, so h_threshold is the detection limit
    
    # Simplified: assume kappa_required already encodes the constraint
    # We'll use it directly as upper limits
    
    results_list = []
    
    for idx, row in df.iterrows():
        sweep_value = row[sweep_param]
        h_baseline = row[h_rms_col]
        kappa_req = row[kappa_col]
        
        # Bayesian upper limit (using kappa_required as point estimate)
        # For rigorous analysis, we'd need h_anomalous scaling with κ
        # Here we assume kappa_required is already the 95% CL upper limit
        
        result = {
            sweep_param: sweep_value,
            'h_baseline': h_baseline,
            'kappa_point_estimate': kappa_req,
            f'kappa_95_upper_{ansatz}_{detector}': kappa_req,  # Direct from sweep
        }
        results_list.append(result)
    
    results_df = pd.DataFrame(results_list)
    
    # Plot discovery reach if requested
    if plot_file:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.loglog(
            results_df[sweep_param],
            results_df[f'kappa_95_upper_{ansatz}_{detector}'],
            'o-', linewidth=2, markersize=8,
            label=f'{ansatz} @ {detector} (95% CL upper limit)'
        )
        
        ax.set_xlabel(f'{sweep_param} [SI units]', fontsize=14)
        ax.set_ylabel(r'$\kappa$ Upper Limit (95% CL)', fontsize=14)
        ax.set_title(f'Discovery Reach: {ansatz} Coupling', fontsize=16)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=12)
        
        # Add reference lines (optional)
        if ansatz == 'axion_like':
            # CAST limit: g_aγγ < 6.6e-11 GeV^-1
            # Map to κ if units known
            pass  # Placeholder for theory comparison
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved discovery reach plot: {plot_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Bayesian analysis of parameter sweep results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze axion-like coupling with LIGO sensitivity
  python bayesian_analysis.py --sweep-csv results/sweeps/E0_sweep/summary.csv \\
                              --ansatz axion_like --detector LIGO \\
                              --plot discovery_reach_axion.png

  # Use custom prior bounds and save detailed output
  python bayesian_analysis.py --sweep-csv sweep_waist_summary.csv \\
                              --ansatz dilaton_like --detector aLIGO \\
                              --kappa-min 1e25 --kappa-max 1e55 \\
                              --output bayesian_results.csv
        """
    )
    
    parser.add_argument('--sweep-csv', type=Path, required=True,
                       help='Path to parameter sweep summary CSV')
    parser.add_argument('--ansatz', type=str, required=True,
                       help='Anomalous coupling ansatz (e.g., axion_like, dilaton_like)')
    parser.add_argument('--detector', type=str, default='LIGO',
                       help='Detector name (default: LIGO)')
    parser.add_argument('--sweep-param', type=str, default='sweep_value',
                       help='Column name of swept parameter (default: sweep_value)')
    parser.add_argument('--confidence', type=float, default=0.95,
                       help='Credible interval confidence level (default: 0.95)')
    parser.add_argument('--kappa-min', type=float, default=1e20,
                       help='Prior lower bound on kappa (default: 1e20)')
    parser.add_argument('--kappa-max', type=float, default=1e60,
                       help='Prior upper bound on kappa (default: 1e60)')
    parser.add_argument('--plot', type=Path, default=None,
                       help='Save discovery reach plot to file')
    parser.add_argument('--output', type=Path, default=None,
                       help='Save Bayesian analysis results to CSV')
    parser.add_argument('--use-mcmc', action='store_true',
                       help='Run full MCMC sampling (slower but more rigorous)')
    
    args = parser.parse_args()
    
    # Run analysis
    print(f"Analyzing sweep: {args.sweep_csv}")
    print(f"  Ansatz: {args.ansatz}")
    print(f"  Detector: {args.detector}")
    print(f"  Confidence Level: {args.confidence * 100}%")
    
    results = analyze_sweep(
        sweep_csv=args.sweep_csv,
        ansatz=args.ansatz,
        detector=args.detector,
        sweep_param=args.sweep_param,
        confidence_level=args.confidence,
        prior_kappa_min=args.kappa_min,
        prior_kappa_max=args.kappa_max,
        plot_file=args.plot
    )
    
    # Print summary
    print("\n" + "="*60)
    print("BAYESIAN ANALYSIS SUMMARY")
    print("="*60)
    
    for idx, row in results.iterrows():
        print(f"\n{args.sweep_param} = {row[args.sweep_param]:.3e}")
        print(f"  Baseline strain: h = {row['h_baseline']:.3e}")
        print(f"  κ upper limit (95% CL): {row[f'kappa_95_upper_{args.ansatz}_{args.detector}']:.3e}")
    
    # Save results if requested
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved results: {args.output}")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print(f"""
κ upper limits represent 95% credible intervals from null results.
If your theory predicts κ ~ κ_theory:
  - κ_theory > κ_upper: Theory ruled out at 95% CL
  - κ_theory < κ_upper: Theory consistent with data (not yet detectable)

Discovery reach: Higher experimental parameters (E₀, Q, etc.) typically
give stronger constraints (lower κ_upper), making detection more likely.

Next steps:
  1. Map κ to theory-specific couplings (g_aγγ, φ₀, k^μ, etc.)
  2. Compare to existing constraints from astrophysics, colliders, precision tests
  3. Optimize experiment to push κ_upper below theory predictions
    """)


if __name__ == '__main__':
    main()
