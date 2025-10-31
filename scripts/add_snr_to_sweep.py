#!/usr/bin/env python3
"""
Post-process sweep CSV to add SNR columns for each detector.

Reads a sweep summary CSV with h_rms and freq_peak columns,
computes matched-filter SNR for each detector, and writes
an augmented CSV with SNR_{detector} columns.

Usage:
  python scripts/add_snr_to_sweep.py --input sweep_summary.csv --output sweep_with_snr.csv
  python scripts/add_snr_to_sweep.py --input results/sweeps/test_mini/test_mini_sweep_summary.csv --output results/sweep_mini_snr.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from efqs.detector_noise import DETECTOR_NOISE_CURVES, matched_filter_snr
    HAS_DETECTOR_NOISE = True
except ImportError:
    HAS_DETECTOR_NOISE = False
    print("Warning: detector_noise module not available; SNR calculation skipped")


def compute_snr_for_detectors(h_rms: float, freq_peak: float, T_obs: float = 1.0) -> dict:
    """
    Compute matched-filter SNR for all detectors.
    
    Args:
        h_rms: RMS strain
        freq_peak: Peak frequency (Hz)
        T_obs: Observation time (seconds)
    
    Returns:
        Dictionary with {detector_name: SNR}
    """
    if not HAS_DETECTOR_NOISE:
        return {}
    
    snr_dict = {}
    
    # Handle DC/static sources (freq_peak = 0 or NaN)
    # Use a characteristic frequency of 1 Hz for quasi-static fields
    if freq_peak == 0.0 or not np.isfinite(freq_peak):
        freq_peak = 1.0  # Hz - quasi-static characteristic frequency
    
    # Create a simple Gaussian pulse in frequency domain centered at freq_peak
    # This is a crude approximation; real strains have complex spectra
    f_array = np.linspace(0.1, 10000, 10000)  # Hz
    sigma_f = freq_peak * 0.1  # 10% bandwidth
    h_fft = h_rms * np.exp(-0.5 * ((f_array - freq_peak) / sigma_f) ** 2)
    h_fft = h_fft.astype(complex)  # matched_filter_snr expects complex
    
    for detector_name, detector_curve in DETECTOR_NOISE_CURVES.items():
        # Get frequency band
        f_min, f_max = detector_curve.frequency_band
        
        # Only compute SNR if frequency is within detector band
        if f_min <= freq_peak <= f_max:
            try:
                snr = matched_filter_snr(h_fft, f_array, detector_name, integration_time=T_obs)
                snr_dict[detector_name] = float(snr)
            except Exception as e:
                print(f"Warning: SNR calculation failed for {detector_name}: {e}")
                snr_dict[detector_name] = np.nan
        else:
            # Out of band
            snr_dict[detector_name] = 0.0
    
    return snr_dict


def main():
    ap = argparse.ArgumentParser(description="Add SNR columns to sweep CSV")
    ap.add_argument("--input", required=True, help="Input sweep CSV")
    ap.add_argument("--output", required=True, help="Output CSV with SNR columns")
    ap.add_argument("--h-col", default="R_10.0m_h_rms", help="Column name for h_rms")
    ap.add_argument("--freq-col", default="R_10.0m_freq_peak", help="Column name for peak frequency")
    ap.add_argument("--T-obs", type=float, default=1.0, help="Observation time (seconds)")
    args = ap.parse_args()
    
    if not HAS_DETECTOR_NOISE:
        print("Error: detector_noise module required. Install dependencies.")
        return 1
    
    df = pd.read_csv(args.input)
    
    # Check columns exist
    if args.h_col not in df.columns:
        print(f"Error: Column '{args.h_col}' not found. Available: {df.columns.tolist()}")
        return 1
    if args.freq_col not in df.columns:
        print(f"Error: Column '{args.freq_col}' not found. Available: {df.columns.tolist()}")
        return 1
    
    # Compute SNR for each row
    print(f"Computing SNR for {len(df)} sweep points...")
    for idx, row in df.iterrows():
        h_rms = row[args.h_col]
        freq_peak = row[args.freq_col]
        
        if pd.isna(h_rms) or h_rms == 0.0:
            # Skip rows with no signal
            for detector_name in DETECTOR_NOISE_CURVES.keys():
                df.loc[idx, f"SNR_{detector_name}"] = np.nan
            continue
        
        # compute_snr_for_detectors handles freq_peak=0 with fallback to 1 Hz
        snr_dict = compute_snr_for_detectors(h_rms, freq_peak, args.T_obs)
        
        for detector_name, snr in snr_dict.items():
            df.loc[idx, f"SNR_{detector_name}"] = snr
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows with SNR columns to {args.output}")
    
    # Summary
    snr_cols = [c for c in df.columns if c.startswith("SNR_")]
    if snr_cols:
        print("\nSNR Summary:")
        for col in snr_cols:
            max_snr = df[col].max()
            mean_snr = df[col].mean()
            print(f"  {col}: max={max_snr:.2f}, mean={mean_snr:.2f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
