# Back-Reaction Assessment Guide

## Overview
This guide explains how to use the extreme-field QED simulator to assess whether nonlinear EM corrections (QED vacuum effects, pair production) can meaningfully back-react on spacetime via gravitational coupling.

## Key Metrics

### Coupling Efficiency Metrics
The simulation computes several metrics to quantify the EM → GW coupling:

- **`h_rms`**: Root-mean-square strain magnitude (dimensionless)
  - Typical values: 10⁻²⁵ to 10⁻²⁰ for laboratory-scale configurations
  - Interpretation: Direct measure of spacetime ripple amplitude

- **`h_max`**: Maximum absolute strain component
  - Peak value during simulation
  - Useful for identifying resonances or sharp features

- **`P_avg`**: Average gravitational wave power [W]
  - Radiated power via quadrupole formula
  - Compare to input EM power to assess efficiency

- **`E_em_avg`**: Average total EM energy [J]
  - Total electromagnetic energy in simulation volume
  - Baseline for energy budget

- **`P_in`**: Approximate input power [W]
  - Derived from dE/dt (positive injection only)
  - Proxy for source power requirement

- **`eff_Pgw_over_Pin`**: Efficiency ratio P_GW / P_in
  - Dimensionless coupling efficiency
  - Typical range: 10⁻¹⁰ to 10⁻⁵ for extreme-field configurations
  - **Critical threshold**: Values > 10⁻⁶ suggest potentially observable effects

- **`h_rms_per_J`**: Strain per unit energy [J⁻¹]
  - How much strain you get per joule of EM energy
  - Useful for comparing configurations at fixed energy budget

### Frequency-Domain Metrics
- **`peak_freq_Hz`**: Dominant frequency in strain spectrum
  - Target GW detector sensitivity bands (10 Hz – 10 kHz for ground-based)
- **`peak_amplitude`**: FFT magnitude at peak frequency
- **`bandwidth_Hz`**: Approximate FWHM bandwidth
  - Broadband signals are harder to detect; narrowband preferred

## Energy Loss Channels

### Pair Production Loss
- **Flag**: `include_pair_losses: true`
- **Model**: Uniform Schwinger-based energy drain
  - Rate: w(E) ∝ exp(-π E_s / E) at field E
  - Energy per pair: 2 m_e c²
- **Effect**: Reduces total EM energy over time; lowers P_in
- **Interpretation**: If pair losses dominate (P_loss >> P_GW), the configuration is not viable for GW coupling; most energy goes into matter creation rather than spacetime curvature.

### QED Corrections (Heisenberg-Euler)
- **Flag**: `include_qed_corrections: true`
- **Model**: Perturbative correction to energy density
  - Δu ∝ (E/E_s)⁴ for E << E_s
- **Effect**: Small increase in T_μν → slightly stronger quadrupole source
- **Interpretation**: Typically sub-percent effect unless E ≳ 0.1 E_s

## Running Assessment Sweeps

### Basic Single-Source Sweep
```bash
# Vary beam power and waist for Gaussian beam
python scripts/run_sweep.py --sweep-config examples/configs/sweep_gaussian_beam.json --output results/gaussian_sweep.json
```

### Multi-Source Comparison
```bash
# Compare interfering pulses, Gaussian beam, plasma ring with QED/pair-loss toggles
python scripts/run_sweep.py --sweep-config examples/configs/sweep_multi_source.json --output results/multi_source_sweep.json
```

### Interpreting Sweep Results
1. **Plot efficiency vs. field strength**:
   - Load `results.json` and extract `eff_Pgw_over_Pin` for each configuration
   - Look for sweet spots where efficiency peaks before pair losses dominate

2. **Frequency matching**:
   - Check `peak_freq_Hz` against detector sensitivity
   - LIGO: 10 Hz – 5 kHz; pulsar timing: nHz – μHz

3. **Energy budget**:
   - Compare `E_em_avg` to realistic source capabilities
   - National-class lasers: ~1 MJ for NIF; ~10 kJ for high-rep-rate systems

## Recommended Parameter Exploration

### Field Intensity Sweep (Interfering Pulses)
- **E0_Vpm**: [1e14, 5e14, 1e15, 2e15, 5e15]
  - Explore from sub-critical to near-Schwinger regimes
- **Expected**: Efficiency rises with E⁴ (quadrupole ∝ u ∝ E²), but pair losses kick in above ~0.1 E_s

### Beam Geometry Sweep (Gaussian Beam)
- **P0_W**: [1e11, 1e12, 1e13, 1e14]
- **w0_m**: [1e-4, 5e-4, 1e-3, 2e-3]
- **Expected**: Tighter focus (small w0) → higher peak u → stronger quadrupole, but smaller volume → lower total energy

### Rotation Frequency Sweep (Plasma Ring)
- **frequency_Hz**: [1e4, 1e5, 1e6]
- **I0_A**: [1e5, 1e6, 1e7]
- **Expected**: Higher ω → faster quadrupole time variation → more GW power (P ∝ ω⁶ for rotating quadrupole)

## Viability Criteria

A configuration is considered **potentially viable** for back-reaction if:
1. **Efficiency**: `eff_Pgw_over_Pin > 1e-6` (one part per million)
2. **Pair loss bounded**: P_loss / P_GW < 100 (pair drain not overwhelming GW output)
3. **Detectable strain**: `h_rms > 1e-22` at R = 10 m (scales as 1/R; extrapolate to detector distance)
4. **Frequency match**: `peak_freq_Hz` in detector band

## Example Workflow

```bash
# 1. Run baseline sweep
python scripts/run_sweep.py --sweep-config examples/configs/sweep_multi_source.json --output baseline.json

# 2. Enable pair losses to assess energy drain
# Edit sweep config: "include_pair_losses": true
python scripts/run_sweep.py --sweep-config examples/configs/sweep_multi_source.json --output with_losses.json

# 3. Compare efficiency with and without losses
# (Use plotting scripts or manual JSON analysis)

# 4. Identify best configurations
# Filter for eff_Pgw_over_Pin > 1e-6 and h_rms > 1e-22

# 5. Run high-resolution time series for top candidates
python scripts/simulate_gravity_coupling.py --config examples/configs/optimized_config.json --plot
```

## Caveats and Limitations

- **Linearized gravity**: Only valid for h << 1; no strong-field or nonlinear GR effects
- **Homogeneous approximations**: Real beams have spatiotemporal structure, diffraction, dispersion
- **Schwinger rate**: Exponentially suppressed unless E ~ E_s; requires E > 10¹⁶ V/m for appreciable pairs
- **No self-consistent back-reaction**: We compute GW from EM, but don't feed h back into EM dynamics (future work: birefringence feedback)
- **No detector noise**: Reported h values assume perfect measurement; real detectors have strain noise ~ 10⁻²³ / √Hz

## Next Steps

- **Birefringence feedback**: Implement first-order correction where Δn(E) modulates local intensity → perturbs u → feeds back into quadrupole
- **Time-domain optimization**: Search for pulsed or chirped waveforms that maximize h(f) in narrow detector bands
- **Experimental proposals**: Identify configurations requiring "only" 10–100 TW peak power and 1–10 kJ total energy (within reach of next-gen laser facilities)

## References

- Heisenberg-Euler effective Lagrangian: Phys. Rev. D **2**, 2341 (1970)
- Schwinger pair production: Phys. Rev. **82**, 664 (1951)
- Quadrupole formula: Landau & Lifshitz, *Classical Theory of Fields*
- LIGO sensitivity: [LIGO Document P1200087](https://dcc.ligo.org/LIGO-P1200087/public)

---

**Happy hunting for spacetime back-reaction!**
