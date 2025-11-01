# Extreme-Field QED Simulator

## Status
- Stable research sandbox; actively maintained
- New: gravitational-coupling plugin (EM stress–energy → h(t), P_GW)
- YAML/JSON experiments with automated sweeps

## Capabilities
- Heisenberg–Euler vacuum birefringence and photon–photon scattering (O(1) estimates)
- Schwinger pair production (order-of-magnitude)
- Linearized GR coupling: EM stress–energy → quadrupole → h(t), P_GW
- Detector models and κ-constraint “discovery engine” sweeps

> See also: docs/PROVENANCE.md for reference provenance and how to add new sources.

Simulate ultra-intense light interacting with quantum vacuum structure and spacetime:
- **Vacuum birefringence** via Heisenberg–Euler effective Lagrangian
- **Photon–photon scattering** cross sections (order-of-magnitude)
- **Schwinger pair production** rates near the critical field
- **Linearized gravitational coupling**: EM stress–energy → metric perturbations
- **Gravitational wave estimates**: far-field strain and radiated power

This is a sandbox for "intense EM fields meet spacetime" exploration. It targets near-term, testable regimes (QED nonlinearities, vacuum polarization control) and provides a path-finder for field–spacetime interaction physics.

## Quick start (reproducible)

```bash
git clone https://github.com/arcticoder/extreme-field-qed-simulator.git
cd extreme-field-qed-simulator
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Minimal gravitational-coupling sanity run (tiny h_rms output)
python scripts/simulate_gravity_coupling.py --config examples/configs/interfering_pulses.json

# Expected: a small RMS strain (e.g., h_rms ~ 1e-58 to 1e-60) printed to stdout

# Optional: quick birefringence and pair-production demos
python scripts/simulate_birefringence.py --E0 1e15 --L 1.0 --lambda0 532e-9 --theta 45 --cavity-gain 1e6
python scripts/simulate_pair_production.py --E0 1e16 --volume 1e-12 --duration 1e-12

# Sweeps (see configs for details)
python scripts/run_sweep.py --sweep-config examples/configs/sweep_gaussian_beam.json --output results.json
```

## Features

### Quantum Vacuum Physics
- Heisenberg–Euler Δn for orthogonal polarizations with arbitrary E and B
- Phase retardation and induced ellipticity for a probe beam
- Simple Schwinger-rate pair production estimator (warning: exponentially suppressed unless E ~ E_S)

### Gravitational Coupling (NEW)
- EM stress–energy tensor from E, B fields (with optional QED corrections)
- Quadrupole moment calculation from energy distributions
- Far-field gravitational strain with proper TT projection
- Radiated GW power via quadrupole formula
- Optional pair-production energy-loss channel in time evolution (uniform drain approximation)
- Multiple source models:
  - Interfering laser pulses (standing waves)
  - Rotating quadrupole hotspots
  - Focused Gaussian beams
  - Rotating capacitor (Biefeld-Brown-like geometry)

### Infrastructure
- Installable package (`pip install -e .`)
- JSON/YAML configuration support
- Automated parameter sweeps
- Reusable plotting utilities
- All tests passing (5/5)

### Coupling Metrics and Outputs
- The gravity coupling script now computes:
  - RMS strain (time-averaged magnitude)
  - Average GW power
  - Efficiency metrics: P_GW/P_in and h per Joule
  - Frequency spectrum: dominant frequency, amplitude, bandwidth via FFT
- You can save results to JSON by adding `"output_json": "path/to/results.json"` in the config.
- Toggle pair-production losses in the time evolution via `"include_pair_losses": true` (uses a uniform drain based on Schwinger rate at effective field).
- Toggle birefringence feedback via `"birefringence_feedback": true` (first-order Δn modulation of local energy density).

## Physics Background

### Linearized Einstein Equation

For weak gravitational fields, the metric perturbation h_μν satisfies:

$$\\Box h_{\\mu\\nu} = -16\\pi G T_{\\mu\\nu}$$

where T_μν is computed from EM fields. In the far-field (radiation zone), the strain is:

$$h_{ij}(t, \\mathbf{R}) \\approx \\frac{2G}{c^4 R} \\ddot{Q}_{ij}(t - R/c)$$

with proper TT projection for a given line of sight.

### Radiated GW Power

The radiated gravitational wave power is:

$$P_{\\text{GW}} = \\frac{G}{5c^5} \\langle \\dddot{Q}_{ij} \\dddot{Q}_{ij} \\rangle$$

### QED Corrections

Heisenberg–Euler energy density correction (leading order):

$$\\Delta u \\approx \\frac{2}{45} \\frac{\\alpha}{\\pi} \\frac{1}{E_s^2} \\left[ 4F^2 + 28c^2 G^2 \\right]$$

where F and G are the electromagnetic field invariants.

## Example Outputs

Typical run with interfering pulses at E₀ = 10¹⁵ V/m:
- RMS strain: h ~ 10⁻²² (dimensionless) at R = 10 m
- Average P_GW: ~ 10²³ W (toy configuration; not physical lab setup)

These are path-finder estimates for comparing geometries and configurations, not precision predictions.

## Caution
These models are leading-order, homogeneous-field approximations. Real experiments involve spatiotemporal structure, dispersion, damage thresholds, and noise.

## Discovery Engine: Systematic New Physics Search

This framework has evolved into a **discovery engine** for constraining new physics via null results. Instead of just computing predictions, it systematically derives **coupling strength constraints** (κ-bounds) when experiments see no signal.

### Core Capabilities

1. **κ-Parameterized Ansätze** (7 physics-motivated models):
   - `axion_like`: E·B parity-odd coupling (compare to CAST/ADMX)
   - `dilaton_like`: T^μ_μ scalar-tensor gravity
   - `chern_simons_like`: A·B Lorentz violation (SME framework)
   - `field_invariant_F2`, `vector_potential_squared`, `photon_number`, `spatial_gradient`

2. **Real Detector Models** with frequency-dependent sensitivity:
   - LIGO O1, aLIGO design, LISA, Einstein Telescope
   - Quantum sensors (aspirational h~10⁻³⁰)
   - Matched-filter SNR: `SNR² = 4∫|h̃(f)|²/S_n(f) df`

3. **Automated Parameter Sweeps**:
   ```bash
   python scripts/run_experiments.py --config configs/sweeps.yaml --sweep sweep_E0_colliding_pulses
   ```
   - Outputs consolidated CSV with κ-constraints per ansatz–detector pair
   - Auto-generated 4-panel plots: h_rms, P_avg, κ_required, peak_freq vs. swept parameter
   - Sweep E₀, waist, Q-factor, grid resolution, etc.

4. **Publication-Ready Constraint Derivation**:
   - For each (source config, detector, ansatz): compute κ_required for 5σ detection
   - **Null result** → publish: "κ < κ_required at 95% CL"
   - Discovery reach curves: show parameter space where future experiments can probe

### Quick Start: Discovery Mode

```bash
# Run a 3-point validation sweep (E₀ = 5×10¹³, 1×10¹⁴, 2×10¹⁴ V/m)
python scripts/run_experiments.py --config configs/test_mini_sweep.yaml --sweep test_mini_sweep

# Output:
# - results/sweeps/test_mini/test_mini_sweep_summary.csv (κ-constraints)
# - results/sweeps/test_mini/test_mini_sweep_plots.png (auto-plots)

# Production sweeps (7-point E₀, 6-point waist, 5-point cavity-Q)
python scripts/run_experiments.py --config configs/sweeps.yaml --sweep sweep_E0_colliding_pulses
python scripts/run_experiments.py --config configs/sweeps.yaml --sweep sweep_waist_colliding_pulses
python scripts/run_experiments.py --config configs/sweeps.yaml --sweep sweep_Q_cavity
```

### Expected Scaling (Validation Checkpoints)

- **Strain vs. Field**: `h ∝ E₀²` (quadrupole moment ∝ stress-energy ∝ E²)
- **Power vs. Field**: `P_GW ∝ E₀⁴` (time derivative of quadrupole)
- **κ-Constraint**: `κ_required ∝ 1/E₀²` (higher field → stronger constraint)

Example from `test_mini_sweep`:
```
E₀ = 5×10¹³ V/m  →  h_rms = 3.7×10⁻⁵⁹,  κ_LIGO = 2.3×10³³
E₀ = 2×10¹⁴ V/m  →  h_rms = 5.9×10⁻⁵⁸,  κ_LIGO = 8.9×10³⁰  (16× improvement)
```

### Interpreting Results

**Q**: My κ_required is 10⁵⁰. Is the code broken?

**A**: No! This means the anomalous coupling would need to be *enormous* to produce a detectable signal with current experiments. This is useful science:
- Confirms known physics (EM + GR) predicts negligible signal
- Sets an **upper bound**: null result → κ < 10⁵⁰
- Guides theory: if your model predicts κ ~ 10⁴⁵, experiment is relevant; if κ ~ 10²⁰, need different approach

**Q**: How do I compare ansätze?

**A**: **Don't compare κ-values across ansätze** (different units, different physics). Instead:
- Compare κ-constraints **within same ansatz** across different experiments
- Map κ to theory-specific couplings (e.g., axion g_aγγ, dilaton VEV, SME coefficients)

### Documentation & Guides

- **[Discovery Engine Guide](docs/DiscoveryEngineGuide.md)** – Complete methodology, ansätze catalog, sweep design, troubleshooting
- **[Back-Reaction Assessment Guide](docs/BackReactionGuide.md)** – How to interpret metrics and run systematic sweeps to assess EM ↔ spacetime viability
- **[Detector Sensitivity Curves](docs/detector_sensitivities.png)** – ASDs for LIGO/LISA/ET/quantum sensors

## Documentation
- [Quick Start](docs/quickstart.md) — minimal run, expected outputs, troubleshooting
- [Anomalous Coupling (κ) Modules](docs/anomalous_coupling.md) — parameterization and units
- [Discovery Engine Guide](docs/DiscoveryEngineGuide.md) – **Complete guide** to κ-constraint methodology, parameter sweeps, ansätze catalog, and publication workflow
- [Back-Reaction Assessment Guide](docs/BackReactionGuide.md) – How to interpret metrics and run systematic sweeps to assess EM ↔ spacetime viability
 - [Provenance Guide](docs/PROVENANCE.md) — how references are organized and how to add new ones for agent use

## License
MIT
