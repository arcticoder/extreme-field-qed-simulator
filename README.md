# Extreme-Field QED Simulator

Simulate ultra-intense light interacting with quantum vacuum structure and spacetime:
- **Vacuum birefringence** via Heisenberg–Euler effective Lagrangian
- **Photon–photon scattering** cross sections (order-of-magnitude)
- **Schwinger pair production** rates near the critical field
- **Linearized gravitational coupling**: EM stress–energy → metric perturbations
- **Gravitational wave estimates**: far-field strain and radiated power

This is a sandbox for "intense EM fields meet spacetime" exploration. It targets near-term, testable regimes (QED nonlinearities, vacuum polarization control) and provides a path-finder for field–spacetime interaction physics.

## Quick start

```bash
git clone https://github.com/arcticoder/extreme-field-qed-simulator.git
cd extreme-field-qed-simulator
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Vacuum birefringence (probe through strong field region)
python scripts/simulate_birefringence.py --E0 1e15 --L 1.0 --lambda0 532e-9 --theta 45 --cavity-gain 1e6

# Schwinger pair production estimate
python scripts/simulate_pair_production.py --E0 1e16 --volume 1e-12 --duration 1e-12

# Gravitational coupling from EM fields
python scripts/simulate_gravity_coupling.py --config examples/configs/interfering_pulses.json

# Run parameter sweep
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
- You can save results to JSON by adding `"output_json": "path/to/results.json"` in the config.
- Toggle pair-production losses in the time evolution via `"include_pair_losses": true` (uses a uniform drain based on Schwinger rate at effective field).

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

## Documentation
- [Back-Reaction Assessment Guide](docs/BackReactionGuide.md) – How to interpret metrics and run systematic sweeps to assess EM ↔ spacetime viability

## License
MIT
