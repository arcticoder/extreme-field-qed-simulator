# Anomalous Coupling (κ) Modules

This simulator parameterizes beyond-EM/GR couplings with κ-style coefficients. Each module implements a specific physics-motivated ansatz and returns a source term used to infer the coupling strength needed for detectability.

## Why κ?
- Encodes “how strong would this new coupling need to be” for a given experiment
- Enables null-result constraints: if no signal observed, publish κ < κ_required
- Keeps units explicit and model-local (don’t compare κ across different ansätze)

## Available κ Modules (examples)
- `axion_like`: ∝ E·B (parity-odd)
- `dilaton_like`: ∝ Tr(T^μ_μ) (scalar-tensor–like)
- `chern_simons_like`: ∝ A·B (Lorentz-violating)
- `field_invariant_F2`: ∝ F^μν F_μν = 2(B² − E²/c²)
- `vector_potential_squared`: ∝ A² (gauge-fixing dependent; for toy studies)
- `photon_number`: ∝ |E|² at a reference frequency band
- `spatial_gradient`: ∝ |∇E|² or |∇B|² (structure-induced coupling)

## Units and Scaling
- κ conventions are chosen so that a larger κ implies a larger signal
- Typical scalings in weak-field GR: h ∝ κ × (field)^2, so κ_required ∝ 1/(field)^2
- Always check the docstring of a κ module for its exact units

## Detector Mapping
We convert predicted h(t) into SNR using detector spectral densities S_n(f):

SNR² = 4 ∫ |h̃(f)|² / S_n(f) df.

κ_required is set so that SNR reaches a chosen threshold (e.g., 5σ).

## Extending κ Modules
1. Copy a template from `src/couplings/` and rename appropriately
2. Document units in the module docstring
3. Add a test and a small YAML example under `examples/configs/`
4. Include the module in a sweep via `configs/sweeps.yaml`

