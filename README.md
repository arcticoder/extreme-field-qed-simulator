# Extreme-Field QED Simulator

Simulate ultra-intense light interacting with quantum vacuum structure:
- Vacuum birefringence via Heisenberg–Euler effective Lagrangian
- Photon–photon scattering cross sections (order-of-magnitude)
- Schwinger pair production rates near the critical field

This is a sandbox for “intense EM fields meet spacetime” exploration. It targets near-term, testable regimes (QED nonlinearities, vacuum polarization control) and avoids speculative gravity claims.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Vacuum birefringence (probe through strong field region)
python scripts/simulate_birefringence.py --E0 1e15 --L 1.0 --lambda0 532e-9 --theta 45 --cavity-gain 1e6

# Schwinger pair production estimate
python scripts/simulate_pair_production.py --E0 1e16 --volume 1e-12 --duration 1e-12
```

## Features
- Heisenberg–Euler Δn for orthogonal polarizations with arbitrary E and B
- Phase retardation and induced ellipticity for a probe beam
- Simple Schwinger-rate pair production estimator (warning: exponentially suppressed unless E ~ E_S)
- Constants in SI; vectorized API with NumPy

## Caution
These models are leading-order, homogeneous-field approximations. Real experiments involve spatiotemporal structure, dispersion, damage thresholds, and noise.

## License
MIT
