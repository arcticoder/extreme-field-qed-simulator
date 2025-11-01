# Quick Start

This short guide gets you a tiny, reproducible run that prints a very small gravitational strain `h_rms` to the console.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Minimal run

```bash
# Interfering pulses configuration (weak-field GR coupling)
python scripts/simulate_gravity_coupling.py --config examples/configs/interfering_pulses.json
```

Expected output (varies slightly by machine):

```
h_rms: ~1e-58 to 1e-60 (dimensionless)
P_GW_avg: very small (toy configuration)
```

If you want a quick vacuum birefringence demo:

```bash
python scripts/simulate_birefringence.py --E0 1e15 --L 1.0 --lambda0 532e-9 --theta 45 --cavity-gain 1e6
```

## Troubleshooting
- If you see `ModuleNotFoundError`, ensure your venv is active and `pip install -e .` succeeded.
- If a script prints `File not found`, verify the `--config` path.
- To save outputs, many scripts accept `--output` or `"output_json"` in configs.
