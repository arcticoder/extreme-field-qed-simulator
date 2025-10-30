"""Extreme-Field QED Simulator (efqs)

Modules:
- constants: physical constants in SI
- heisenberg_euler: vacuum birefringence (Δn) via Heisenberg–Euler Lagrangian
- vacuum_birefringence: helper functions to compute phase and ellipticity
- pair_production: Schwinger pair production rate estimate
"""

from . import constants, heisenberg_euler, vacuum_birefringence, pair_production

__all__ = [
    "constants",
    "heisenberg_euler",
    "vacuum_birefringence",
    "pair_production",
]
