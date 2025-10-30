import numpy as np

# Fundamental constants (SI)
alpha = 7.2973525693e-3        # Fine-structure constant
c = 299_792_458.0              # Speed of light [m/s]
mu0 = 4e-7 * np.pi             # Vacuum permeability [N/A^2]
epsilon0 = 1.0 / (mu0 * c**2)  # Vacuum permittivity [F/m]
hbar = 1.054_571_817e-34       # Reduced Planck constant [J s]
me = 9.109_383_7015e-31        # Electron mass [kg]
e_charge = 1.602_176_634e-19   # Elementary charge [C]

# Critical (Schwinger) fields
E_s = (me**2 * c**3) / (e_charge * hbar)  # ~ 1.32e18 V/m
B_c = E_s / c                              # ~ 4.41e9 T

# Heisenberg–Euler coefficients for vacuum birefringence (low-energy limit)
# See, e.g., Dittrich & Gies (2000): n_parallel - 1 = (7/2) A (B/B_c)^2, n_perp - 1 = 2 A (B/B_c)^2
# with A = alpha/(45*pi)
A_he = alpha / (45.0 * np.pi)
