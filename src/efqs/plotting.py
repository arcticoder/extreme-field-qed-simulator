"""
Standardized plotting utilities for extreme-field QED and gravity coupling results.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional


def plot_strain_timeseries(t: np.ndarray, h_t: np.ndarray, title: str = "Gravitational Strain", 
                            save_path: Optional[str] = None) -> plt.Figure:
    """Plot strain tensor components over time.
    
    t: (T,) time array [s]
    h_t: (T, 3, 3) strain tensor time series
    title: plot title
    save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t, h_t[:, 0, 0], label='h_xx', alpha=0.8)
    ax.plot(t, h_t[:, 1, 1], label='h_yy', alpha=0.8)
    ax.plot(t, h_t[:, 2, 2], label='h_zz', alpha=0.8)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('h (dimensionless)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_gw_power(t: np.ndarray, P_t: np.ndarray, title: str = "Gravitational Wave Power",
                  save_path: Optional[str] = None) -> plt.Figure:
    """Plot radiated GW power over time.
    
    t: (T,) time array [s]
    P_t: (T,) power time series [W]
    title: plot title
    save_path: optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(t, P_t, label='P_GW', color='C3', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Power [W]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_strain_and_power(t: np.ndarray, h_t: np.ndarray, P_t: np.ndarray,
                          title: str = "Gravitational Coupling Results",
                          save_path: Optional[str] = None) -> plt.Figure:
    """Combined plot of strain and power.
    
    t: (T,) time array [s]
    h_t: (T, 3, 3) strain tensor time series
    P_t: (T,) power time series [W]
    title: plot title
    save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True, sharex=True)
    
    # Strain plot
    axes[0].plot(t, h_t[:, 0, 0], label='h_xx', alpha=0.8)
    axes[0].plot(t, h_t[:, 1, 1], label='h_yy', alpha=0.8)
    axes[0].plot(t, h_t[:, 2, 2], label='h_zz', alpha=0.8)
    axes[0].set_ylabel('h (dimensionless)')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Power plot
    axes[1].plot(t, P_t, color='C3', linewidth=1.5, label='P_GW')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Power [W]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_sweep_results(results: list[Dict[str, Any]], x_key: str, y_key: str,
                       xlabel: str = None, ylabel: str = None, 
                       title: str = "Parameter Sweep",
                       log_x: bool = False, log_y: bool = False,
                       save_path: Optional[str] = None) -> plt.Figure:
    """Plot results from a parameter sweep.
    
    results: list of dicts from sweep output
    x_key: key for x-axis values
    y_key: key for y-axis values
    xlabel/ylabel: axis labels (defaults to keys)
    title: plot title
    log_x/log_y: use log scale
    save_path: optional path to save figure
    """
    x_vals = [r[x_key] for r in results]
    y_vals = [r[y_key] for r in results]
    
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.plot(x_vals, y_vals, 'o-', markersize=6, linewidth=1.5)
    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_energy_density_slice(grid, u: np.ndarray, z_slice: float = 0.0,
                               title: str = "Energy Density",
                               save_path: Optional[str] = None) -> plt.Figure:
    """Plot 2D slice of energy density at constant z.
    
    grid: Grid object
    u: (Nx, Ny, Nz) energy density array [J/m^3]
    z_slice: z-coordinate for slice [m]
    title: plot title
    save_path: optional path to save figure
    """
    # Find closest z index
    z_idx = np.argmin(np.abs(grid.z - z_slice))
    
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.pcolormesh(grid.x, grid.y, u[:, :, z_idx].T, shading='auto', cmap='hot')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f"{title} (z = {grid.z[z_idx]:.3e} m)")
    ax.set_aspect('equal')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('u [J/m³]')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
