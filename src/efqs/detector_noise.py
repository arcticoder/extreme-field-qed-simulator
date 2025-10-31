"""
Real detector noise models for gravitational wave observatories.

Provides amplitude spectral density (ASD) and power spectral density (PSD) 
for major detectors, enabling realistic SNR calculations via matched filtering.

References:
- LIGO: https://dcc.ligo.org/LIGO-T1800044/public
- LISA: https://arxiv.org/abs/1702.00786
- Einstein Telescope: https://arxiv.org/abs/1012.0908
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable
from efqs.constants import c


@dataclass
class DetectorNoiseCurve:
    """Detector noise amplitude spectral density."""
    name: str
    frequency_band: tuple[float, float]  # [Hz] (f_min, f_max)
    asd_func: Callable[[np.ndarray], np.ndarray]  # f [Hz] -> h_n(f) [1/sqrt(Hz)]
    description: str


def aLIGO_design_asd(f: np.ndarray) -> np.ndarray:
    """Advanced LIGO design sensitivity ASD.
    
    Analytic fit from LIGO-T1800044 (simplified).
    Valid from ~10 Hz to 5 kHz.
    
    Returns: h_n(f) [strain / sqrt(Hz)]
    """
    f = np.asarray(f, dtype=float)
    
    # Seismic wall below 10 Hz
    seismic = 1e-23 * (10.0 / np.maximum(f, 1e-3))**4
    
    # Quantum noise (shot noise) above ~100 Hz
    shot = 2e-24 * (f / 100.0)**2
    
    # Thermal noise peak ~50-200 Hz
    thermal = 1.5e-24 * np.exp(-((f - 150.0) / 50.0)**2)
    
    # Combined (quadrature sum)
    h_n = np.sqrt(seismic**2 + shot**2 + thermal**2)
    
    return h_n


def LIGO_O1_asd(f: np.ndarray) -> np.ndarray:
    """LIGO O1 run sensitivity (2015).
    
    Roughly 3x worse than design at low frequencies.
    """
    return 3.0 * aLIGO_design_asd(f)


def LISA_asd(f: np.ndarray) -> np.ndarray:
    """LISA amplitude spectral density.
    
    Space-based detector: mHz range (10^-4 to 1 Hz).
    Analytic fit from arXiv:1702.00786.
    
    Returns: h_n(f) [strain / sqrt(Hz)]
    """
    f = np.asarray(f, dtype=float)
    
    # LISA arm length
    L = 2.5e9  # meters
    
    # Acceleration noise
    S_a = (3e-15)**2 * (1.0 + (0.4e-3 / f)**2) * (1.0 + (f / 8e-3)**4)
    
    # Optical metrology noise
    S_x = (1.5e-11)**2 * (1.0 + (2e-3 / f)**4)
    
    # Galactic confusion noise (simplified)
    S_conf = 0.0  # Detailed model requires time-varying component
    
    # Total PSD (single-link)
    S_n = (S_x + S_a * (2.0 * np.pi * f)**(-4)) / (L**2)
    
    # ASD
    h_n = np.sqrt(S_n)
    
    return h_n


def einstein_telescope_asd(f: np.ndarray) -> np.ndarray:
    """Einstein Telescope (ET-D) design sensitivity.
    
    Third-generation detector: ~1 Hz to 10 kHz.
    Factor of ~10 better than aLIGO at most frequencies.
    
    Returns: h_n(f) [strain / sqrt(Hz)]
    """
    f = np.asarray(f, dtype=float)
    
    # Simplified ET-D model
    # Seismic wall
    seismic = 1e-24 * (1.0 / np.maximum(f, 0.1))**5
    
    # Quantum noise
    shot = 1e-25 * (f / 100.0)**2
    
    # Thermal
    thermal = 8e-26 * np.exp(-((f - 200.0) / 100.0)**2)
    
    h_n = np.sqrt(seismic**2 + shot**2 + thermal**2)
    
    return h_n


def tabletop_interferometer_asd(f: np.ndarray) -> np.ndarray:
    """Generic tabletop interferometer.
    
    Assumes ~m-scale arms, kHz range, limited by shot noise.
    Typical h ~ 1e-18 @ 1 kHz for realistic laser power.
    """
    f = np.asarray(f, dtype=float)
    
    # Assume 1 m arms, 1 W laser, optimistic quantum limit
    L = 1.0  # meters
    P_laser = 1.0  # watts
    wavelength = 1064e-9  # Nd:YAG
    
    # Shot noise limit: h_n ~ sqrt(hbar c / (lambda P L^2)) at high f
    # Simplified model
    h_shot = 1e-18 * np.sqrt(1e3 / np.maximum(f, 100.0))
    
    # Environmental noise dominates below ~100 Hz
    env = 1e-16 * (100.0 / np.maximum(f, 10.0))**2
    
    h_n = np.sqrt(h_shot**2 + env**2)
    
    return h_n


# Registry of detector noise curves
DETECTOR_NOISE_CURVES = {
    "LIGO_O1": DetectorNoiseCurve(
        name="LIGO O1",
        frequency_band=(10.0, 5000.0),
        asd_func=LIGO_O1_asd,
        description="LIGO Observing Run 1 (2015) sensitivity"
    ),
    "aLIGO_design": DetectorNoiseCurve(
        name="Advanced LIGO Design",
        frequency_band=(10.0, 5000.0),
        asd_func=aLIGO_design_asd,
        description="Advanced LIGO design sensitivity (factor ~3 better than O1)"
    ),
    "LISA": DetectorNoiseCurve(
        name="LISA",
        frequency_band=(1e-4, 1.0),
        asd_func=LISA_asd,
        description="Laser Interferometer Space Antenna (mHz range)"
    ),
    "Einstein_Telescope": DetectorNoiseCurve(
        name="Einstein Telescope",
        frequency_band=(1.0, 10000.0),
        asd_func=einstein_telescope_asd,
        description="Third-generation ground-based detector (ET-D configuration)"
    ),
    "Tabletop": DetectorNoiseCurve(
        name="Tabletop Interferometer",
        frequency_band=(100.0, 10000.0),
        asd_func=tabletop_interferometer_asd,
        description="Generic table-top interferometer with meter-scale arms"
    ),
}


def matched_filter_snr(h_signal: np.ndarray, f_signal: np.ndarray, 
                       detector: str, integration_time: float = 1.0) -> float:
    """Compute matched-filter SNR for a signal in detector noise.
    
    SNR^2 = 4 ∫ |h_tilde(f)|^2 / S_n(f) df
    
    where h_tilde is the Fourier transform of the strain signal.
    
    Parameters:
    -----------
    h_signal : strain time series (assumes regular sampling)
    f_signal : frequency array corresponding to FFT of h_signal [Hz]
    detector : detector name (key in DETECTOR_NOISE_CURVES)
    integration_time : observation time [s]
    
    Returns:
    --------
    snr : matched-filter signal-to-noise ratio
    """
    if detector not in DETECTOR_NOISE_CURVES:
        raise ValueError(f"Unknown detector: {detector}")
    
    det = DETECTOR_NOISE_CURVES[detector]
    
    # Filter frequencies to detector band
    f_min, f_max = det.frequency_band
    mask = (f_signal >= f_min) & (f_signal <= f_max)
    
    if not np.any(mask):
        return 0.0  # Signal outside detector band
    
    f_band = f_signal[mask]
    h_tilde_band = h_signal[mask]
    
    # Get noise ASD at these frequencies
    h_n = det.asd_func(f_band)
    
    # Matched filter integral: SNR^2 = 4 ∫ |h_tilde|^2 / S_n df
    # S_n(f) = h_n(f)^2 (PSD = ASD^2)
    integrand = np.abs(h_tilde_band)**2 / h_n**2
    
    # Trapezoidal integration
    df = f_band[1] - f_band[0] if len(f_band) > 1 else 1.0
    snr_squared = 4.0 * np.trapz(integrand, dx=df)
    
    # Scale by integration time (longer observations improve SNR)
    snr = np.sqrt(snr_squared * integration_time)
    
    return float(snr)


def characteristic_strain_sensitivity(detector: str, f: np.ndarray, 
                                     integration_time: float = 1.0) -> np.ndarray:
    """Compute characteristic strain h_c(f) for 5-sigma detection.
    
    h_c(f) = SNR_threshold * h_n(f) / sqrt(integration_time)
    
    Useful for plotting detector sensitivity curves.
    
    Parameters:
    -----------
    detector : detector name
    f : frequency array [Hz]
    integration_time : observation time [s]
    
    Returns:
    --------
    h_c : characteristic strain amplitude for SNR=5 detection
    """
    if detector not in DETECTOR_NOISE_CURVES:
        raise ValueError(f"Unknown detector: {detector}")
    
    det = DETECTOR_NOISE_CURVES[detector]
    h_n = det.asd_func(f)
    
    SNR_threshold = 5.0  # 5-sigma detection
    h_c = SNR_threshold * h_n / np.sqrt(integration_time)
    
    return h_c


def plot_detector_sensitivities(f_min: float = 1e-4, f_max: float = 1e4, 
                                save_path: str | None = None):
    """Plot ASD curves for all detectors.
    
    Useful for visualizing frequency coverage and sensitivity.
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Frequency array spanning all detectors
    f = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
    
    for det_name, det in DETECTOR_NOISE_CURVES.items():
        f_band_mask = (f >= det.frequency_band[0]) & (f <= det.frequency_band[1])
        f_band = f[f_band_mask]
        
        if len(f_band) > 0:
            h_n = det.asd_func(f_band)
            ax.loglog(f_band, h_n, linewidth=2, label=det.name)
    
    ax.set_xlabel('Frequency [Hz]', fontsize=14)
    ax.set_ylabel('Strain ASD [1/√Hz]', fontsize=14)
    ax.set_title('Gravitational Wave Detector Sensitivities', fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(f_min, f_max)
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved detector sensitivity curves to {save_path}")
    
    return fig, ax
