"""
Sensitivity and detectability analysis module.

Computes detection metrics and compares to realistic detector thresholds:
- Gravitational wave strain h at various distances
- Quantum-sensor thresholds (Tobar et al. single-graviton proposals)
- Classical interferometer sensitivities (LIGO, LISA, etc.)
- Force/momentum transfer measurements (torsion balance, etc.)

Provides tools to answer: "How large must κ be for detection?"
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from .detector_noise import DETECTOR_NOISE_CURVES, matched_filter_snr
    HAS_NOISE_CURVES = True
except ImportError:
    HAS_NOISE_CURVES = False


@dataclass
class DetectorSensitivity:
    """Container for detector specifications."""
    name: str
    strain_threshold: float  # Minimum detectable h (dimensionless)
    frequency_band: Tuple[float, float]  # (f_min, f_max) [Hz]
    integration_time: float  # Observation time [s]
    description: str = ""


# Standard detector sensitivities (representative values)
DETECTORS = {
    "LIGO_O1": DetectorSensitivity(
        name="LIGO O1",
        strain_threshold=1e-21,
        frequency_band=(10.0, 5000.0),
        integration_time=3600.0,  # 1 hour
        description="Initial LIGO observing run sensitivity"
    ),
    "aLIGO_design": DetectorSensitivity(
        name="Advanced LIGO Design",
        strain_threshold=1e-22,
        frequency_band=(10.0, 5000.0),
        integration_time=3600.0,
        description="Advanced LIGO design sensitivity"
    ),
    "LISA": DetectorSensitivity(
        name="LISA",
        strain_threshold=1e-20,
        frequency_band=(1e-4, 1.0),
        integration_time=3.15e7,  # 1 year
        description="Laser Interferometer Space Antenna"
    ),
    "ET_design": DetectorSensitivity(
        name="Einstein Telescope",
        strain_threshold=1e-23,
        frequency_band=(1.0, 10000.0),
        integration_time=3600.0,
        description="Next-generation ground-based detector"
    ),
    "quantum_sensor_aspirational": DetectorSensitivity(
        name="Quantum Sensor (Aspirational)",
        strain_threshold=1e-30,
        frequency_band=(1.0, 1e6),
        integration_time=1.0,
        description="Tobar et al. single-graviton quantum sensing proposal"
    ),
    "tabletop_interferometer": DetectorSensitivity(
        name="Tabletop Interferometer",
        strain_threshold=1e-18,
        frequency_band=(0.1, 1e4),
        integration_time=100.0,
        description="Realistic table-top optical interferometer"
    ),
    "torsion_balance": DetectorSensitivity(
        name="Cryogenic Torsion Balance",
        strain_threshold=1e-22,  # Equivalent strain from force measurement
        frequency_band=(1e-5, 1.0),
        integration_time=1e4,
        description="Ultra-sensitive force measurement"
    ),
}


def strain_at_distance(h_source: float, R_source: float, R_target: float) -> float:
    """Scale strain from source distance to target distance.
    
    h scales as 1/R in the far field.
    
    h_source: strain at R_source
    R_source: source distance [m]
    R_target: target distance [m]
    Returns: strain at target distance
    """
    return h_source * (R_source / R_target)


def snr_estimate(h_signal: float, h_noise: float, integration_time: float, 
                bandwidth: float) -> float:
    """Estimate signal-to-noise ratio for GW detection.
    
    SNR ~ (h_signal / h_noise) * sqrt(T * BW)
    
    h_signal: signal strain amplitude
    h_noise: noise strain spectral density
    integration_time: observation time [s]
    bandwidth: frequency bandwidth [Hz]
    Returns: estimated SNR
    """
    return (h_signal / h_noise) * np.sqrt(integration_time * bandwidth)


def snr_from_noise_curve(h_timeseries: np.ndarray, dt: float, detector_name: str,
                        integration_time: float = 1.0) -> float:
    """Compute matched-filter SNR using real detector noise curves.
    
    h_timeseries: strain time series (can be full tensor; will use trace)
    dt: time step [s]
    detector_name: detector name (must be in DETECTOR_NOISE_CURVES)
    integration_time: observation time [s]
    
    Returns: SNR from matched filtering against detector noise
    """
    if not HAS_NOISE_CURVES:
        raise ImportError("detector_noise module not available; cannot compute noise-curve SNR")
    
    # Extract scalar strain (trace or specific component)
    if h_timeseries.ndim > 1:
        # Assume (..., 3, 3) tensor; take trace
        h_scalar = np.trace(h_timeseries, axis1=-2, axis2=-1)
    else:
        h_scalar = h_timeseries
    
    # FFT of strain signal
    N = len(h_scalar)
    h_fft = np.fft.rfft(h_scalar)
    freqs = np.fft.rfftfreq(N, dt)
    
    # Use matched filter SNR
    snr = matched_filter_snr(h_fft, freqs, detector_name, integration_time)
    
    return snr


def is_detectable(h_rms: float, freq_peak: float, detector: DetectorSensitivity,
                 R_source: float = 10.0, R_detector: float = None,
                 required_snr: float = 5.0) -> Tuple[bool, float]:
    """Determine if signal is detectable by given detector.
    
    Parameters:
    -----------
    h_rms : RMS strain at source distance
    freq_peak : dominant frequency [Hz]
    detector : DetectorSensitivity object
    R_source : distance at which h_rms is computed [m]
    R_detector : actual distance to detector [m] (if None, use R_source)
    required_snr : minimum SNR for detection (default 5)
    
    Returns:
    --------
    is_detected : bool
    snr : estimated signal-to-noise ratio
    """
    if R_detector is None:
        R_detector = R_source
    
    # Scale strain to detector distance
    h_at_detector = strain_at_distance(h_rms, R_source, R_detector)
    
    # Check frequency band
    if not (detector.frequency_band[0] <= freq_peak <= detector.frequency_band[1]):
        return False, 0.0
    
    # Compute SNR
    bandwidth = detector.frequency_band[1] - detector.frequency_band[0]
    snr = snr_estimate(h_at_detector, detector.strain_threshold,
                      detector.integration_time, bandwidth)
    
    is_detected = (snr >= required_snr)
    
    return is_detected, snr


def required_energy_for_detection(h_per_joule: float, detector: DetectorSensitivity,
                                  R_source: float = 10.0,
                                  R_detector: float = None) -> float:
    """Compute EM energy required for detectable GW signal.
    
    h_per_joule : strain produced per joule of EM energy at R_source
    detector : target detector
    R_source : distance at which h_per_joule is calibrated [m]
    R_detector : actual detector distance [m]
    
    Returns: required energy [J]
    """
    if R_detector is None:
        R_detector = R_source
    
    # Scale threshold to source distance
    h_threshold_at_source = strain_at_distance(
        detector.strain_threshold, R_detector, R_source
    )
    
    # Energy required
    if h_per_joule > 0:
        E_required = h_threshold_at_source / h_per_joule
    else:
        E_required = np.inf
    
    return E_required


def quantum_sensing_limits(h_signal: float, freq: float, 
                           mass_test: float = 1e-6,
                           temperature: float = 1e-3) -> Dict[str, float]:
    """Estimate quantum sensing detectability (Tobar et al. framework).
    
    Based on proposals for graviton detection via quantum resonators.
    This is highly aspirational and assumes optimistic quantum-limited performance.
    
    Parameters:
    -----------
    h_signal : GW strain amplitude
    freq : GW frequency [Hz]
    mass_test : test mass [kg] (default 1 μg microresonator)
    temperature : operating temperature [K] (default 1 mK)
    
    Returns:
    --------
    Dict with:
        - displacement: induced displacement [m]
        - phonon_occupation: thermal phonon number
        - quantum_limited_sensitivity: fundamental quantum limit on h
    """
    from efqs.constants import hbar, c
    
    # GW-induced displacement: δx ~ h * L where L is detector size
    # For resonator: δx ~ h * (c / ω_res)
    omega = 2.0 * np.pi * freq
    L_eff = c / omega  # Effective length scale
    
    displacement = h_signal * L_eff
    
    # Thermal phonon occupation
    k_B = 1.380649e-23  # Boltzmann constant
    n_thermal = k_B * temperature / (hbar * omega) if omega > 0 else np.inf
    
    # Quantum-limited sensitivity (SQL for resonator)
    # h_SQL ~ sqrt(ħ / (m ω² L²))
    if mass_test > 0 and omega > 0:
        h_quantum_limit = np.sqrt(hbar / (mass_test * omega**2 * L_eff**2))
    else:
        h_quantum_limit = 0.0
    
    return {
        'displacement_m': displacement,
        'phonon_occupation': n_thermal,
        'quantum_limited_h': h_quantum_limit,
        'above_quantum_limit': h_signal > h_quantum_limit,
    }


def sensitivity_report(h_rms: float, h_max: float, freq_peak: float,
                      P_GW: float, E_EM: float,
                      R_source: float = 10.0) -> Dict[str, Any]:
    """Generate comprehensive sensitivity report across all detectors.
    
    Parameters:
    -----------
    h_rms, h_max : strain metrics
    freq_peak : dominant frequency [Hz]
    P_GW : radiated GW power [W]
    E_EM : total EM energy [J]
    R_source : distance at which h is computed [m]
    
    Returns:
    --------
    Dict with detectability for each detector and key metrics
    """
    h_per_joule = h_rms / E_EM if E_EM > 0 else 0.0
    
    report = {
        'source_metrics': {
            'h_rms': h_rms,
            'h_max': h_max,
            'freq_peak_Hz': freq_peak,
            'P_GW_W': P_GW,
            'E_EM_J': E_EM,
            'h_per_joule': h_per_joule,
            'R_source_m': R_source,
        },
        'detector_analysis': {},
        'quantum_sensing': {},
    }
    
    # Check each detector
    for det_name, detector in DETECTORS.items():
        # Various realistic distances
        for R_det in [1.0, 10.0, 100.0, 1000.0]:
            is_det, snr = is_detectable(h_rms, freq_peak, detector,
                                       R_source=R_source, R_detector=R_det)
            
            E_req = required_energy_for_detection(h_per_joule, detector,
                                                 R_source=R_source, R_detector=R_det)
            
            report['detector_analysis'][f'{det_name}_R{R_det}m'] = {
                'detectable': is_det,
                'snr': snr,
                'required_energy_J': E_req,
                'h_at_detector': strain_at_distance(h_rms, R_source, R_det),
            }
    
    # Quantum sensing estimates
    for mass in [1e-9, 1e-6, 1e-3]:  # ng, μg, mg test masses
        qs = quantum_sensing_limits(h_rms, freq_peak, mass_test=mass, temperature=1e-3)
        report['quantum_sensing'][f'mass_{mass:.0e}kg'] = qs
    
    return report


def kappa_detection_matrix(h_EM_baseline: float, 
                          detection_targets: Dict[str, float],
                          T00_EM: float, F_characteristic: float) -> Dict[str, float]:
    """Compute κ required to reach each detection threshold.
    
    Given baseline EM-only strain h_EM and target thresholds,
    compute anomalous coupling κ needed to boost signal.
    
    Parameters:
    -----------
    h_EM_baseline : strain from pure EM stress-energy
    detection_targets : dict of {name: h_threshold}
    T00_EM : average EM energy density [J/m³]
    F_characteristic : typical magnitude of anomalous functional
    
    Returns:
    --------
    Dict of {target_name: κ_required}
    """
    kappa_matrix = {}
    
    for name, h_target in detection_targets.items():
        if h_EM_baseline >= h_target:
            kappa_matrix[name] = 0.0  # Already detectable
        else:
            # h ∝ T00, so boost_factor = h_target / h_EM
            boost = h_target / h_EM_baseline
            
            # T00_eff = T00_EM + κ F
            # boost = T00_eff / T00_EM = 1 + (κ F / T00_EM)
            # => κ = (boost - 1) * T00_EM / F
            
            kappa_required = (boost - 1.0) * T00_EM / (F_characteristic + 1e-50)
            kappa_matrix[name] = kappa_required
    
    return kappa_matrix
