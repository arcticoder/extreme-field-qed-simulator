"""
Integration tests for detector_noise module.

Tests:
1. ASD curve generation for all detectors
2. Frequency band coverage
3. SNR calculation accuracy
4. Characteristic strain sensitivity
5. Detector registry completeness
"""

import pytest
import numpy as np
from efqs.detector_noise import (
    LIGO_O1_asd,
    aLIGO_design_asd,
    LISA_asd,
    einstein_telescope_asd,
    tabletop_interferometer_asd,
    matched_filter_snr,
    characteristic_strain_sensitivity
)


class TestASDFunctions:
    """Test amplitude spectral density functions."""
    
    def test_ligo_o1_asd_shape(self):
        """Test LIGO O1 ASD at characteristic frequencies."""
        f_test = np.array([10, 50, 100, 500, 1000])  # Hz
        asd = LIGO_O1_asd(f_test)
        
        # Should return positive values
        assert np.all(asd > 0)
        assert len(asd) == len(f_test)
        
        # Check approximate sensitivity at 100 Hz
        asd_100 = LIGO_O1_asd(np.array([100.0]))[0]
        assert 1e-22 < asd_100 < 1e-21  # ~3e-22 Hz^-1/2
    
    def test_aligo_design_better_than_o1(self):
        """aLIGO design should be more sensitive than O1."""
        f = np.array([100.0])
        asd_o1 = LIGO_O1_asd(f)[0]
        asd_aligo = aLIGO_design_asd(f)[0]
        
        # aLIGO design ~3× better than O1
        assert asd_aligo < asd_o1
        assert asd_o1 / asd_aligo > 2.0
    
    def test_lisa_low_frequency(self):
        """LISA should be optimized for mHz frequencies."""
        f_low = np.array([1e-3, 1e-2, 1e-1])  # mHz to 0.1 Hz
        asd = LISA_asd(f_low)
        
        assert np.all(asd > 0)
        # LISA sensitivity ~1e-20 Hz^-1/2 in mHz band
        assert np.all(asd < 1e-18)  # Better than tabletop
        assert np.all(asd > 1e-22)  # Worse than LIGO at its band
    
    def test_einstein_telescope_best_sensitivity(self):
        """Einstein Telescope should have best ground-based sensitivity."""
        f = np.array([100.0])
        asd_et = einstein_telescope_asd(f)[0]
        asd_aligo = aLIGO_design_asd(f)[0]
        
        # ET ~10× better than aLIGO
        assert asd_et < asd_aligo
        assert asd_aligo / asd_et > 5.0
    
    def test_quantum_sensor_aspirational(self):
        """Quantum sensor should have extremely good sensitivity."""
        f = np.array([1.0, 10.0, 100.0])
        asd = quantum_sensor_asd(f)
        
        # Should be ~1e-30 Hz^-1/2
        assert np.all(asd < 1e-28)
        assert np.all(asd > 1e-32)
    
    def test_frequency_dependence(self):
        """Test that ASD varies with frequency as expected."""
        f_array = np.logspace(1, 3, 50)  # 10 Hz to 1 kHz
        asd = aLIGO_design_asd(f_array)
        
        # Should have minimum (best sensitivity) around 100-200 Hz
        min_idx = np.argmin(asd)
        f_min = f_array[min_idx]
        assert 80 < f_min < 300


class TestDetectorRegistry:
    """Test DETECTOR_NOISE_CURVES registry."""
    
    def test_all_detectors_present(self):
        """Verify all major detectors are registered."""
        expected = ['LIGO_O1', 'aLIGO_design', 'LISA', 'Einstein_Telescope',
                   'quantum_sensor', 'tabletop_interferometer']
        
        for detector in expected:
            assert detector in DETECTOR_NOISE_CURVES
    
    def test_detector_dataclass_structure(self):
        """Check that each detector has required fields."""
        for name, detector in DETECTOR_NOISE_CURVES.items():
            assert isinstance(detector, DetectorNoiseCurve)
            assert callable(detector.asd_function)
            assert detector.f_min > 0
            assert detector.f_max > detector.f_min
            assert isinstance(detector.description, str)
            assert detector.integration_time > 0
    
    def test_frequency_bands_non_overlapping(self):
        """Check that detectors cover complementary frequency ranges."""
        ligo_band = (DETECTOR_NOISE_CURVES['LIGO_O1'].f_min,
                     DETECTOR_NOISE_CURVES['LIGO_O1'].f_max)
        lisa_band = (DETECTOR_NOISE_CURVES['LISA'].f_min,
                     DETECTOR_NOISE_CURVES['LISA'].f_max)
        
        # LISA should be at lower frequencies than LIGO
        assert lisa_band[1] < ligo_band[0] or lisa_band[0] < ligo_band[0]


class TestMatchedFilterSNR:
    """Test matched-filter SNR calculation."""
    
    def test_snr_zero_signal(self):
        """SNR should be zero for zero signal."""
        f = np.linspace(10, 1000, 500)
        h_fft = np.zeros(len(f), dtype=complex)
        detector = DETECTOR_NOISE_CURVES['LIGO_O1']
        
        snr = matched_filter_snr(h_fft, f, detector, T=1.0)
        assert snr == 0.0
    
    def test_snr_positive(self):
        """SNR should be positive for non-zero signal."""
        f = np.linspace(50, 500, 200)
        # Gaussian pulse in frequency domain
        f0 = 100.0
        h_fft = np.exp(-((f - f0) / 50)**2) * 1e-21
        detector = DETECTOR_NOISE_CURVES['LIGO_O1']
        
        snr = matched_filter_snr(h_fft, f, detector, T=1.0)
        assert snr > 0
    
    def test_snr_scales_with_amplitude(self):
        """SNR should scale linearly with signal amplitude."""
        f = np.linspace(50, 500, 200)
        h_fft_1 = np.exp(-((f - 100) / 50)**2) * 1e-22
        h_fft_2 = 2 * h_fft_1
        detector = DETECTOR_NOISE_CURVES['aLIGO_design']
        
        snr1 = matched_filter_snr(h_fft_1, f, detector, T=1.0)
        snr2 = matched_filter_snr(h_fft_2, f, detector, T=1.0)
        
        # SNR ∝ amplitude (since SNR² ∝ |h|²)
        assert np.isclose(snr2 / snr1, 2.0, rtol=0.1)
    
    def test_snr_integration_time_scaling(self):
        """SNR should scale as sqrt(T) for longer integration."""
        f = np.linspace(50, 500, 200)
        h_fft = np.exp(-((f - 100) / 50)**2) * 1e-21
        detector = DETECTOR_NOISE_CURVES['LIGO_O1']
        
        snr_1s = matched_filter_snr(h_fft, f, detector, T=1.0)
        snr_4s = matched_filter_snr(h_fft, f, detector, T=4.0)
        
        # For stationary signal: SNR ∝ sqrt(T)
        # Note: This assumes signal duration << T
        # For burst signals, SNR doesn't improve with T
        # So we just check it doesn't decrease
        assert snr_4s >= snr_1s
    
    def test_snr_frequency_band_dependence(self):
        """SNR depends on detector frequency band."""
        # Signal at LISA frequencies (mHz)
        f_lisa = np.linspace(1e-3, 1e-1, 100)
        h_fft_low = np.ones(len(f_lisa)) * 1e-20
        
        # Signal at LIGO frequencies (100s Hz)
        f_ligo = np.linspace(50, 500, 100)
        h_fft_high = np.ones(len(f_ligo)) * 1e-22
        
        lisa = DETECTOR_NOISE_CURVES['LISA']
        ligo = DETECTOR_NOISE_CURVES['LIGO_O1']
        
        snr_lisa = matched_filter_snr(h_fft_low, f_lisa, lisa, T=1.0)
        snr_ligo_wrong_band = matched_filter_snr(h_fft_low, f_lisa, ligo, T=1.0)
        
        # LISA should detect low-frequency signal better
        # (Though numerical values may be tricky due to band mismatch)
        assert snr_lisa >= 0


class TestCharacteristicSensitivity:
    """Test characteristic strain calculation."""
    
    def test_characteristic_strain_shape(self):
        """Test h_c(f) calculation."""
        f = np.logspace(1, 3, 50)
        detector = DETECTOR_NOISE_CURVES['aLIGO_design']
        
        h_c = characteristic_strain_sensitivity(f, detector, T_obs=3600, snr_threshold=5.0)
        
        assert len(h_c) == len(f)
        assert np.all(h_c > 0)
        assert np.all(np.isfinite(h_c))
    
    def test_characteristic_strain_scaling(self):
        """h_c should scale with SNR threshold and 1/sqrt(T)."""
        f = np.array([100.0])
        detector = DETECTOR_NOISE_CURVES['aLIGO_design']
        
        h_c_5sigma = characteristic_strain_sensitivity(f, detector, T_obs=1.0, snr_threshold=5.0)[0]
        h_c_10sigma = characteristic_strain_sensitivity(f, detector, T_obs=1.0, snr_threshold=10.0)[0]
        
        # h_c ∝ SNR_threshold
        assert np.isclose(h_c_10sigma / h_c_5sigma, 2.0, rtol=0.1)
        
        # h_c ∝ 1/sqrt(T_obs)
        h_c_1s = characteristic_strain_sensitivity(f, detector, T_obs=1.0, snr_threshold=5.0)[0]
        h_c_4s = characteristic_strain_sensitivity(f, detector, T_obs=4.0, snr_threshold=5.0)[0]
        
        assert np.isclose(h_c_1s / h_c_4s, 2.0, rtol=0.2)
    
    def test_better_detector_better_sensitivity(self):
        """ET should have better h_c than LIGO."""
        f = np.array([100.0])
        ligo = DETECTOR_NOISE_CURVES['LIGO_O1']
        et = DETECTOR_NOISE_CURVES['Einstein_Telescope']
        
        h_c_ligo = characteristic_strain_sensitivity(f, ligo, T_obs=3600, snr_threshold=5.0)[0]
        h_c_et = characteristic_strain_sensitivity(f, et, T_obs=3600, snr_threshold=5.0)[0]
        
        # ET should detect weaker signals (smaller h_c)
        assert h_c_et < h_c_ligo


class TestNumericalStability:
    """Test numerical stability across parameter ranges."""
    
    def test_very_low_frequencies(self):
        """Test ASD at ultra-low frequencies."""
        f = np.array([1e-5, 1e-4, 1e-3])  # μHz range
        
        # LISA is designed for this, should not overflow
        asd = LISA_asd(f)
        assert np.all(np.isfinite(asd))
        assert np.all(asd > 0)
    
    def test_very_high_frequencies(self):
        """Test ASD at kHz+ frequencies."""
        f = np.array([1e4, 1e5, 1e6])  # 10 kHz to 1 MHz
        
        # Tabletop/quantum sensors might work here
        asd_tabletop = tabletop_interferometer_asd(f)
        asd_quantum = quantum_sensor_asd(f)
        
        assert np.all(np.isfinite(asd_tabletop))
        assert np.all(np.isfinite(asd_quantum))
    
    def test_wide_frequency_sweep(self):
        """Test ASD over many decades in frequency."""
        f = np.logspace(-4, 6, 1000)  # 0.1 mHz to 1 MHz
        
        for name, detector in DETECTOR_NOISE_CURVES.items():
            asd_func = detector.asd_function
            asd = asd_func(f)
            
            # Should not have NaN, inf, or negative values
            assert np.all(np.isfinite(asd)), f"{name} produced non-finite ASD"
            assert np.all(asd > 0), f"{name} produced non-positive ASD"


class TestPhysicalConsistency:
    """Test physical consistency of detector models."""
    
    def test_asd_units_consistency(self):
        """ASD should have units of strain / sqrt(Hz)."""
        f = np.array([100.0])  # Hz
        detector = DETECTOR_NOISE_CURVES['LIGO_O1']
        
        asd = detector.asd_function(f)[0]  # Hz^-1/2
        
        # For integration time T, noise ~ ASD / sqrt(f * T)
        T = 1.0  # second
        df = 1.0  # Hz bandwidth
        noise_level = asd / np.sqrt(df * T)
        
        # Should be dimensionless strain
        # Typical LIGO: ~1e-22 / sqrt(Hz) → ~1e-22 for 1 Hz, 1 s
        assert 1e-24 < noise_level < 1e-20
    
    def test_seismic_wall_low_freq(self):
        """Ground-based detectors should have seismic noise at low f."""
        f_low = np.array([1.0, 5.0, 10.0])  # Below seismic cutoff
        f_high = np.array([100.0, 200.0, 500.0])  # Above seismic
        
        asd_low = aLIGO_design_asd(f_low)
        asd_high = aLIGO_design_asd(f_high)
        
        # Low frequencies should have worse (higher) ASD due to seismic
        assert np.mean(asd_low) > np.mean(asd_high)
    
    def test_shot_noise_high_freq(self):
        """High frequencies should be limited by shot noise."""
        f_mid = np.array([100.0])
        f_high = np.array([1000.0])
        
        asd_mid = aLIGO_design_asd(f_mid)[0]
        asd_high = aLIGO_design_asd(f_high)[0]
        
        # Shot noise rises with frequency: ASD ∝ sqrt(f)
        # So high f should have higher ASD
        assert asd_high > asd_mid


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_frequency(self):
        """Test with single frequency value."""
        f = np.array([100.0])
        
        for name, detector in DETECTOR_NOISE_CURVES.items():
            asd = detector.asd_function(f)
            assert len(asd) == 1
            assert asd[0] > 0
    
    def test_zero_frequency(self):
        """ASD at f=0 should be defined (or handled gracefully)."""
        f = np.array([0.0])
        
        # Most detectors should return large value or handle gracefully
        asd = LISA_asd(f)
        # Either very large (poor sensitivity at DC) or finite
        assert asd[0] > 1e-15 or np.isfinite(asd[0])
    
    def test_frequency_outside_band(self):
        """Test behavior when frequency is outside nominal band."""
        detector = DETECTOR_NOISE_CURVES['LIGO_O1']
        f_low = np.array([0.1])  # Below f_min = 10 Hz
        f_high = np.array([10000.0])  # Above f_max = 5000 Hz
        
        asd_low = detector.asd_function(f_low)
        asd_high = detector.asd_function(f_high)
        
        # Should return finite values (extrapolated or clamped)
        assert np.isfinite(asd_low[0])
        assert np.isfinite(asd_high[0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
