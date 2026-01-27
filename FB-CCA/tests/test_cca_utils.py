"""
Unit tests for CCA utilities.

Tests cover:
- cca_correlation: SVD-based canonical correlation computation
- generate_reference_signals: Sinusoidal reference signal generation
- design_filter_bank: Butterworth filter bank design
- compute_subband_weights: FB-CCA weighting schemes
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cca_utils import (
    cca_correlation,
    generate_reference_signals,
    design_filter_bank,
    compute_subband_weights
)


class TestCCACorrelation:
    """Tests for the SVD-based CCA correlation function."""
    
    def test_identical_signals_perfect_correlation(self):
        """Identical signals should have correlation = 1."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        Y = X.copy()
        
        corr = cca_correlation(X, Y)
        assert np.isclose(corr, 1.0, atol=1e-6), f"Expected ~1.0, got {corr}"
    
    def test_orthogonal_signals_zero_correlation(self):
        """Orthogonal signals should have correlation near 0."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create orthogonal signals
        X = np.column_stack([
            np.sin(2 * np.pi * 10 * np.arange(n_samples) / 256),
            np.cos(2 * np.pi * 10 * np.arange(n_samples) / 256)
        ])
        Y = np.column_stack([
            np.sin(2 * np.pi * 25 * np.arange(n_samples) / 256),
            np.cos(2 * np.pi * 25 * np.arange(n_samples) / 256)
        ])
        
        corr = cca_correlation(X, Y)
        assert corr < 0.2, f"Expected low correlation for orthogonal signals, got {corr}"
    
    def test_known_correlation(self):
        """Test with signals of known correlation structure."""
        np.random.seed(42)
        n_samples = 1000
        t = np.arange(n_samples) / 256
        
        # Create sinusoidal signals at 10 Hz
        signal = np.sin(2 * np.pi * 10 * t)
        
        X = np.column_stack([signal, signal * 0.8, signal * 0.6])
        Y = np.column_stack([np.sin(2 * np.pi * 10 * t), np.cos(2 * np.pi * 10 * t)])
        
        corr = cca_correlation(X, Y)
        assert 0.9 < corr <= 1.0, f"Expected high correlation, got {corr}"
    
    def test_correlation_bounds(self):
        """Correlation should always be between 0 and 1."""
        np.random.seed(42)
        
        for _ in range(10):
            X = np.random.randn(100, np.random.randint(1, 5))
            Y = np.random.randn(100, np.random.randint(1, 5))
            
            corr = cca_correlation(X, Y)
            assert 0 <= corr <= 1, f"Correlation out of bounds: {corr}"
    
    def test_shape_mismatch_samples(self):
        """Different number of samples should raise error."""
        X = np.random.randn(100, 3)
        Y = np.random.randn(50, 3)  # Different n_samples
        
        with pytest.raises((ValueError, IndexError)):
            cca_correlation(X, Y)
    
    def test_single_feature(self):
        """Should work with single feature in each dataset."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        Y = X * 2 + np.random.randn(100, 1) * 0.1  # Linear relationship
        
        corr = cca_correlation(X, Y)
        assert corr > 0.9, f"Expected high correlation for linear relationship, got {corr}"


class TestReferenceSignals:
    """Tests for reference signal generation."""
    
    def test_output_shape(self):
        """Check output dimensions."""
        n_samples = 1000
        n_harmonics = 3
        
        ref = generate_reference_signals(n_samples, frequency=10.0, sfreq=256, n_harmonics=n_harmonics)
        
        assert ref.shape == (n_samples, 2 * n_harmonics), \
            f"Expected shape ({n_samples}, {2 * n_harmonics}), got {ref.shape}"
    
    def test_fundamental_frequency(self):
        """Check that fundamental frequency is correct."""
        sfreq = 256
        freq = 10.0
        n_samples = 1024
        
        ref = generate_reference_signals(n_samples, frequency=freq, sfreq=sfreq, n_harmonics=1)
        
        # Compute power spectrum of sine component
        spectrum = np.abs(np.fft.fft(ref[:, 0]))
        freqs = np.fft.fftfreq(n_samples, 1/sfreq)
        
        # Peak should be at fundamental frequency
        peak_freq = freqs[np.argmax(spectrum[:n_samples//2])]
        assert np.isclose(peak_freq, freq, atol=0.5), \
            f"Expected peak at {freq} Hz, got {peak_freq} Hz"
    
    def test_harmonics(self):
        """Check that harmonics are generated correctly."""
        sfreq = 256
        freq = 8.0
        n_samples = 1024
        n_harmonics = 3
        
        ref = generate_reference_signals(n_samples, frequency=freq, sfreq=sfreq, n_harmonics=n_harmonics)
        
        # Check each harmonic
        for h in range(1, n_harmonics + 1):
            sin_idx = 2 * (h - 1)
            spectrum = np.abs(np.fft.fft(ref[:, sin_idx]))
            freqs = np.fft.fftfreq(n_samples, 1/sfreq)
            
            peak_freq = freqs[np.argmax(spectrum[:n_samples//2])]
            expected_freq = freq * h
            
            assert np.isclose(peak_freq, expected_freq, atol=0.5), \
                f"Harmonic {h}: expected {expected_freq} Hz, got {peak_freq} Hz"
    
    def test_sin_cos_orthogonality(self):
        """Sin and cos pairs should be orthogonal."""
        ref = generate_reference_signals(1000, frequency=10.0, sfreq=256, n_harmonics=2)
        
        # Check sin/cos pairs are orthogonal (dot product ≈ 0)
        for h in range(2):
            sin_signal = ref[:, 2*h]
            cos_signal = ref[:, 2*h + 1]
            
            dot = np.dot(sin_signal, cos_signal) / len(sin_signal)
            assert np.abs(dot) < 0.05, f"Harmonic {h+1}: sin/cos not orthogonal, dot={dot}"
    
    def test_amplitude_normalized(self):
        """Signals should have amplitude in [-1, 1]."""
        ref = generate_reference_signals(1000, frequency=10.0, sfreq=256, n_harmonics=3)
        
        assert np.all(ref >= -1.0) and np.all(ref <= 1.0), "Signals not normalized"


class TestFilterBank:
    """Tests for filter bank design."""
    
    def test_output_structure(self):
        """Check filter bank returns correct structure."""
        fb = design_filter_bank(n_subbands=5, fundamental_freq=6.0, bandwidth=8.0, sfreq=256)
        
        assert len(fb) == 5, f"Expected 5 sub-bands, got {len(fb)}"
        
        for i, (b, a, low, high) in enumerate(fb):
            assert len(b) > 0, f"Sub-band {i}: empty b coefficients"
            assert len(a) > 0, f"Sub-band {i}: empty a coefficients"
            assert low < high, f"Sub-band {i}: low >= high ({low} >= {high})"
    
    def test_frequency_ranges(self):
        """Check sub-band frequency ranges are correct."""
        fundamental = 6.0
        bandwidth = 8.0
        n_subbands = 4
        
        fb = design_filter_bank(n_subbands=n_subbands, fundamental_freq=fundamental, 
                               bandwidth=bandwidth, sfreq=256)
        
        for i, (b, a, low, high) in enumerate(fb):
            expected_low = fundamental + i * bandwidth
            expected_high = expected_low + bandwidth
            
            assert np.isclose(low, expected_low), \
                f"Sub-band {i}: expected low={expected_low}, got {low}"
            assert high <= expected_high + 1, \
                f"Sub-band {i}: high={high} exceeds expected {expected_high}"
    
    def test_nyquist_limit(self):
        """High frequency should not exceed Nyquist, sub-bands beyond Nyquist are skipped."""
        sfreq = 256
        nyquist = sfreq / 2
        
        # This should skip sub-bands that exceed Nyquist instead of raising an error
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fb = design_filter_bank(n_subbands=10, fundamental_freq=6.0, bandwidth=20.0, sfreq=sfreq)
        
        # Some sub-bands should be skipped (those exceeding Nyquist)
        assert len(fb) < 10, "Expected some sub-bands to be skipped due to Nyquist limit"
        
        # All remaining sub-bands should be valid
        for i, (b, a, low, high) in enumerate(fb):
            assert high < nyquist, f"Sub-band {i}: high freq {high} >= Nyquist {nyquist}"
            assert low < high, f"Sub-band {i}: invalid range {low} >= {high}"
    
    def test_filter_stability(self):
        """Filters should be stable (poles inside unit circle)."""
        fb = design_filter_bank(n_subbands=5, fundamental_freq=6.0, bandwidth=8.0, sfreq=256)
        
        for i, (b, a, low, high) in enumerate(fb):
            poles = np.roots(a)
            assert np.all(np.abs(poles) < 1), f"Sub-band {i}: unstable filter"
    
    def test_filter_application(self):
        """Filters should not produce NaN or Inf values."""
        from scipy.signal import filtfilt
        
        fb = design_filter_bank(n_subbands=3, fundamental_freq=8.0, bandwidth=10.0, sfreq=256)
        
        # Create test signal
        signal = np.random.randn(1024)
        
        for i, (b, a, low, high) in enumerate(fb):
            filtered = filtfilt(b, a, signal)
            assert not np.any(np.isnan(filtered)), f"Sub-band {i}: NaN in output"
            assert not np.any(np.isinf(filtered)), f"Sub-band {i}: Inf in output"


class TestSubbandWeights:
    """Tests for FB-CCA sub-band weights computation."""
    
    def test_standard_weights_shape(self):
        """Check output shape."""
        weights = compute_subband_weights(n_subbands=5)
        assert weights.shape == (5,), f"Expected shape (5,), got {weights.shape}"
    
    def test_standard_weights_decreasing(self):
        """Standard weights should decrease with sub-band index."""
        weights = compute_subband_weights(n_subbands=5, method='standard')
        
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i+1], \
                f"Weights not decreasing: w[{i}]={weights[i]} < w[{i+1}]={weights[i+1]}"
    
    def test_all_weights_positive(self):
        """All weights should be positive."""
        for method in ['standard', 'exponential', 'linear', 'gaussian', 'uniform']:
            weights = compute_subband_weights(n_subbands=5, method=method)
            assert np.all(weights > 0), f"Method {method}: non-positive weights"
    
    def test_uniform_weights(self):
        """Uniform weights should all be equal."""
        weights = compute_subband_weights(n_subbands=5, method='uniform')
        assert np.allclose(weights, 1.0), "Uniform weights should all be 1.0"
    
    def test_exponential_decay(self):
        """Exponential weights should decay exponentially."""
        weights = compute_subband_weights(n_subbands=5, a=0.5, b=0, method='exponential')
        
        # Check exponential decay pattern
        for i in range(1, len(weights)):
            ratio = weights[i] / weights[i-1]
            expected_ratio = np.exp(-0.5)
            assert np.isclose(ratio, expected_ratio, rtol=0.1), \
                f"Not exponential decay: ratio={ratio}, expected={expected_ratio}"
    
    def test_invalid_method(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            compute_subband_weights(n_subbands=5, method='invalid_method')
    
    def test_chen_paper_defaults(self):
        """Test default parameters match Chen et al. (2015) paper."""
        weights = compute_subband_weights(n_subbands=5, a=1.25, b=0.25, method='standard')
        
        # First weight should be n^(-1.25) + 0.25 = 1^(-1.25) + 0.25 = 1.25
        assert np.isclose(weights[0], 1.25, rtol=0.01), f"First weight: {weights[0]}"


class TestCCAWithReferenceSignals:
    """Integration tests: CCA with generated reference signals."""
    
    def test_ssvep_simulation(self):
        """Simulate SSVEP and verify CCA detects correct frequency."""
        np.random.seed(42)
        sfreq = 256
        n_samples = 1024
        stim_freq = 12.0  # Target frequency
        test_freqs = [8.0, 10.0, 12.0, 15.0, 20.0]
        
        t = np.arange(n_samples) / sfreq
        
        # Simulate SSVEP signal (12 Hz + harmonics + noise)
        ssvep = (np.sin(2 * np.pi * stim_freq * t) + 
                 0.5 * np.sin(2 * np.pi * stim_freq * 2 * t) +
                 0.3 * np.random.randn(n_samples))
        
        # Multi-channel EEG (3 channels with same SSVEP)
        eeg = np.column_stack([ssvep, ssvep * 0.9, ssvep * 0.8])
        
        # Test CCA with each frequency
        correlations = {}
        for freq in test_freqs:
            ref = generate_reference_signals(n_samples, frequency=freq, sfreq=sfreq, n_harmonics=2)
            correlations[freq] = cca_correlation(eeg, ref)
        
        # The target frequency should have highest correlation
        detected_freq = max(correlations, key=correlations.get)
        assert detected_freq == stim_freq, \
            f"Expected {stim_freq} Hz, detected {detected_freq} Hz. Correlations: {correlations}"
    
    def test_multiple_channels(self):
        """Test with varying number of EEG channels."""
        np.random.seed(42)
        sfreq = 256
        n_samples = 512
        freq = 10.0
        
        t = np.arange(n_samples) / sfreq
        base_signal = np.sin(2 * np.pi * freq * t)
        
        ref = generate_reference_signals(n_samples, frequency=freq, sfreq=sfreq, n_harmonics=1)
        
        for n_channels in [1, 3, 5, 10]:
            eeg = np.column_stack([base_signal * (0.5 + 0.5 * np.random.rand()) 
                                   for _ in range(n_channels)])
            corr = cca_correlation(eeg, ref)
            assert corr > 0.7, f"{n_channels} channels: correlation too low ({corr})"


class TestArtifactRejection:
    """Tests for automatic artifact rejection (preprocessing module)."""
    
    def test_import_artifact_rejector(self):
        """Test that ArtifactRejector can be imported (requires MNE)."""
        try:
            from preprocessing import ArtifactRejector
            rejector = ArtifactRejector(verbose=False)
            assert rejector is not None
        except ImportError as e:
            # Skip if MNE is not installed
            pytest.skip(f"MNE not installed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
