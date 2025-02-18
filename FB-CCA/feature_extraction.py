import numpy as np
from scipy.signal import stft

class FeatureExtractor:
    """Extracts spectral and temporal features from EEG data."""
    
    def __init__(self, method="STFT", filterbank=False, num_harmonics=3):
        self.method = method
        self.filterbank = filterbank
        self.num_harmonics = num_harmonics

    def extract_features(self, epochs):
        """Extract features using STFT or CCA."""
        if self.method == "STFT":
            return self._compute_stft(epochs)
        elif self.method == "CCA":
            return self._compute_cca(epochs)
        else:
            raise ValueError("Invalid feature extraction method")

    def _compute_stft(self, epochs):
        """Compute the STFT features of EEG signals."""
        data = epochs.get_data()
        freqs, _, Zxx = stft(data, fs=epochs.info['sfreq'], nperseg=256)
        return np.abs(Zxx)

    def _compute_cca(self, epochs):
        """Compute CCA features for SSVEP signals."""
        data = epochs.get_data()
        num_trials, num_channels, num_samples = data.shape
        cca_features = np.zeros((num_trials, self.num_harmonics))
        
        for i in range(num_trials):
            # Simulated CCA feature computation (actual implementation needed)
            cca_features[i, :] = np.random.rand(self.num_harmonics)
        
        return cca_features
