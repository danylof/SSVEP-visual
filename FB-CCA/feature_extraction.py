import numpy as np
from sklearn.cross_decomposition import CCA


class FeatureExtractor:
    """Extracts features from EEG epochs using Canonical Correlation Analysis (CCA) or similar methods."""

    def __init__(self, method="CCA", sfreq=256, num_harmonics=1):
        """
        Initialize the feature extractor.

        Parameters:
        - method (str): Method used for feature extraction (default: "CCA"). Currently supports only "CCA".
        - sfreq (int): Sampling frequency of EEG data in Hz (default: 256).
        - num_harmonics (int): Number of harmonic components to include in reference signals (default: 1).
        """
        self.method = method
        self.sfreq = sfreq
        self.num_harmonics = num_harmonics

    def extract_features(self, epochs, stim_frequencies):
        """
        Extract features from EEG epochs by computing correlations with frequency-specific reference signals.

        Parameters:
        - epochs (mne.Epochs): EEG epochs from which to extract features.
        - stim_frequencies (List[float]): List of stimulation frequencies for reference signals.

        Returns:
        - cca_features (np.ndarray): Array containing CCA correlation features, shape (num_trials, num_frequencies).
        """
        data = epochs.get_data()  # EEG data shape: (num_trials, num_channels, num_samples)
        num_trials, num_channels, num_samples = data.shape
        num_freqs = len(stim_frequencies)

        cca_features = np.zeros((num_trials, num_freqs))  # Placeholder for feature values

        for trial_idx in range(num_trials):
            eeg_trial = data[trial_idx].T  # Reshape to (num_samples, num_channels)

            for freq_idx, freq in enumerate(stim_frequencies):
                # Generate reference signals for current frequency
                ref_signals = self._get_reference_signals(eeg_trial.shape[0], freq)

                # Compute correlation between EEG trial data and reference signals using CCA
                cca_corr = self._compute_corr(eeg_trial, ref_signals)

                # Store computed correlation
                cca_features[trial_idx, freq_idx] = cca_corr

        return cca_features

    def _get_reference_signals(self, length, freq):
        """
        Generate sinusoidal reference signals (sine and cosine) for a given frequency and its harmonics.

        Parameters:
        - length (int): Number of time samples in the EEG trial.
        - freq (float): Frequency of stimulation to generate reference signals.

        Returns:
        - ref_signals (np.ndarray): Array of reference signals, shape (num_reference_signals, num_samples).
        """
        t = np.arange(length) / self.sfreq
        ref_signals = []

        # Generate sine and cosine signals for the fundamental frequency and specified harmonics
        for harm in range(1, self.num_harmonics + 1):
            ref_signals.append(np.sin(2 * np.pi * freq * harm * t))
            ref_signals.append(np.cos(2 * np.pi * freq * harm * t))

        return np.array(ref_signals)

    def _compute_corr(self, eeg_data, ref_signals):
        """
        Compute canonical correlation between EEG trial data and generated reference signals.

        Parameters:
        - eeg_data (np.ndarray): EEG data for a single trial, shape (num_samples, num_channels).
        - ref_signals (np.ndarray): Reference signals, shape (num_reference_signals, num_samples).

        Returns:
        - corr (float): Canonical correlation coefficient between EEG data and reference signals.
        """
        cca = CCA(n_components=1)
        cca.fit(eeg_data, ref_signals.T)
        # Transform both datasets (EEG data and reference signals) to their canonical variables
        # U corresponds to EEG data canonical variables, V corresponds to reference signals canonical variables
        U, V = cca.transform(eeg_data, ref_signals.T)

        # Compute the Pearson correlation between the first pair of canonical variables from U and V
        # This correlation reflects the strength of the linear relationship between EEG and reference signals
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]
        return corr
