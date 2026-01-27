"""
Benchmark Testing Module for SSVEP-BCI
======================================

Provides utilities to download, load, and evaluate FB-CCA on public SSVEP datasets
for reproducible benchmarking and comparison with published results.

Supported Datasets:
- Tsinghua BETA Dataset (70 subjects, 40 frequencies)
- SSVEP Benchmark Dataset (35 subjects, 40 frequencies)

References:
-----------
[1] Liu, B., et al. (2020). BETA: A Large Benchmark Database Toward SSVEP-BCI Application.
    Frontiers in Neuroscience, 14, 627.
    
[2] Wang, Y., et al. (2016). A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces.
    IEEE Transactions on Neural Systems and Rehabilitation Engineering.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None

from cca_utils import cca_correlation, generate_reference_signals, design_filter_bank, compute_subband_weights
from scipy.signal import filtfilt


class BenchmarkDataset:
    """
    Base class for benchmark SSVEP datasets.
    
    Provides common interface for loading and processing different benchmark datasets.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize benchmark dataset.
        
        Parameters
        ----------
        data_path : str
            Path to the downloaded benchmark dataset folder.
        """
        self.data_path = Path(data_path)
        self.subjects = []
        self.frequencies = []
        self.sfreq = None
        self.n_channels = None
        self.data = {}
        
    def load_all_subjects(self) -> None:
        """Load data for all subjects."""
        raise NotImplementedError("Subclasses must implement load_all_subjects()")
    
    def get_subject_data(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data for a specific subject.
        
        Returns
        -------
        X : np.ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y : np.ndarray, shape (n_trials,)
            Labels (frequency indices)
        """
        raise NotImplementedError("Subclasses must implement get_subject_data()")


class BETADataset(BenchmarkDataset):
    """
    Tsinghua BETA Benchmark Dataset.
    
    Dataset characteristics:
    - 70 subjects
    - 40 target frequencies (8-15.8 Hz in 0.2 Hz steps)
    - 64 EEG channels (9 occipital selected for SSVEP)
    - 250 Hz sampling rate
    - 4 blocks, each with 40 trials
    
    Download: http://bci.med.tsinghua.edu.cn/download.html
    
    The data should be organized as:
    data_path/
        S1.mat
        S2.mat
        ...
        S70.mat
    """
    
    # BETA dataset frequencies (40 targets)
    FREQUENCIES = [8.0 + i * 0.2 for i in range(40)]  # 8.0 to 15.8 Hz
    
    # Occipital channels typically used for SSVEP (0-indexed)
    OCCIPITAL_CHANNELS = [47, 53, 54, 55, 56, 57, 60, 61, 62]  # Pz, PO5-8, POz, O1, Oz, O2
    
    def __init__(self, data_path: str, channels: Optional[List[int]] = None):
        """
        Initialize BETA dataset loader.
        
        Parameters
        ----------
        data_path : str
            Path to folder containing S1.mat, S2.mat, etc.
        channels : List[int], optional
            Channel indices to use. Default: occipital channels.
        """
        super().__init__(data_path)
        
        if loadmat is None:
            raise ImportError("scipy.io.loadmat required for BETA dataset. Install scipy.")
        
        self.frequencies = self.FREQUENCIES
        self.sfreq = 250
        self.channels = channels if channels is not None else self.OCCIPITAL_CHANNELS
        self.n_channels = len(self.channels)
        
        # Find available subjects
        self._find_subjects()
    
    def _find_subjects(self) -> None:
        """Find all subject files in data path."""
        self.subjects = []
        for mat_file in sorted(self.data_path.glob("S*.mat")):
            subj_id = mat_file.stem  # e.g., "S1"
            self.subjects.append(subj_id)
        
        if not self.subjects:
            warnings.warn(f"No subject files found in {self.data_path}. "
                         f"Expected files like S1.mat, S2.mat, ...")
    
    def load_subject(self, subject_id: str) -> Dict[str, np.ndarray]:
        """
        Load data for a single subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID (e.g., "S1")
            
        Returns
        -------
        data : dict
            Dictionary with 'X' (EEG data) and 'y' (labels)
        """
        mat_path = self.data_path / f"{subject_id}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Subject file not found: {mat_path}")
        
        mat_data = loadmat(str(mat_path))
        
        # BETA format: data shape is (n_channels, n_samples, n_targets, n_blocks)
        # or similar depending on exact version
        raw_data = mat_data.get('data', mat_data.get('eeg', None))
        
        if raw_data is None:
            raise KeyError(f"Cannot find 'data' or 'eeg' key in {mat_path}")
        
        # Reshape to (n_trials, n_channels, n_samples)
        # Typical BETA shape: (64, 1500, 40, 4) -> 64 ch, 6s@250Hz, 40 freq, 4 blocks
        if raw_data.ndim == 4:
            n_ch, n_samp, n_freq, n_blocks = raw_data.shape
            
            # Select channels
            raw_data = raw_data[self.channels, :, :, :]
            
            # Reshape: (n_ch, n_samp, n_freq, n_blocks) -> (n_trials, n_ch, n_samp)
            # n_trials = n_freq * n_blocks
            X = raw_data.transpose(2, 3, 0, 1)  # (n_freq, n_blocks, n_ch, n_samp)
            X = X.reshape(-1, len(self.channels), n_samp)  # (n_trials, n_ch, n_samp)
            
            # Create labels (frequency index for each trial)
            y = np.repeat(np.arange(n_freq), n_blocks)
            
        else:
            raise ValueError(f"Unexpected data shape: {raw_data.shape}")
        
        return {'X': X, 'y': y}
    
    def load_all_subjects(self) -> None:
        """Load data for all subjects."""
        print(f"Loading BETA dataset from {self.data_path}...")
        
        for subj in self.subjects:
            try:
                self.data[subj] = self.load_subject(subj)
                print(f"  Loaded {subj}: {self.data[subj]['X'].shape[0]} trials")
            except Exception as e:
                warnings.warn(f"Failed to load {subj}: {e}")
        
        print(f"Loaded {len(self.data)}/{len(self.subjects)} subjects")
    
    def get_subject_data(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get data for a specific subject."""
        if subject_id not in self.data:
            self.data[subject_id] = self.load_subject(subject_id)
        
        d = self.data[subject_id]
        return d['X'], d['y']


class FBCCABenchmark:
    """
    FB-CCA benchmarking class.
    
    Runs FB-CCA (and standard CCA) on benchmark datasets and reports
    accuracy comparable to published results.
    """
    
    def __init__(
        self,
        num_harmonics: int = 5,
        num_subbands: int = 5,
        fb_fundamental_freq: float = 6.0,
        fb_bandwidth: float = 8.0,
        window_length: float = 1.0,
        fb_weight_a: float = 1.25,
        fb_weight_b: float = 0.25
    ):
        """
        Initialize FB-CCA benchmark.
        
        Parameters
        ----------
        num_harmonics : int
            Number of harmonics for reference signals (default: 5)
        num_subbands : int
            Number of filter bank sub-bands (default: 5)
        fb_fundamental_freq : float
            Starting frequency for filter bank (default: 6.0 Hz)
        fb_bandwidth : float
            Bandwidth of each sub-band (default: 8.0 Hz)
        window_length : float
            Analysis window length in seconds (default: 1.0s)
        fb_weight_a, fb_weight_b : float
            FB-CCA weighting parameters
        """
        self.num_harmonics = num_harmonics
        self.num_subbands = num_subbands
        self.fb_fundamental_freq = fb_fundamental_freq
        self.fb_bandwidth = fb_bandwidth
        self.window_length = window_length
        self.fb_weight_a = fb_weight_a
        self.fb_weight_b = fb_weight_b
        
        self.results = {}
    
    def run_benchmark(
        self,
        dataset: BenchmarkDataset,
        methods: List[str] = ['CCA', 'FBCCA'],
        window_lengths: Optional[List[float]] = None,
        n_subjects: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark evaluation on dataset.
        
        Parameters
        ----------
        dataset : BenchmarkDataset
            Loaded benchmark dataset
        methods : List[str]
            Methods to evaluate ('CCA', 'FBCCA')
        window_lengths : List[float], optional
            Window lengths to test. Default: [0.5, 1.0, 2.0, 3.0, 4.0]
        n_subjects : int, optional
            Number of subjects to evaluate (for quick testing). Default: all.
            
        Returns
        -------
        results : dict
            Dictionary with accuracy results for each method and window length
        """
        if window_lengths is None:
            window_lengths = [0.5, 1.0, 2.0, 3.0, 4.0]
        
        subjects = dataset.subjects[:n_subjects] if n_subjects else dataset.subjects
        
        print(f"\n{'='*60}")
        print(f"FB-CCA Benchmark Evaluation")
        print(f"{'='*60}")
        print(f"Dataset: {dataset.data_path}")
        print(f"Subjects: {len(subjects)}")
        print(f"Frequencies: {len(dataset.frequencies)}")
        print(f"Window lengths: {window_lengths}")
        print(f"Methods: {methods}")
        print(f"{'='*60}\n")
        
        results = {
            'methods': methods,
            'window_lengths': window_lengths,
            'per_subject': {},
            'mean_accuracy': {},
            'std_accuracy': {}
        }
        
        for method in methods:
            results['mean_accuracy'][method] = {}
            results['std_accuracy'][method] = {}
            
            for wl in window_lengths:
                results['mean_accuracy'][method][wl] = None
                results['std_accuracy'][method][wl] = None
        
        # Evaluate each subject
        for subj in subjects:
            print(f"Processing {subj}...")
            
            try:
                X, y = dataset.get_subject_data(subj)
            except Exception as e:
                print(f"  Skipping {subj}: {e}")
                continue
            
            results['per_subject'][subj] = {}
            
            for method in methods:
                results['per_subject'][subj][method] = {}
                
                for wl in window_lengths:
                    acc = self._evaluate_subject(
                        X, y, 
                        frequencies=dataset.frequencies,
                        sfreq=dataset.sfreq,
                        method=method,
                        window_length=wl
                    )
                    results['per_subject'][subj][method][wl] = acc
                    print(f"  {method} @ {wl}s: {acc*100:.1f}%")
        
        # Compute mean and std across subjects
        for method in methods:
            for wl in window_lengths:
                accs = [results['per_subject'][s][method][wl] 
                        for s in results['per_subject'] 
                        if method in results['per_subject'][s]]
                
                if accs:
                    results['mean_accuracy'][method][wl] = np.mean(accs)
                    results['std_accuracy'][method][wl] = np.std(accs)
        
        # Print summary
        self._print_summary(results)
        
        self.results = results
        return results
    
    def _evaluate_subject(
        self,
        X: np.ndarray,
        y: np.ndarray,
        frequencies: List[float],
        sfreq: float,
        method: str,
        window_length: float
    ) -> float:
        """
        Evaluate single subject with specified method.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_channels, n_samples)
            EEG data
        y : np.ndarray, shape (n_trials,)
            Labels (frequency indices)
        frequencies : List[float]
            Stimulus frequencies
        sfreq : float
            Sampling frequency
        method : str
            'CCA' or 'FBCCA'
        window_length : float
            Analysis window in seconds
            
        Returns
        -------
        accuracy : float
            Classification accuracy
        """
        n_trials = X.shape[0]
        n_samples = int(window_length * sfreq)
        
        # Ensure we don't exceed available samples
        n_samples = min(n_samples, X.shape[2])
        
        # Design filter bank if needed
        if method == 'FBCCA':
            filter_bank = design_filter_bank(
                n_subbands=self.num_subbands,
                fundamental_freq=self.fb_fundamental_freq,
                bandwidth=self.fb_bandwidth,
                sfreq=sfreq
            )
            weights = compute_subband_weights(
                self.num_subbands, 
                a=self.fb_weight_a, 
                b=self.fb_weight_b
            )
        
        predictions = []
        
        for trial_idx in range(n_trials):
            trial_data = X[trial_idx, :, :n_samples].T  # (n_samples, n_channels)
            
            if method == 'CCA':
                corrs = self._cca_all_frequencies(trial_data, frequencies, sfreq)
            else:  # FBCCA
                corrs = self._fbcca_all_frequencies(
                    trial_data, frequencies, sfreq, filter_bank, weights
                )
            
            predictions.append(np.argmax(corrs))
        
        predictions = np.array(predictions)
        accuracy = np.mean(predictions == y)
        
        return accuracy
    
    def _cca_all_frequencies(
        self,
        data: np.ndarray,
        frequencies: List[float],
        sfreq: float
    ) -> np.ndarray:
        """Compute CCA correlation for all target frequencies."""
        n_samples = data.shape[0]
        corrs = np.zeros(len(frequencies))
        
        for i, freq in enumerate(frequencies):
            ref = generate_reference_signals(
                n_samples, frequency=freq, sfreq=sfreq, n_harmonics=self.num_harmonics
            )
            corrs[i] = cca_correlation(data, ref)
        
        return corrs
    
    def _fbcca_all_frequencies(
        self,
        data: np.ndarray,
        frequencies: List[float],
        sfreq: float,
        filter_bank: List,
        weights: np.ndarray
    ) -> np.ndarray:
        """Compute FB-CCA features for all target frequencies."""
        n_samples = data.shape[0]
        corrs = np.zeros(len(frequencies))
        
        for i, freq in enumerate(frequencies):
            ref = generate_reference_signals(
                n_samples, frequency=freq, sfreq=sfreq, n_harmonics=self.num_harmonics
            )
            
            # Weighted sum across sub-bands
            weighted_corr = 0
            for sb_idx, (b, a, low, high) in enumerate(filter_bank):
                # Apply filter to data
                filtered = filtfilt(b, a, data, axis=0)
                
                # Compute CCA
                sb_corr = cca_correlation(filtered, ref)
                weighted_corr += weights[sb_idx] * (sb_corr ** 2)
            
            corrs[i] = weighted_corr
        
        return corrs
    
    def _print_summary(self, results: Dict) -> None:
        """Print results summary table."""
        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Method':<10} | " + " | ".join([f"{wl}s" for wl in results['window_lengths']]))
        print("-" * 60)
        
        for method in results['methods']:
            accs = [results['mean_accuracy'][method].get(wl, 0) for wl in results['window_lengths']]
            stds = [results['std_accuracy'][method].get(wl, 0) for wl in results['window_lengths']]
            
            row = f"{method:<10} | "
            row += " | ".join([f"{a*100:.1f}±{s*100:.1f}" if a else "N/A" 
                             for a, s in zip(accs, stds)])
            print(row)
        
        print(f"{'='*60}\n")


def download_beta_dataset(target_path: str) -> str:
    """
    Print instructions for downloading the BETA dataset.
    
    The BETA dataset requires manual download due to size and licensing.
    
    Parameters
    ----------
    target_path : str
        Where to store the downloaded data
        
    Returns
    -------
    instructions : str
        Download instructions
    """
    instructions = f"""
    ============================================================
    BETA Dataset Download Instructions
    ============================================================
    
    The BETA dataset is a large benchmark for SSVEP-BCI evaluation.
    
    1. Visit: http://bci.med.tsinghua.edu.cn/download.html
    
    2. Request access to the BETA dataset
    
    3. Download and extract to: {target_path}
    
    4. The folder should contain:
       {target_path}/
           S1.mat
           S2.mat
           ...
           S70.mat
    
    5. Then run:
       >>> from benchmark import BETADataset, FBCCABenchmark
       >>> dataset = BETADataset("{target_path}")
       >>> dataset.load_all_subjects()
       >>> benchmark = FBCCABenchmark()
       >>> results = benchmark.run_benchmark(dataset)
    
    ============================================================
    """
    print(instructions)
    return instructions


# Convenience function for quick benchmarking
def run_comparison(data_path: str, n_subjects: int = 5) -> Dict:
    """
    Run a quick CCA vs FB-CCA comparison on benchmark data.
    
    Parameters
    ----------
    data_path : str
        Path to BETA dataset folder
    n_subjects : int
        Number of subjects to test (for quick evaluation)
        
    Returns
    -------
    results : dict
        Comparison results
    """
    dataset = BETADataset(data_path)
    dataset.load_all_subjects()
    
    benchmark = FBCCABenchmark(
        num_harmonics=5,
        num_subbands=5,
        fb_fundamental_freq=6.0,
        fb_bandwidth=8.0
    )
    
    results = benchmark.run_benchmark(
        dataset,
        methods=['CCA', 'FBCCA'],
        window_lengths=[0.5, 1.0, 2.0],
        n_subjects=n_subjects
    )
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        n_subj = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        run_comparison(data_path, n_subjects=n_subj)
    else:
        download_beta_dataset("./BETA_data")
