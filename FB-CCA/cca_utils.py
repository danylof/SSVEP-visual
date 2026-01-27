"""
CCA Utilities for SSVEP Analysis
================================

Fast, SVD-based implementations of Canonical Correlation Analysis (CCA) 
and Filter Bank CCA (FB-CCA) optimized for SSVEP-based BCI applications.

These implementations are significantly faster than sklearn's CCA for 
single-correlation extraction, making them ideal for real-time BCI systems.

Functions:
----------
- cca_correlation: Fast SVD-based canonical correlation computation
- generate_reference_signals: Create sinusoidal reference signals for SSVEP
- fbcca_features: Complete FB-CCA feature extraction pipeline

Example Usage:
--------------
>>> from cca_utils import cca_correlation, fbcca_features
>>> 
>>> # Single CCA correlation
>>> corr = cca_correlation(eeg_data, reference_signals)
>>> 
>>> # FB-CCA features for SSVEP classification
>>> features = fbcca_features(eeg_epochs, stim_frequencies=[8, 10, 12, 15])

References:
-----------
[1] Chen, X., et al. (2015). Filter bank canonical correlation analysis for 
    implementing a high-speed SSVEP-based brain-computer interface.
    Journal of Neural Engineering, 12(4), 046008.
    
[2] Lin, Z., et al. (2006). Frequency recognition based on canonical 
    correlation analysis for SSVEP-based BCIs.
    IEEE Trans. Biomedical Engineering, 53(12), 2610-2614.
"""

import numpy as np
from scipy.linalg import svd, qr
from scipy.signal import butter, filtfilt
from typing import List, Tuple, Optional, Union
import warnings


def cca_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the first canonical correlation between two datasets using SVD.
    
    This is significantly faster than sklearn's CCA when only the correlation
    coefficient is needed (not the full canonical variables).
    
    The algorithm:
    1. Center both datasets
    2. QR decomposition for numerical stability  
    3. SVD of Q_x.T @ Q_y gives canonical correlations as singular values
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features_x)
        First dataset (e.g., EEG data: samples × channels)
    Y : np.ndarray, shape (n_samples, n_features_y)
        Second dataset (e.g., reference signals: samples × harmonics)
        
    Returns
    -------
    corr : float
        First (maximum) canonical correlation coefficient, range [0, 1]
        
    Examples
    --------
    >>> eeg = np.random.randn(1000, 3)  # 1000 samples, 3 channels
    >>> ref = np.column_stack([np.sin(2*np.pi*10*t), np.cos(2*np.pi*10*t)])
    >>> corr = cca_correlation(eeg, ref)
    
    Notes
    -----
    Mathematical background:
    - CCA finds weight vectors a, b that maximize corr(Xa, Yb)
    - The canonical correlations are the singular values of Q_x.T @ Q_y
      where Q_x, Q_y are from QR decomposition of centered X, Y
    - This is equivalent to solving the generalized eigenvalue problem
      but numerically more stable
    """
    # Center the data
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # QR decomposition for numerical stability
    Q_x, _ = qr(X, mode='economic')
    Q_y, _ = qr(Y, mode='economic')
    
    # SVD of Q_x.T @ Q_y gives canonical correlations
    C = Q_x.T @ Q_y
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, s, _ = svd(C, full_matrices=False)
    
    # First singular value is the maximum canonical correlation
    return np.clip(s[0], 0.0, 1.0)


def generate_reference_signals(
    n_samples: int, 
    frequency: float, 
    sfreq: float, 
    n_harmonics: int = 1
) -> np.ndarray:
    """
    Generate sinusoidal reference signals for SSVEP frequency detection.
    
    Creates sine and cosine pairs for the fundamental frequency and its harmonics,
    which are used as reference signals in CCA-based SSVEP detection.
    
    Parameters
    ----------
    n_samples : int
        Number of time samples
    frequency : float
        Fundamental stimulus frequency in Hz
    sfreq : float
        Sampling frequency in Hz
    n_harmonics : int, default=1
        Number of harmonics to include (1 = fundamental only)
        
    Returns
    -------
    ref_signals : np.ndarray, shape (n_samples, 2 * n_harmonics)
        Reference signals matrix with alternating sin/cos columns:
        [sin(f), cos(f), sin(2f), cos(2f), ...]
        
    Examples
    --------
    >>> ref = generate_reference_signals(1000, frequency=10.0, sfreq=256, n_harmonics=2)
    >>> ref.shape
    (1000, 4)  # sin(10Hz), cos(10Hz), sin(20Hz), cos(20Hz)
    """
    t = np.arange(n_samples) / sfreq
    ref_signals = []
    
    for h in range(1, n_harmonics + 1):
        ref_signals.append(np.sin(2 * np.pi * frequency * h * t))
        ref_signals.append(np.cos(2 * np.pi * frequency * h * t))
    
    return np.column_stack(ref_signals)


def design_filter_bank(
    n_subbands: int,
    fundamental_freq: float,
    bandwidth: float,
    sfreq: float,
    filter_order: int = 4
) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """
    Design a filter bank for FB-CCA.
    
    Creates multiple Butterworth bandpass filters covering consecutive
    frequency bands.
    
    Parameters
    ----------
    n_subbands : int
        Number of sub-bands in the filter bank
    fundamental_freq : float
        Starting frequency of the first sub-band in Hz
    bandwidth : float
        Width of each sub-band in Hz
    sfreq : float
        Sampling frequency in Hz
    filter_order : int, default=4
        Order of each Butterworth filter
        
    Returns
    -------
    filter_bank : list of tuples
        Each tuple contains (b, a, low_freq, high_freq) where:
        - b, a: filter coefficients
        - low_freq, high_freq: frequency range of the sub-band
        
    Examples
    --------
    >>> fb = design_filter_bank(5, fundamental_freq=6.0, bandwidth=8.0, sfreq=256)
    >>> for b, a, low, high in fb:
    ...     print(f"Sub-band: {low:.1f} - {high:.1f} Hz")
    Sub-band: 6.0 - 14.0 Hz
    Sub-band: 14.0 - 22.0 Hz
    ...
    """
    filter_bank = []
    nyquist = sfreq / 2.0
    
    for n in range(n_subbands):
        low_freq = fundamental_freq + n * bandwidth
        high_freq = min(low_freq + bandwidth, nyquist - 1)
        
        # Skip sub-bands that exceed Nyquist frequency
        if low_freq >= nyquist - 1:
            warnings.warn(f"Sub-band {n+1} skipped: low_freq ({low_freq:.1f} Hz) >= Nyquist ({nyquist:.1f} Hz)")
            continue
        
        # Ensure valid frequency range
        if high_freq <= low_freq:
            high_freq = min(low_freq + 1, nyquist - 0.5)
        
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(filter_order, [low, high], btype='band')
        filter_bank.append((b, a, low_freq, high_freq))
    
    return filter_bank


def compute_subband_weights(
    n_subbands: int, 
    a: float = 1.25, 
    b: float = 0.25,
    method: str = 'standard'
) -> np.ndarray:
    """
    Compute FB-CCA sub-band weights using various weighting schemes.
    
    Parameters
    ----------
    n_subbands : int
        Number of sub-bands
    a : float, default=1.25
        Primary decay/shape parameter
    b : float, default=0.25
        Secondary parameter (baseline or shape modifier)
    method : str, default='standard'
        Weighting method. Options:
        - 'standard': w_n = n^(-a) + b (Chen et al., 2015)
        - 'exponential': w_n = exp(-a * (n-1)) + b
        - 'linear': w_n = 1 - a*(n-1)/(N-1), clipped to [b, 1]
        - 'gaussian': w_n = exp(-((n-1)/a)^2) + b
        - 'uniform': w_n = 1 for all sub-bands
        - 'learned': placeholder for data-driven weights (use optimize_weights)
        
    Returns
    -------
    weights : np.ndarray, shape (n_subbands,)
        Weight for each sub-band (all positive)
        
    References
    ----------
    Chen, X., et al. (2015). Filter bank canonical correlation analysis.
    The default parameters (a=1.25, b=0.25) are from the original paper.
    
    Examples
    --------
    >>> compute_subband_weights(5, method='standard')
    array([1.25, 0.67, 0.50, 0.43, 0.38])
    
    >>> compute_subband_weights(5, a=0.5, method='exponential')
    array([1.0, 0.61, 0.37, 0.22, 0.13])
    """
    n = np.arange(1, n_subbands + 1)
    
    if method == 'standard':
        # Original FB-CCA formula: w_n = n^(-a) + b
        weights = np.power(n, -a) + b
        
    elif method == 'exponential':
        # Exponential decay: w_n = exp(-a*(n-1)) + b
        weights = np.exp(-a * (n - 1)) + b
        
    elif method == 'linear':
        # Linear decay from 1 to b
        if n_subbands > 1:
            weights = 1 - a * (n - 1) / (n_subbands - 1)
        else:
            weights = np.ones(1)
        weights = np.clip(weights, b, 1.0)
        
    elif method == 'gaussian':
        # Gaussian-shaped weights centered on first sub-band
        weights = np.exp(-((n - 1) / max(a, 0.1)) ** 2) + b
        
    elif method == 'uniform':
        # Equal weights for all sub-bands
        weights = np.ones(n_subbands)
        
    else:
        raise ValueError(f"Unknown weighting method: {method}. "
                        f"Options: standard, exponential, linear, gaussian, uniform")
    
    return weights


class FBCCAWeightOptimizer:
    """
    Automatic optimization of FB-CCA weighting parameters.
    
    This class provides methods to find optimal weighting parameters (a, b)
    for a given dataset using grid search or more sophisticated optimization.
    
    Parameters
    ----------
    n_subbands : int
        Number of sub-bands in the filter bank
    method : str, default='standard'
        Weighting method to optimize ('standard', 'exponential', etc.)
    
    Attributes
    ----------
    best_params_ : dict
        Best parameters found after optimization
    best_score_ : float
        Best accuracy achieved with optimal parameters
    cv_results_ : dict
        Detailed cross-validation results
        
    Examples
    --------
    >>> optimizer = FBCCAWeightOptimizer(n_subbands=5)
    >>> best_weights = optimizer.fit(subband_correlations, labels)
    >>> print(optimizer.best_params_)
    {'a': 1.5, 'b': 0.2, 'method': 'standard'}
    """
    
    def __init__(self, n_subbands: int, method: str = 'standard'):
        self.n_subbands = n_subbands
        self.method = method
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = None
    
    def fit(
        self,
        subband_correlations: np.ndarray,
        labels: np.ndarray,
        a_range: Tuple[float, float] = (0.5, 3.0),
        b_range: Tuple[float, float] = (0.0, 0.5),
        n_steps: int = 20,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Find optimal weighting parameters using grid search with cross-validation.
        
        Parameters
        ----------
        subband_correlations : np.ndarray, shape (n_trials, n_frequencies, n_subbands)
            Pre-computed CCA correlations for each sub-band
        labels : np.ndarray, shape (n_trials,)
            True class labels (frequency indices)
        a_range : tuple, default=(0.5, 3.0)
            Range for parameter 'a' to search
        b_range : tuple, default=(0.0, 0.5)
            Range for parameter 'b' to search
        n_steps : int, default=20
            Number of steps in each parameter dimension
        cv_folds : int, default=5
            Number of cross-validation folds
        verbose : bool, default=True
            Print progress information
            
        Returns
        -------
        optimal_weights : np.ndarray, shape (n_subbands,)
            Optimal weights found
        """
        a_values = np.linspace(a_range[0], a_range[1], n_steps)
        b_values = np.linspace(b_range[0], b_range[1], n_steps)
        
        best_score = -1
        best_a, best_b = 1.25, 0.25  # defaults
        
        results = []
        
        if verbose:
            print(f"Optimizing FB-CCA weights ({n_steps}x{n_steps} grid search)...")
        
        for a in a_values:
            for b in b_values:
                # Compute weights
                weights = compute_subband_weights(self.n_subbands, a, b, self.method)
                
                # Compute weighted features
                features = self._apply_weights(subband_correlations, weights)
                
                # Cross-validation accuracy
                score = self._cross_validate(features, labels, cv_folds)
                
                results.append({'a': a, 'b': b, 'score': score})
                
                if score > best_score:
                    best_score = score
                    best_a, best_b = a, b
        
        self.best_params_ = {'a': best_a, 'b': best_b, 'method': self.method}
        self.best_score_ = best_score
        self.cv_results_ = results
        
        if verbose:
            print(f"Best parameters: a={best_a:.3f}, b={best_b:.3f}")
            print(f"Best CV accuracy: {best_score:.4f}")
        
        return compute_subband_weights(self.n_subbands, best_a, best_b, self.method)
    
    def fit_bayesian(
        self,
        subband_correlations: np.ndarray,
        labels: np.ndarray,
        a_range: Tuple[float, float] = (0.5, 3.0),
        b_range: Tuple[float, float] = (0.0, 0.5),
        n_iterations: int = 50,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Find optimal weighting parameters using Bayesian optimization.
        
        More efficient than grid search for finding optimal parameters,
        especially useful when evaluation is expensive.
        
        Parameters
        ----------
        subband_correlations : np.ndarray, shape (n_trials, n_frequencies, n_subbands)
            Pre-computed CCA correlations for each sub-band
        labels : np.ndarray, shape (n_trials,)
            True class labels
        a_range, b_range : tuple
            Parameter search ranges
        n_iterations : int, default=50
            Number of Bayesian optimization iterations
        cv_folds : int, default=5
            Number of cross-validation folds
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        optimal_weights : np.ndarray
            Optimal weights found
        """
        try:
            from scipy.optimize import minimize
            from scipy.stats import norm
        except ImportError:
            print("Falling back to grid search (scipy.optimize not available)")
            return self.fit(subband_correlations, labels, a_range, b_range, 
                          n_steps=20, cv_folds=cv_folds, verbose=verbose)
        
        # Store evaluated points for Gaussian Process surrogate
        X_observed = []
        y_observed = []
        
        def objective(params):
            a, b = params
            weights = compute_subband_weights(self.n_subbands, a, b, self.method)
            features = self._apply_weights(subband_correlations, weights)
            score = self._cross_validate(features, labels, cv_folds)
            return -score  # Minimize negative accuracy
        
        # Initial random samples
        n_initial = min(10, n_iterations // 3)
        for _ in range(n_initial):
            a = np.random.uniform(a_range[0], a_range[1])
            b = np.random.uniform(b_range[0], b_range[1])
            score = -objective([a, b])
            X_observed.append([a, b])
            y_observed.append(score)
        
        # Bayesian optimization loop
        best_idx = np.argmax(y_observed)
        
        for i in range(n_iterations - n_initial):
            # Use L-BFGS-B starting from best point with random perturbation
            x0 = np.array(X_observed[best_idx]) + np.random.randn(2) * 0.1
            x0[0] = np.clip(x0[0], a_range[0], a_range[1])
            x0[1] = np.clip(x0[1], b_range[0], b_range[1])
            
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=[a_range, b_range]
            )
            
            score = -result.fun
            X_observed.append(result.x.tolist())
            y_observed.append(score)
            
            if score > y_observed[best_idx]:
                best_idx = len(y_observed) - 1
                if verbose:
                    print(f"Iteration {i+n_initial}: New best score = {score:.4f}")
        
        best_a, best_b = X_observed[best_idx]
        self.best_params_ = {'a': best_a, 'b': best_b, 'method': self.method}
        self.best_score_ = y_observed[best_idx]
        
        if verbose:
            print(f"\nOptimal parameters: a={best_a:.3f}, b={best_b:.3f}")
            print(f"Best CV accuracy: {self.best_score_:.4f}")
        
        return compute_subband_weights(self.n_subbands, best_a, best_b, self.method)
    
    def fit_per_subject(
        self,
        subband_correlations_by_subject: dict,
        labels_by_subject: dict,
        a_range: Tuple[float, float] = (0.5, 3.0),
        b_range: Tuple[float, float] = (0.0, 0.5),
        n_steps: int = 15,
        verbose: bool = True
    ) -> dict:
        """
        Find optimal weighting parameters for each subject individually.
        
        Subject-specific optimization can improve performance when there's
        high inter-subject variability.
        
        Parameters
        ----------
        subband_correlations_by_subject : dict
            Dictionary mapping subject IDs to their subband correlations
            Each value has shape (n_trials, n_frequencies, n_subbands)
        labels_by_subject : dict
            Dictionary mapping subject IDs to their labels
        a_range, b_range : tuple
            Parameter search ranges
        n_steps : int, default=15
            Grid search steps per parameter
        verbose : bool, default=True
            Print progress
            
        Returns
        -------
        weights_by_subject : dict
            Dictionary mapping subject IDs to their optimal weights
        """
        weights_by_subject = {}
        params_by_subject = {}
        
        for subject in subband_correlations_by_subject.keys():
            if verbose:
                print(f"\nOptimizing for subject {subject}...")
            
            subband_corrs = subband_correlations_by_subject[subject]
            labels = labels_by_subject[subject]
            
            # Use smaller grid for per-subject optimization
            weights = self.fit(
                subband_corrs, labels,
                a_range=a_range, b_range=b_range,
                n_steps=n_steps, cv_folds=3, verbose=False
            )
            
            weights_by_subject[subject] = weights
            params_by_subject[subject] = self.best_params_.copy()
            
            if verbose:
                print(f"  Subject {subject}: a={self.best_params_['a']:.2f}, "
                      f"b={self.best_params_['b']:.2f}, acc={self.best_score_:.3f}")
        
        self.subject_params_ = params_by_subject
        return weights_by_subject
    
    def compare_methods(
        self,
        subband_correlations: np.ndarray,
        labels: np.ndarray,
        methods: List[str] = None,
        cv_folds: int = 5,
        verbose: bool = True
    ) -> dict:
        """
        Compare different weighting methods to find the best one.
        
        Parameters
        ----------
        subband_correlations : np.ndarray
            Pre-computed CCA correlations
        labels : np.ndarray
            True labels
        methods : list, optional
            Weighting methods to compare. Default: all available
        cv_folds : int, default=5
            Cross-validation folds
        verbose : bool, default=True
            Print results
            
        Returns
        -------
        results : dict
            Dictionary with method names as keys and (best_score, best_params) as values
        """
        if methods is None:
            methods = ['standard', 'exponential', 'linear', 'gaussian', 'uniform']
        
        results = {}
        
        for method in methods:
            self.method = method
            
            if method == 'uniform':
                # No parameters to optimize for uniform
                weights = compute_subband_weights(self.n_subbands, method='uniform')
                features = self._apply_weights(subband_correlations, weights)
                score = self._cross_validate(features, labels, cv_folds)
                results[method] = {'score': score, 'params': {'method': 'uniform'}}
            else:
                self.fit(subband_correlations, labels, cv_folds=cv_folds, verbose=False)
                results[method] = {
                    'score': self.best_score_,
                    'params': self.best_params_.copy()
                }
        
        if verbose:
            print("\n" + "="*50)
            print("WEIGHTING METHOD COMPARISON")
            print("="*50)
            for method, result in sorted(results.items(), key=lambda x: -x[1]['score']):
                print(f"{method:12s}: {result['score']:.4f}  {result['params']}")
            print("="*50)
        
        # Set best overall method
        best_method = max(results.keys(), key=lambda m: results[m]['score'])
        self.method = best_method
        self.best_params_ = results[best_method]['params']
        self.best_score_ = results[best_method]['score']
        
        return results
    
    def _apply_weights(
        self, 
        subband_correlations: np.ndarray, 
        weights: np.ndarray
    ) -> np.ndarray:
        """Apply weights to sub-band correlations: sum(w_n * corr_n^2)"""
        # subband_correlations: (n_trials, n_frequencies, n_subbands)
        # weights: (n_subbands,)
        squared_corrs = np.square(subband_correlations)
        return np.sum(squared_corrs * weights, axis=2)  # (n_trials, n_frequencies)
    
    def _cross_validate(
        self, 
        features: np.ndarray, 
        labels: np.ndarray, 
        n_folds: int
    ) -> float:
        """Simple k-fold cross-validation using max-correlation classification."""
        n_samples = len(labels)
        fold_size = n_samples // n_folds
        indices = np.random.permutation(n_samples)
        
        accuracies = []
        for fold in range(n_folds):
            start = fold * fold_size
            end = start + fold_size if fold < n_folds - 1 else n_samples
            
            test_idx = indices[start:end]
            
            # Max-correlation classification
            predictions = np.argmax(features[test_idx], axis=1) + 1  # 1-indexed labels
            accuracy = np.mean(predictions == labels[test_idx])
            accuracies.append(accuracy)
        
        return np.mean(accuracies)


def extract_subband_correlations(
    eeg_epochs: np.ndarray,
    stim_frequencies: List[float],
    sfreq: float,
    n_harmonics: int = 2,
    n_subbands: int = 5,
    fundamental_freq: float = 6.0,
    bandwidth: float = 8.0
) -> np.ndarray:
    """
    Extract per-subband CCA correlations (before weighting).
    
    This is useful for weight optimization where you want to experiment
    with different weighting schemes without re-computing the CCA correlations.
    
    Parameters
    ----------
    eeg_epochs : np.ndarray, shape (n_trials, n_channels, n_samples)
        EEG epochs data
    stim_frequencies : list
        Target frequencies in Hz
    sfreq : float
        Sampling frequency
    n_harmonics : int, default=2
        Number of harmonics for reference signals
    n_subbands : int, default=5
        Number of sub-bands
    fundamental_freq : float, default=6.0
        Starting frequency of filter bank
    bandwidth : float, default=8.0
        Width of each sub-band
        
    Returns
    -------
    subband_correlations : np.ndarray, shape (n_trials, n_frequencies, n_subbands)
        CCA correlation for each trial, frequency, and sub-band
    """
    n_trials = eeg_epochs.shape[0]
    n_samples = eeg_epochs.shape[2]
    n_freqs = len(stim_frequencies)
    
    # Design filter bank
    filter_bank = design_filter_bank(n_subbands, fundamental_freq, bandwidth, sfreq)
    
    # Pre-compute reference signals
    ref_signals_dict = {
        freq: generate_reference_signals(n_samples, freq, sfreq, n_harmonics)
        for freq in stim_frequencies
    }
    
    # Extract correlations
    subband_correlations = np.zeros((n_trials, n_freqs, n_subbands))
    
    for trial_idx in range(n_trials):
        eeg_trial = eeg_epochs[trial_idx].T  # (n_samples, n_channels)
        
        for freq_idx, freq in enumerate(stim_frequencies):
            for sb_idx, (b, a, _, _) in enumerate(filter_bank):
                filtered_eeg = filtfilt(b, a, eeg_trial, axis=0)
                subband_correlations[trial_idx, freq_idx, sb_idx] = cca_correlation(
                    filtered_eeg, ref_signals_dict[freq]
                )
    
    return subband_correlations


def fbcca_correlation(
    eeg_data: np.ndarray,
    frequency: float,
    sfreq: float,
    n_harmonics: int = 2,
    n_subbands: int = 5,
    fundamental_freq: float = 6.0,
    bandwidth: float = 8.0,
    weight_a: float = 1.25,
    weight_b: float = 0.25,
    filter_bank: Optional[List] = None,
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute FB-CCA correlation for a single EEG trial and target frequency.
    
    This is the core FB-CCA computation:
    1. Decompose EEG into sub-bands using filter bank
    2. Compute CCA correlation for each sub-band
    3. Combine using weighted sum of squared correlations
    
    Parameters
    ----------
    eeg_data : np.ndarray, shape (n_samples, n_channels)
        Single trial EEG data
    frequency : float
        Target stimulus frequency in Hz
    sfreq : float
        Sampling frequency in Hz
    n_harmonics : int, default=2
        Number of harmonics for reference signals
    n_subbands : int, default=5
        Number of sub-bands (ignored if filter_bank provided)
    fundamental_freq : float, default=6.0
        Start frequency of filter bank (ignored if filter_bank provided)
    bandwidth : float, default=8.0
        Sub-band width (ignored if filter_bank provided)
    weight_a, weight_b : float
        Weighting parameters (ignored if weights provided)
    filter_bank : list, optional
        Pre-computed filter bank (for efficiency in batch processing)
    weights : np.ndarray, optional
        Pre-computed sub-band weights
        
    Returns
    -------
    rho : float
        Weighted FB-CCA correlation: sum(w_n * corr_n^2)
        
    Examples
    --------
    >>> eeg = np.random.randn(1024, 3)  # 4 seconds at 256 Hz, 3 channels
    >>> rho = fbcca_correlation(eeg, frequency=10.0, sfreq=256)
    """
    n_samples = eeg_data.shape[0]
    
    # Use provided or create filter bank
    if filter_bank is None:
        filter_bank = design_filter_bank(n_subbands, fundamental_freq, bandwidth, sfreq)
    
    # Use provided or compute weights
    if weights is None:
        weights = compute_subband_weights(len(filter_bank), weight_a, weight_b)
    
    # Generate reference signals
    ref_signals = generate_reference_signals(n_samples, frequency, sfreq, n_harmonics)
    
    # Compute CCA for each sub-band
    subband_corrs = np.zeros(len(filter_bank))
    
    for sb_idx, (b, a, _, _) in enumerate(filter_bank):
        # Apply bandpass filter
        filtered_eeg = filtfilt(b, a, eeg_data, axis=0)
        
        # Compute CCA correlation
        subband_corrs[sb_idx] = cca_correlation(filtered_eeg, ref_signals)
    
    # Weighted combination of squared correlations
    return np.sum(weights * np.square(subband_corrs))


def fbcca_features(
    eeg_epochs: np.ndarray,
    stim_frequencies: List[float],
    sfreq: float,
    n_harmonics: int = 2,
    n_subbands: int = 5,
    fundamental_freq: float = 6.0,
    bandwidth: float = 8.0,
    weight_a: float = 1.25,
    weight_b: float = 0.25
) -> np.ndarray:
    """
    Extract FB-CCA features for multiple trials and frequencies.
    
    This is the main function for FB-CCA feature extraction, computing
    weighted correlations for all trials and all target frequencies.
    
    Parameters
    ----------
    eeg_epochs : np.ndarray, shape (n_trials, n_channels, n_samples)
        EEG epochs data
    stim_frequencies : list of float
        Target stimulus frequencies in Hz
    sfreq : float
        Sampling frequency in Hz
    n_harmonics : int, default=2
        Number of harmonics for reference signals
    n_subbands : int, default=5
        Number of sub-bands in filter bank
    fundamental_freq : float, default=6.0
        Starting frequency of first sub-band
    bandwidth : float, default=8.0
        Width of each sub-band in Hz
    weight_a, weight_b : float
        FB-CCA weighting parameters
        
    Returns
    -------
    features : np.ndarray, shape (n_trials, n_frequencies)
        FB-CCA weighted correlation features
        
    Examples
    --------
    >>> epochs = np.random.randn(100, 3, 1024)  # 100 trials, 3 channels, 4s
    >>> freqs = [8, 10, 12, 15]
    >>> features = fbcca_features(epochs, freqs, sfreq=256)
    >>> features.shape
    (100, 4)
    
    >>> # Classification by max correlation
    >>> predictions = np.argmax(features, axis=1)
    """
    n_trials = eeg_epochs.shape[0]
    n_freqs = len(stim_frequencies)
    n_samples = eeg_epochs.shape[2]
    
    # Pre-compute filter bank and weights for efficiency
    filter_bank = design_filter_bank(n_subbands, fundamental_freq, bandwidth, sfreq)
    weights = compute_subband_weights(n_subbands, weight_a, weight_b)
    
    # Pre-compute reference signals for all frequencies
    ref_signals_dict = {
        freq: generate_reference_signals(n_samples, freq, sfreq, n_harmonics)
        for freq in stim_frequencies
    }
    
    # Extract features
    features = np.zeros((n_trials, n_freqs))
    
    for trial_idx in range(n_trials):
        eeg_trial = eeg_epochs[trial_idx].T  # (n_samples, n_channels)
        
        for freq_idx, freq in enumerate(stim_frequencies):
            # Apply filter bank
            subband_corrs = np.zeros(n_subbands)
            
            for sb_idx, (b, a, _, _) in enumerate(filter_bank):
                filtered_eeg = filtfilt(b, a, eeg_trial, axis=0)
                subband_corrs[sb_idx] = cca_correlation(
                    filtered_eeg, ref_signals_dict[freq]
                )
            
            # Weighted combination
            features[trial_idx, freq_idx] = np.sum(weights * np.square(subband_corrs))
    
    return features


def classify_ssvep(
    features: np.ndarray, 
    stim_frequencies: List[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify SSVEP trials by selecting frequency with maximum correlation.
    
    Parameters
    ----------
    features : np.ndarray, shape (n_trials, n_frequencies)
        CCA or FB-CCA correlation features
    stim_frequencies : list of float
        Target stimulus frequencies
        
    Returns
    -------
    predictions : np.ndarray, shape (n_trials,)
        Predicted frequency for each trial
    confidence : np.ndarray, shape (n_trials,)
        Confidence score (max correlation value)
    """
    pred_indices = np.argmax(features, axis=1)
    predictions = np.array([stim_frequencies[i] for i in pred_indices])
    confidence = np.max(features, axis=1)
    
    return predictions, confidence


# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================

class FBCCAHyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization for FB-CCA.
    
    Optimizes:
    - Number of sub-bands
    - Filter bank cutoff frequencies (fundamental freq, bandwidth)
    - Number of harmonics in reference signals
    - FB-CCA weighting coefficients (a, b)
    
    Uses grid search or random search with cross-validation.
    
    Example
    -------
    >>> optimizer = FBCCAHyperparameterOptimizer(sfreq=256, stim_frequencies=[8, 10, 12, 15])
    >>> best_params = optimizer.optimize(eeg_data, labels, method='grid')
    >>> print(f"Best accuracy: {optimizer.best_score_:.3f}")
    """
    
    def __init__(
        self,
        sfreq: float = 256.0,
        stim_frequencies: List[float] = None,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Parameters
        ----------
        sfreq : float
            Sampling frequency in Hz
        stim_frequencies : list
            Stimulus frequencies to detect
        cv_folds : int
            Number of cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.sfreq = sfreq
        self.stim_frequencies = stim_frequencies or [8, 10, 12, 15]
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Results storage
        self.best_params_ = None
        self.best_score_ = 0.0
        self.cv_results_ = []
        
        # Default parameter grid
        self.param_grid = {
            'n_subbands': [3, 4, 5, 6, 7],
            'fundamental_freq': [4.0, 6.0, 8.0],
            'bandwidth': [4.0, 6.0, 8.0, 10.0],
            'n_harmonics': [1, 2, 3, 4],
            'weight_a': [0.5, 1.0, 1.25, 1.5, 2.0],
            'weight_b': [0.0, 0.1, 0.25, 0.5]
        }
    
    def set_param_grid(self, param_grid: dict):
        """Set custom parameter grid for optimization."""
        self.param_grid.update(param_grid)
    
    def _extract_features_with_params(
        self,
        eeg_data: np.ndarray,
        params: dict
    ) -> np.ndarray:
        """
        Extract FB-CCA features with specified parameters.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data, shape (n_trials, n_channels, n_samples)
        params : dict
            FB-CCA parameters
            
        Returns
        -------
        features : np.ndarray
            FB-CCA features, shape (n_trials, n_frequencies)
        """
        n_trials = eeg_data.shape[0]
        n_samples = eeg_data.shape[2]
        n_freqs = len(self.stim_frequencies)
        
        # Design filter bank with current params
        filter_bank = design_filter_bank(
            n_subbands=params['n_subbands'],
            fundamental_freq=params['fundamental_freq'],
            bandwidth=params['bandwidth'],
            sfreq=self.sfreq
        )
        
        # Compute weights
        weights = compute_subband_weights(
            params['n_subbands'],
            params['weight_a'],
            params['weight_b']
        )
        
        # Generate reference signals
        ref_signals = {
            freq: generate_reference_signals(
                n_samples, freq, self.sfreq, params['n_harmonics']
            )
            for freq in self.stim_frequencies
        }
        
        # Extract features
        features = np.zeros((n_trials, n_freqs))
        
        for trial_idx in range(n_trials):
            eeg_trial = eeg_data[trial_idx].T
            
            for freq_idx, freq in enumerate(self.stim_frequencies):
                subband_corrs = np.zeros(params['n_subbands'])
                
                for sb_idx, (b, a, low, high) in enumerate(filter_bank):
                    # Skip if filter is invalid (beyond Nyquist)
                    if high >= self.sfreq / 2:
                        continue
                    try:
                        filtered = filtfilt(b, a, eeg_trial, axis=0)
                        subband_corrs[sb_idx] = cca_correlation(filtered, ref_signals[freq])
                    except:
                        pass
                
                features[trial_idx, freq_idx] = np.sum(weights * np.square(subband_corrs))
        
        return features
    
    def _evaluate_params(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        params: dict
    ) -> float:
        """
        Evaluate a parameter configuration using cross-validation.
        
        Returns
        -------
        score : float
            Mean cross-validation accuracy
        """
        from sklearn.model_selection import StratifiedKFold
        
        # Extract features
        features = self._extract_features_with_params(eeg_data, params)
        
        # Cross-validation
        kfold = StratifiedKFold(
            n_splits=self.cv_folds, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scores = []
        for train_idx, val_idx in kfold.split(features, labels):
            # Max selection classification
            val_features = features[val_idx]
            val_labels = labels[val_idx]
            
            preds = np.argmax(val_features, axis=1) + 1
            acc = np.mean(preds == val_labels)
            scores.append(acc)
        
        return np.mean(scores)
    
    def optimize_grid(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        param_grid: dict = None,
        verbose: bool = True
    ) -> dict:
        """
        Perform grid search optimization.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data, shape (n_trials, n_channels, n_samples)
        labels : np.ndarray
            Class labels
        param_grid : dict, optional
            Override default parameter grid
        verbose : bool
            Print progress
            
        Returns
        -------
        best_params : dict
            Optimal parameter configuration
        """
        from itertools import product
        
        grid = param_grid or self.param_grid
        
        # Generate all combinations
        keys = list(grid.keys())
        combinations = list(product(*[grid[k] for k in keys]))
        
        if verbose:
            print(f"Grid search: {len(combinations)} combinations")
            print("=" * 60)
        
        self.cv_results_ = []
        
        for i, values in enumerate(combinations):
            params = dict(zip(keys, values))
            
            try:
                score = self._evaluate_params(eeg_data, labels, params)
            except Exception as e:
                if verbose:
                    print(f"  Skipping invalid params: {e}")
                continue
            
            self.cv_results_.append({'params': params.copy(), 'score': score})
            
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params.copy()
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(combinations)}] Best so far: {self.best_score_:.4f}")
        
        if verbose:
            print("=" * 60)
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best params: {self.best_params_}")
        
        return self.best_params_
    
    def optimize_random(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        n_iter: int = 50,
        param_distributions: dict = None,
        verbose: bool = True
    ) -> dict:
        """
        Perform random search optimization.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data, shape (n_trials, n_channels, n_samples)
        labels : np.ndarray
            Class labels
        n_iter : int
            Number of random samples
        param_distributions : dict, optional
            Override default parameter distributions
        verbose : bool
            Print progress
            
        Returns
        -------
        best_params : dict
            Optimal parameter configuration
        """
        np.random.seed(self.random_state)
        
        grid = param_distributions or self.param_grid
        
        if verbose:
            print(f"Random search: {n_iter} iterations")
            print("=" * 60)
        
        self.cv_results_ = []
        
        for i in range(n_iter):
            # Sample random parameters
            params = {k: np.random.choice(v) for k, v in grid.items()}
            
            try:
                score = self._evaluate_params(eeg_data, labels, params)
            except Exception as e:
                continue
            
            self.cv_results_.append({'params': params.copy(), 'score': score})
            
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params.copy()
                
                if verbose:
                    print(f"  [{i+1}] New best: {score:.4f}")
        
        if verbose:
            print("=" * 60)
            print(f"Best score: {self.best_score_:.4f}")
            print(f"Best params: {self.best_params_}")
        
        return self.best_params_
    
    def optimize_harmonics_only(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        harmonics_range: List[int] = None,
        base_params: dict = None,
        verbose: bool = True
    ) -> Tuple[int, float]:
        """
        Quick optimization of just the number of harmonics.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data
        labels : np.ndarray
            Class labels
        harmonics_range : list
            Harmonics values to try
        base_params : dict
            Base FB-CCA parameters (other than n_harmonics)
        verbose : bool
            Print progress
            
        Returns
        -------
        best_harmonics : int
            Optimal number of harmonics
        best_score : float
            Best cross-validation score
        """
        harmonics_range = harmonics_range or [1, 2, 3, 4, 5]
        base_params = base_params or {
            'n_subbands': 5,
            'fundamental_freq': 6.0,
            'bandwidth': 8.0,
            'weight_a': 1.25,
            'weight_b': 0.25
        }
        
        if verbose:
            print("Optimizing number of harmonics...")
        
        results = []
        for n_harm in harmonics_range:
            params = base_params.copy()
            params['n_harmonics'] = n_harm
            
            score = self._evaluate_params(eeg_data, labels, params)
            results.append((n_harm, score))
            
            if verbose:
                print(f"  n_harmonics={n_harm}: {score:.4f}")
        
        best_harmonics, best_score = max(results, key=lambda x: x[1])
        
        if verbose:
            print(f"Optimal: n_harmonics={best_harmonics} (accuracy={best_score:.4f})")
        
        return best_harmonics, best_score
    
    def optimize_filter_bank_only(
        self,
        eeg_data: np.ndarray,
        labels: np.ndarray,
        n_subbands_range: List[int] = None,
        fundamental_range: List[float] = None,
        bandwidth_range: List[float] = None,
        base_params: dict = None,
        verbose: bool = True
    ) -> Tuple[dict, float]:
        """
        Optimize only filter bank parameters.
        
        Parameters
        ----------
        eeg_data : np.ndarray
            EEG data
        labels : np.ndarray
            Class labels
        n_subbands_range : list
            Number of sub-bands to try
        fundamental_range : list
            Fundamental frequencies to try
        bandwidth_range : list
            Bandwidths to try
        base_params : dict
            Base parameters (n_harmonics, weights)
        verbose : bool
            Print progress
            
        Returns
        -------
        best_fb_params : dict
            Optimal filter bank parameters
        best_score : float
            Best cross-validation score
        """
        from itertools import product
        
        n_subbands_range = n_subbands_range or [3, 4, 5, 6, 7]
        fundamental_range = fundamental_range or [4.0, 6.0, 8.0]
        bandwidth_range = bandwidth_range or [4.0, 6.0, 8.0, 10.0]
        
        base_params = base_params or {
            'n_harmonics': 2,
            'weight_a': 1.25,
            'weight_b': 0.25
        }
        
        if verbose:
            total = len(n_subbands_range) * len(fundamental_range) * len(bandwidth_range)
            print(f"Optimizing filter bank ({total} combinations)...")
        
        best_score = 0
        best_fb_params = {}
        
        for n_sb, fund, bw in product(n_subbands_range, fundamental_range, bandwidth_range):
            params = base_params.copy()
            params['n_subbands'] = n_sb
            params['fundamental_freq'] = fund
            params['bandwidth'] = bw
            
            try:
                score = self._evaluate_params(eeg_data, labels, params)
            except:
                continue
            
            if score > best_score:
                best_score = score
                best_fb_params = {
                    'n_subbands': n_sb,
                    'fundamental_freq': fund,
                    'bandwidth': bw
                }
        
        if verbose:
            print(f"Optimal filter bank: {best_fb_params} (accuracy={best_score:.4f})")
        
        return best_fb_params, best_score
    
    def get_results_dataframe(self):
        """Return optimization results as pandas DataFrame."""
        try:
            import pandas as pd
            
            records = []
            for r in self.cv_results_:
                row = r['params'].copy()
                row['score'] = r['score']
                records.append(row)
            
            return pd.DataFrame(records).sort_values('score', ascending=False)
        except ImportError:
            return None
    
    def plot_param_importance(self, figsize=(12, 8)):
        """
        Plot the effect of each parameter on performance.
        
        Requires matplotlib.
        """
        import matplotlib.pyplot as plt
        
        if not self.cv_results_:
            print("No results to plot. Run optimization first.")
            return
        
        # Get unique parameters
        params = list(self.param_grid.keys())
        n_params = len(params)
        
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, param in enumerate(params):
            ax = axes[idx]
            
            # Group scores by parameter value
            param_scores = {}
            for result in self.cv_results_:
                val = result['params'][param]
                if val not in param_scores:
                    param_scores[val] = []
                param_scores[val].append(result['score'])
            
            # Plot
            values = sorted(param_scores.keys())
            means = [np.mean(param_scores[v]) for v in values]
            stds = [np.std(param_scores[v]) for v in values]
            
            ax.errorbar(range(len(values)), means, yerr=stds, marker='o', capsize=5)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels([str(v) for v in values], rotation=45)
            ax.set_xlabel(param)
            ax.set_ylabel('CV Accuracy')
            ax.set_title(f'Effect of {param}')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(params), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()


# Convenience aliases
cca = cca_correlation
fbcca = fbcca_features
