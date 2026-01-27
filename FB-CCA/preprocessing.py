"""
EEG preprocessing module for SSVEP analysis.

Provides preprocessing pipelines including:
- Automatic artifact rejection with detailed statistics
- Baseline correction
- Resampling and filtering
- Event standardization across subjects
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from numpy.typing import NDArray


@dataclass
class RejectionStats:
    """
    Statistics from artifact rejection.
    
    Attributes
    ----------
    n_original : int
        Number of epochs before rejection
    n_rejected : int
        Number of epochs rejected
    n_remaining : int
        Number of epochs remaining
    percent_rejected : float
        Percentage of epochs rejected
    reasons : dict
        Breakdown of rejections by reason
    """
    n_original: int
    n_rejected: int
    n_remaining: int
    percent_rejected: float
    reasons: Dict[str, int] = field(default_factory=dict)
    
    @property
    def rejection_rate(self) -> float:
        """Rejection rate as a fraction [0, 1]."""
        return self.percent_rejected / 100
    
    def __str__(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Artifact Rejection: {self.n_rejected}/{self.n_original} epochs ({self.percent_rejected:.1f}%)",
            f"  Remaining: {self.n_remaining} epochs"
        ]
        for reason, count in self.reasons.items():
            lines.append(f"  - {reason}: {count}")
        return '\n'.join(lines)


@dataclass
class ArtifactRejectorConfig:
    """Configuration for artifact rejection thresholds."""
    reject_peak_to_peak: Optional[float] = 150e-6  # 150 µV
    reject_variance_threshold: Optional[float] = 3.0  # Z-score
    reject_flat_threshold: Optional[float] = 1e-6  # 1 µV
    verbose: bool = True


class ArtifactRejector:
    """
    Automatic artifact rejection for EEG epochs.
    
    Implements multiple automated rejection methods suitable for online BCI:
    - Peak-to-peak threshold rejection
    - Variance-based rejection (z-score)
    - Flat channel detection
    
    These methods are computationally efficient and don't require manual inspection,
    making them suitable for real-time applications.
    
    Examples
    --------
    >>> rejector = ArtifactRejector(reject_peak_to_peak=150e-6)
    >>> clean_epochs = rejector.reject_epochs(epochs)
    >>> print(rejector.stats)
    
    >>> # Using config object
    >>> config = ArtifactRejectorConfig(reject_variance_threshold=2.5)
    >>> rejector = ArtifactRejector.from_config(config)
    """
    
    __slots__ = (
        '_reject_peak_to_peak', '_reject_variance_threshold',
        '_reject_flat_threshold', '_verbose', '_stats'
    )
    
    def __init__(
        self, 
        reject_peak_to_peak: Optional[float] = 150e-6,
        reject_variance_threshold: Optional[float] = 3.0,
        reject_flat_threshold: Optional[float] = 1e-6,
        verbose: bool = True
    ) -> None:
        """
        Initialize artifact rejector with threshold parameters.
        
        Parameters
        ----------
        reject_peak_to_peak : float, optional
            Maximum peak-to-peak amplitude in Volts (default: 150 µV)
        reject_variance_threshold : float, optional
            Z-score threshold for variance rejection (default: 3.0)
        reject_flat_threshold : float, optional
            Minimum peak-to-peak amplitude (default: 1 µV)
        verbose : bool
            Whether to print rejection statistics
        """
        self._reject_peak_to_peak = reject_peak_to_peak
        self._reject_variance_threshold = reject_variance_threshold
        self._reject_flat_threshold = reject_flat_threshold
        self._verbose = verbose
        self._stats: Optional[RejectionStats] = None
    
    @classmethod
    def from_config(cls, config: ArtifactRejectorConfig) -> 'ArtifactRejector':
        """Create rejector from configuration object."""
        return cls(
            reject_peak_to_peak=config.reject_peak_to_peak,
            reject_variance_threshold=config.reject_variance_threshold,
            reject_flat_threshold=config.reject_flat_threshold,
            verbose=config.verbose
        )
    
    @property
    def stats(self) -> Optional[RejectionStats]:
        """Rejection statistics from last run."""
        return self._stats
    
    def reject_epochs(self, epochs: mne.Epochs) -> mne.Epochs:
        """
        Apply automatic artifact rejection to epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs to clean
            
        Returns
        -------
        clean_epochs : mne.Epochs
            Epochs with artifacts removed
        """
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        n_epochs_original = len(data)
        
        reject_mask = np.zeros(n_epochs_original, dtype=bool)
        reasons: Dict[str, int] = {'peak_to_peak': 0, 'variance': 0, 'flat': 0}
        
        # 1. Peak-to-peak threshold
        if self._reject_peak_to_peak is not None:
            ptp = np.ptp(data, axis=2).max(axis=1)
            bad_ptp = ptp > self._reject_peak_to_peak
            reasons['peak_to_peak'] = int(bad_ptp.sum())
            reject_mask |= bad_ptp
        
        # 2. Flat channel detection
        if self._reject_flat_threshold is not None:
            ptp = np.ptp(data, axis=2).min(axis=1)
            bad_flat = ptp < self._reject_flat_threshold
            reasons['flat'] = int(bad_flat.sum())
            reject_mask |= bad_flat
        
        # 3. Variance-based rejection (z-score)
        if self._reject_variance_threshold is not None:
            var_per_epoch = np.var(data, axis=(1, 2))
            mean_var = np.mean(var_per_epoch)
            std_var = np.std(var_per_epoch)
            if std_var > 0:
                z_scores = np.abs(var_per_epoch - mean_var) / std_var
                bad_var = z_scores > self._reject_variance_threshold
                reasons['variance'] = int(bad_var.sum())
                reject_mask |= bad_var
        
        # Apply rejection
        good_indices = np.where(~reject_mask)[0]
        clean_epochs = epochs[good_indices]
        
        # Store statistics
        n_rejected = int(reject_mask.sum())
        self._stats = RejectionStats(
            n_original=n_epochs_original,
            n_rejected=n_rejected,
            n_remaining=len(good_indices),
            percent_rejected=100 * n_rejected / n_epochs_original if n_epochs_original > 0 else 0,
            reasons=reasons
        )
        
        if self._verbose:
            print(self._stats)
        
        return clean_epochs
    
    def get_stats(self) -> Dict[str, Any]:
        """Return rejection statistics as dictionary (legacy compatibility)."""
        if self._stats is None:
            return {}
        return {
            'n_original': self._stats.n_original,
            'n_rejected': self._stats.n_rejected,
            'n_remaining': self._stats.n_remaining,
            'percent_rejected': self._stats.percent_rejected,
            'reasons': self._stats.reasons
        }


@dataclass
class PreprocessorConfig:
    """Configuration for EEG preprocessing pipeline."""
    sfreq: int = 256
    l_freq: Optional[float] = None
    h_freq: Optional[float] = None
    event_mapping_path: Path = Path("event_mapping.json")
    invalid_keys: List[str] = field(default_factory=lambda: ['100', '99', '36'])
    baseline: Optional[Tuple[float, float]] = (-0.2, 0)
    artifact_rejection: bool = True
    reject_peak_to_peak: float = 150e-6


class EEGPreprocessor:
    """
    EEG preprocessing pipeline for SSVEP analysis.
    
    Handles resampling, filtering, epoching, and event standardization
    across subjects with automatic artifact rejection.
    
    Examples
    --------
    >>> preprocessor = EEGPreprocessor(sfreq=256, l_freq=1.0, h_freq=50.0)
    >>> epochs = preprocessor.create_epochs_from_raw(raw, tmin=0.5, tmax=4.5)
    
    >>> # Using config object
    >>> config = PreprocessorConfig(sfreq=512, baseline=(-0.5, 0))
    >>> preprocessor = EEGPreprocessor.from_config(config)
    """
    
    __slots__ = (
        '_sfreq', '_l_freq', '_h_freq', '_event_mapping_path',
        '_invalid_keys', '_standard_event_map', '_baseline',
        '_artifact_rejection', '_artifact_rejector'
    )

    def __init__(
        self, 
        sfreq: int = 256, 
        l_freq: Optional[float] = None, 
        h_freq: Optional[float] = None, 
        event_mapping_path: str | Path = "event_mapping.json",
        invalid_keys: Optional[List[str]] = None,
        baseline: Optional[Tuple[float, float]] = (-0.2, 0),
        artifact_rejection: bool = True,
        reject_peak_to_peak: float = 150e-6
    ) -> None:
        """
        Initialize EEG preprocessing parameters.

        Parameters
        ----------
        sfreq : int
            Target sampling frequency (Hz)
        l_freq : float, optional
            Low cutoff for bandpass filter
        h_freq : float, optional
            High cutoff for bandpass filter
        event_mapping_path : str or Path
            Path for event ID mapping storage
        invalid_keys : list of str, optional
            Event keys to exclude
        baseline : tuple of (float, float), optional
            Time interval for baseline correction (start, end) in seconds.
            Default: (-0.2, 0). Set to None to disable.
        artifact_rejection : bool
            Whether to apply automatic artifact rejection
        reject_peak_to_peak : float
            Peak-to-peak threshold in Volts (default: 150 µV)
        """
        self._sfreq = sfreq
        self._l_freq = l_freq
        self._h_freq = h_freq
        self._event_mapping_path = Path(event_mapping_path)
        self._invalid_keys = invalid_keys or ['100', '99', '36']
        self._standard_event_map = self._load_event_mapping()
        self._baseline = baseline
        self._artifact_rejection = artifact_rejection
        
        self._artifact_rejector: Optional[ArtifactRejector] = (
            ArtifactRejector(reject_peak_to_peak=reject_peak_to_peak, verbose=True)
            if artifact_rejection else None
        )
    
    @classmethod
    def from_config(cls, config: PreprocessorConfig) -> 'EEGPreprocessor':
        """Create preprocessor from configuration object."""
        return cls(
            sfreq=config.sfreq,
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            event_mapping_path=config.event_mapping_path,
            invalid_keys=config.invalid_keys,
            baseline=config.baseline,
            artifact_rejection=config.artifact_rejection,
            reject_peak_to_peak=config.reject_peak_to_peak
        )
    
    # -------------------- Properties --------------------
    
    @property
    def sfreq(self) -> int:
        """Target sampling frequency."""
        return self._sfreq
    
    @property
    def baseline(self) -> Optional[Tuple[float, float]]:
        """Baseline correction interval."""
        return self._baseline
    
    @property
    def artifact_rejector(self) -> Optional[ArtifactRejector]:
        """Artifact rejector instance."""
        return self._artifact_rejector
    
    # -------------------- Event Mapping --------------------
    
    def _load_event_mapping(self) -> Dict[str, int]:
        """Load standardized event mappings from JSON file."""
        try:
            with self._event_mapping_path.open('r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_event_mapping(self) -> None:
        """Save event mappings to JSON file."""
        with self._event_mapping_path.open('w') as f:
            json.dump(self._standard_event_map, f, indent=4)
    
    # -------------------- Preprocessing Methods --------------------
    
    def resample_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Resample EEG data to target sampling frequency."""
        if raw.info['sfreq'] != self._sfreq:
            raw_resampled = raw.copy().resample(self._sfreq, npad="auto")
            print(f"Data resampled to {self._sfreq} Hz.")
            return raw_resampled
        return raw

    def filter_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply bandpass filter to EEG signals."""
        raw_filtered = raw.copy().filter(
            l_freq=self._l_freq, 
            h_freq=self._h_freq, 
            fir_design='firwin', 
            verbose=False
        )
        print(f"Applied bandpass filter: {self._l_freq} - {self._h_freq} Hz.")
        return raw_filtered

    def get_event_mapping(self, raw: mne.io.Raw) -> Tuple[NDArray, Dict[str, int]]:
        """
        Extract events and mapping from EEG annotations.

        Returns
        -------
        events : ndarray
            Event timestamps and IDs
        event_id : dict
            Event name to ID mapping
        """
        events, event_id = mne.events_from_annotations(raw)
        
        # Retain only present event IDs
        unique_codes = set(events[:, 2])
        filtered_event_id = {k: v for k, v in event_id.items() if v in unique_codes}
        
        print(f"Extracted {len(events)} events. IDs: {filtered_event_id}")
        return events, filtered_event_id

    def standardize_event_ids(
        self, 
        events: NDArray, 
        event_id: Dict[str, int]
    ) -> Tuple[NDArray, Dict[str, int]]:
        """
        Standardize event IDs across datasets.

        Parameters
        ----------
        events : ndarray
            Original events array
        event_id : dict
            Original event mapping

        Returns
        -------
        standardized_events : ndarray
            Events with standardized IDs
        standardized_event_id : dict
            Global standardized mapping
        """
        # Filter invalid keys
        valid_event_id = {k: v for k, v in event_id.items() if k not in self._invalid_keys}

        # Initialize global map if empty
        if not self._standard_event_map:
            self._standard_event_map = {
                key: idx + 1 
                for idx, key in enumerate(sorted(valid_event_id.keys()))
            }

        # Update events with standardized IDs
        standardized_events = events.copy()
        standardized_event_id = {
            key: self._standard_event_map[key] 
            for key in valid_event_id if key in self._standard_event_map
        }

        for i, event in enumerate(standardized_events):
            event_code = event[2]
            matching_keys = [k for k, v in event_id.items() if v == event_code]
            
            if matching_keys and matching_keys[0] in standardized_event_id:
                standardized_events[i, 2] = standardized_event_id[matching_keys[0]]

        self._save_event_mapping()
        return standardized_events, standardized_event_id

    def create_epochs_from_raw(
        self, 
        raw: mne.io.Raw, 
        tmin: float = 0.5, 
        tmax: float = 4.5,
        valid_keys: Optional[List[str]] = None,
        apply_baseline: Optional[bool] = None,
        apply_artifact_rejection: Optional[bool] = None
    ) -> Optional[mne.Epochs]:
        """
        Generate epochs from EEG data based on standardized events.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        tmin : float
            Epoch start time (seconds)
        tmax : float
            Epoch end time (seconds)
        valid_keys : list of str, optional
            Event keys to include
        apply_baseline : bool, optional
            Whether to apply baseline correction
        apply_artifact_rejection : bool, optional
            Whether to apply artifact rejection

        Returns
        -------
        epochs : mne.Epochs or None
            Segmented epochs, or None if no valid events
        """
        raw_resampled = self.resample_data(raw)
        events, event_id = self.get_event_mapping(raw_resampled)
        standardized_events, event_id = self.standardize_event_ids(events, event_id)

        valid_keys = valid_keys or list(event_id.keys())
        valid_codes = [event_id[k] for k in valid_keys if k in event_id]
        trial_events = standardized_events[np.isin(standardized_events[:, 2], valid_codes)]

        if len(trial_events) == 0:
            print("No valid events found.")
            return None

        # Determine baseline setting
        use_baseline = apply_baseline if apply_baseline is not None else (self._baseline is not None)
        baseline_tuple = self._baseline if use_baseline else None
        
        # Adjust tmin if baseline requires earlier start
        effective_tmin = min(tmin, baseline_tuple[0]) if baseline_tuple else tmin
        
        epochs = mne.Epochs(
            raw_resampled, 
            events=trial_events, 
            event_id={k: event_id[k] for k in valid_keys if k in event_id},
            tmin=effective_tmin, 
            tmax=tmax, 
            baseline=baseline_tuple,
            preload=True
        )
        
        if baseline_tuple:
            print(f"Baseline correction: {baseline_tuple[0]}s to {baseline_tuple[1]}s")
        
        # Apply artifact rejection
        use_rejection = (
            apply_artifact_rejection if apply_artifact_rejection is not None 
            else self._artifact_rejection
        )
        
        if use_rejection and self._artifact_rejector is not None:
            epochs = self._artifact_rejector.reject_epochs(epochs)
        
        # Crop to original time window if baseline extended it
        if effective_tmin < tmin:
            epochs = epochs.crop(tmin=tmin, tmax=tmax)

        return epochs
    
    # -------------------- Legacy Compatibility --------------------
    
    def load_event_mapping(self) -> Dict[str, int]:
        """Legacy method for backward compatibility."""
        return self._load_event_mapping()
    
    def save_event_mapping(self) -> None:
        """Legacy method for backward compatibility."""
        self._save_event_mapping()
