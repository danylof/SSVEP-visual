"""
EEG classification module for SSVEP analysis.

Provides classification methods and comprehensive evaluation metrics for
CCA and FB-CCA based SSVEP recognition:
- Max correlation selection
- K-Nearest Neighbors with leave-subject-out cross-validation
- ITR computation and comprehensive metrics
- Model persistence (save/load)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier


class ClassificationMethod(Enum):
    """Classification method enumeration."""
    MAX = auto()      # Maximum correlation selection
    KNN = auto()      # K-Nearest Neighbors
    FBCCA_MAX = auto()  # FB-CCA specific max selection
    
    @classmethod
    def from_string(cls, method: str) -> 'ClassificationMethod':
        """Convert string to enum (case-insensitive)."""
        method_map = {
            'max': cls.MAX,
            'knn': cls.KNN,
            'fbcca_max': cls.FBCCA_MAX,
            'fbcca': cls.FBCCA_MAX,
        }
        return method_map.get(method.lower(), cls.MAX)


def compute_itr(
    accuracy: float, 
    n_classes: int, 
    trial_duration: float
) -> float:
    """
    Compute Information Transfer Rate (ITR) for BCI evaluation.
    
    ITR measures the amount of information communicated per unit time,
    accounting for both accuracy and number of choices.
    
    Formula: ITR = (log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))) * (60/T)
    
    Parameters
    ----------
    accuracy : float
        Classification accuracy (0 to 1)
    n_classes : int
        Number of classes/targets
    trial_duration : float
        Trial duration in seconds (including any gap/rest period)
        
    Returns
    -------
    itr : float
        Information transfer rate in bits per minute
        
    Notes
    -----
    - ITR assumes equal probability of all classes
    - Returns 0 if accuracy <= 1/n_classes (at or below chance)
    - Maximum ITR = log2(n_classes) * (60/trial_duration) when accuracy = 1
    
    References
    ----------
    [1] Wolpaw, J.R., et al. (2002). Brain-computer interfaces for 
        communication and control. Clinical Neurophysiology, 113(6), 767-791.
    """
    N = n_classes
    P = accuracy
    
    # Handle edge cases
    if P <= 1/N or P <= 0:
        return 0.0
    if P >= 1.0:
        P = 0.9999  # Avoid log(0)
    
    # ITR formula in bits per trial
    bits_per_trial = (
        np.log2(N) + 
        P * np.log2(P) + 
        (1 - P) * np.log2((1 - P) / (N - 1))
    )
    
    # Convert to bits per minute
    trials_per_minute = 60.0 / trial_duration
    itr = bits_per_trial * trials_per_minute
    
    return max(0.0, itr)


@dataclass
class EvaluationMetrics:
    """
    Comprehensive BCI evaluation metrics container.
    
    Stores accuracy, ITR, per-class metrics, and confusion matrix
    for easy comparison and export. Uses dataclass for cleaner code.
    
    Attributes
    ----------
    accuracy : float
        Overall classification accuracy
    itr : float
        Information Transfer Rate in bits/minute
    confusion_matrix : ndarray
        Confusion matrix of shape (n_classes, n_classes)
    f1_macro : float
        Macro-averaged F1 score
    f1_weighted : float
        Weighted-averaged F1 score
    precision_macro : float
        Macro-averaged precision
    recall_macro : float
        Macro-averaged recall
    precision_per_class : ndarray
        Per-class precision scores
    recall_per_class : ndarray
        Per-class recall scores
    f1_per_class : ndarray
        Per-class F1 scores
    """
    # Core metrics
    accuracy: float
    itr: float
    confusion_matrix: NDArray[np.int64]
    
    # Averaged metrics
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    
    # Per-class metrics
    precision_per_class: NDArray[np.float64]
    recall_per_class: NDArray[np.float64]
    f1_per_class: NDArray[np.float64]
    
    # Metadata
    n_classes: int
    trial_duration: float
    class_names: List[str]
    
    # Raw data (not included in repr)
    y_true: NDArray[np.int64] = field(repr=False)
    y_pred: NDArray[np.int64] = field(repr=False)
    
    @classmethod
    def from_predictions(
        cls,
        y_true: NDArray,
        y_pred: NDArray,
        n_classes: int,
        trial_duration: float = 4.0,
        class_names: Optional[List[str]] = None
    ) -> 'EvaluationMetrics':
        """
        Factory method to compute all metrics from predictions.
        
        Parameters
        ----------
        y_true : ndarray
            True labels
        y_pred : ndarray
            Predicted labels
        n_classes : int
            Number of classes
        trial_duration : float
            Trial duration in seconds for ITR calculation
        class_names : list of str, optional
            Names for each class (e.g., frequency labels)
            
        Returns
        -------
        metrics : EvaluationMetrics
            Computed evaluation metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = list(range(1, n_classes + 1))
        class_names = class_names or [str(i) for i in labels]
        
        # Compute all metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        itr = compute_itr(accuracy, n_classes, trial_duration)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        
        # Macro and weighted averages
        f1_macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
        
        return cls(
            accuracy=accuracy,
            itr=itr,
            confusion_matrix=conf_matrix,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            precision_per_class=precision_per_class,
            recall_per_class=recall_per_class,
            f1_per_class=f1_per_class,
            n_classes=n_classes,
            trial_duration=trial_duration,
            class_names=class_names,
            y_true=y_true,
            y_pred=y_pred
        )
    
    def summary(self) -> Dict[str, Any]:
        """Return a dictionary summary of all metrics."""
        return {
            'accuracy': self.accuracy,
            'itr_bits_per_min': self.itr,
            'f1_macro': self.f1_macro,
            'f1_weighted': self.f1_weighted,
            'precision_macro': self.precision_macro,
            'recall_macro': self.recall_macro,
            'per_class': {
                'precision': self.precision_per_class.tolist(),
                'recall': self.recall_per_class.tolist(),
                'f1': self.f1_per_class.tolist()
            }
        }
    
    def print_report(self, title: str = "Classification Report") -> None:
        """Print a formatted evaluation report."""
        print(f"\n{'=' * 60}")
        print(title)
        print('=' * 60)
        print(f"Accuracy:           {self.accuracy:.4f} ({self.accuracy * 100:.1f}%)")
        print(f"ITR:                {self.itr:.2f} bits/min")
        print(f"F1 Score (macro):   {self.f1_macro:.4f}")
        print(f"F1 Score (weighted):{self.f1_weighted:.4f}")
        print('-' * 60)
        print("Per-Class Metrics:")
        print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print('-' * 48)
        for i, name in enumerate(self.class_names):
            print(f"{name:<12} {self.precision_per_class[i]:<12.4f} "
                  f"{self.recall_per_class[i]:<12.4f} {self.f1_per_class[i]:<12.4f}")
        print('=' * 60)
    
    def to_dataframe(self):
        """Convert per-class metrics to pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            return pd.DataFrame({
                'Class': self.class_names,
                'Precision': self.precision_per_class,
                'Recall': self.recall_per_class,
                'F1-Score': self.f1_per_class
            })
        except ImportError:
            return None


@dataclass
class ClassifierConfig:
    """Configuration for EEGClassifier."""
    n_neighbors: int = 5
    trial_duration: float = 4.0
    
    # KNN parameters
    weights: str = 'uniform'  # 'uniform' or 'distance'
    metric: str = 'minkowski'


class EEGClassifier:
    """
    EEG classification using Max-CCA, Max-FBCCA, or K-Nearest Neighbors.
    
    Features:
    - Multiple classification strategies (max selection, KNN)
    - Comprehensive evaluation metrics (accuracy, ITR, F1, precision, recall)
    - Model persistence (save/load trained classifiers)
    - CCA spatial filter export
    
    Examples
    --------
    >>> classifier = EEGClassifier(n_neighbors=5, trial_duration=4.0)
    >>> accuracy, metrics = classifier.evaluate_with_metrics(
    ...     features_by_subject, labels_by_subject, stim_frequencies
    ... )
    
    >>> # Using config object
    >>> config = ClassifierConfig(n_neighbors=7, trial_duration=3.0)
    >>> classifier = EEGClassifier.from_config(config)
    """
    
    __slots__ = (
        '_n_neighbors', '_trial_duration', '_trained_model',
        '_model_config', '_spatial_filters'
    )

    def __init__(
        self, 
        n_neighbors: int = 5, 
        trial_duration: float = 4.0
    ) -> None:
        """
        Initialize the EEG classifier.

        Parameters
        ----------
        n_neighbors : int
            Number of neighbors for KNN classification
        trial_duration : float
            Trial duration in seconds for ITR calculation
        """
        self._n_neighbors = n_neighbors
        self._trial_duration = trial_duration
        self._trained_model: Optional[KNeighborsClassifier] = None
        self._model_config: Dict[str, Any] = {}
        self._spatial_filters: Dict[str, Dict[float, NDArray]] = {}
    
    @classmethod
    def from_config(cls, config: ClassifierConfig) -> 'EEGClassifier':
        """Create classifier from configuration object."""
        return cls(
            n_neighbors=config.n_neighbors,
            trial_duration=config.trial_duration
        )
    
    # -------------------- Properties --------------------
    
    @property
    def n_neighbors(self) -> int:
        """Number of KNN neighbors."""
        return self._n_neighbors
    
    @property
    def trial_duration(self) -> float:
        """Trial duration for ITR calculation."""
        return self._trial_duration
    
    @property
    def trained_model(self) -> Optional[KNeighborsClassifier]:
        """Trained KNN model (if any)."""
        return self._trained_model
    
    @property
    def model_config(self) -> Dict[str, Any]:
        """Model configuration dictionary."""
        return self._model_config
    
    @property
    def spatial_filters(self) -> Dict[str, Dict[float, NDArray]]:
        """Stored spatial filters."""
        return self._spatial_filters
    
    # -------------------- Classification Methods --------------------
    
    def classify_max_cca(
        self, 
        features_by_subject: Dict[str, NDArray], 
        labels_by_subject: Dict[str, NDArray], 
        stim_frequencies: List[float], 
        num_channels: int = 1, 
        num_harmonics: int = 1
    ) -> Tuple[float, NDArray]:
        """
        Classify by selecting stimulus frequency with highest CCA correlation.

        Parameters
        ----------
        features_by_subject : dict
            Subject-wise features (num_trials, num_freqs*num_channels*num_harmonics)
        labels_by_subject : dict
            Subject-wise labels (integers representing stimulus frequencies)
        stim_frequencies : list
            List of stimulus frequencies
        num_channels : int
            Number of EEG channels
        num_harmonics : int
            Number of harmonics per frequency

        Returns
        -------
        accuracy : float
            Classification accuracy
        conf_matrix : ndarray
            Confusion matrix
        """
        all_preds, all_labels = [], []
        num_freqs = len(stim_frequencies)

        for subject, features in features_by_subject.items():
            labels = labels_by_subject.get(subject)
            
            if labels is None or len(labels) == 0 or features.shape[0] == 0:
                print(f"Skipping subject {subject}: no valid data or labels.")
                continue

            print(f"Classifying subject {subject} with {features.shape[0]} trials.")

            # Reshape and find max correlation per frequency
            trial_features = features.reshape(features.shape[0], num_channels, num_freqs, num_harmonics)
            freq_corrs = trial_features.max(axis=(1, 3))  # (n_trials, n_freqs)
            preds = np.argmax(freq_corrs, axis=1) + 1  # Labels start from 1

            all_preds.extend(preds)
            all_labels.extend(labels.astype(int))

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(1, num_freqs + 1)))

        print(f"\nConfusion Matrix (Max-CCA):\n{conf_matrix}")
        print(f"Overall Max-CCA Classification Accuracy: {accuracy:.4f}")

        return accuracy, conf_matrix

    def classify_max_fbcca(
        self, 
        features_by_subject: Dict[str, NDArray], 
        labels_by_subject: Dict[str, NDArray], 
        stim_frequencies: List[float]
    ) -> Tuple[float, NDArray]:
        """
        Classify using FB-CCA features by selecting highest weighted correlation.
        
        Parameters
        ----------
        features_by_subject : dict
            Subject-wise FB-CCA features (num_trials, num_freqs)
        labels_by_subject : dict
            Subject-wise labels
        stim_frequencies : list
            List of stimulus frequencies

        Returns
        -------
        accuracy : float
            Classification accuracy
        conf_matrix : ndarray
            Confusion matrix
        """
        all_preds, all_labels = [], []
        num_freqs = len(stim_frequencies)

        for subject, features in features_by_subject.items():
            labels = labels_by_subject.get(subject)
            
            if labels is None or len(labels) == 0 or features.shape[0] == 0:
                print(f"Skipping subject {subject}: no valid data or labels.")
                continue

            print(f"Classifying subject {subject} with {features.shape[0]} trials using FB-CCA.")

            # Select frequency with maximum weighted correlation
            preds = np.argmax(features, axis=1) + 1
            all_preds.extend(preds)
            all_labels.extend(labels.astype(int))

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        accuracy = accuracy_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds, labels=list(range(1, num_freqs + 1)))

        print(f"\nConfusion Matrix (Max-FBCCA):\n{conf_matrix}")
        print(f"Overall Max-FBCCA Classification Accuracy: {accuracy:.4f}")

        return accuracy, conf_matrix

    def classify_knn(
        self, 
        features_by_subject: Dict[str, NDArray], 
        labels_by_subject: Dict[str, NDArray]
    ) -> Tuple[float, NDArray]:
        """
        Perform leave-subject-out cross-validation using KNN.

        Parameters
        ----------
        features_by_subject : dict
            Subject-wise feature arrays
        labels_by_subject : dict
            Subject-wise labels

        Returns
        -------
        avg_accuracy : float
            Average accuracy across subjects
        overall_conf_matrix : ndarray
            Aggregated confusion matrix
        """
        subjects = list(features_by_subject.keys())
        overall_conf_matrix = None
        accuracies = []

        for test_subject in subjects:
            print(f"Testing on subject {test_subject}, training on remaining subjects.")

            # Construct training data excluding test subject
            X_train = np.vstack([features_by_subject[s] for s in subjects if s != test_subject])
            y_train = np.concatenate([labels_by_subject[s] for s in subjects if s != test_subject])

            X_test = features_by_subject[test_subject]
            y_test = labels_by_subject[test_subject]

            # Train and predict
            knn = KNeighborsClassifier(n_neighbors=self._n_neighbors)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            print(f"Subject {test_subject} Accuracy: {accuracy:.4f}")
            print(f"Confusion Matrix:\n{conf_matrix}")

            accuracies.append(accuracy)
            overall_conf_matrix = conf_matrix if overall_conf_matrix is None else overall_conf_matrix + conf_matrix

        avg_accuracy = np.mean(accuracies)

        print(f"\nOverall Aggregated Confusion Matrix (KNN):\n{overall_conf_matrix}")
        print(f"Average Leave-Subject-Out KNN Accuracy: {avg_accuracy:.4f}")

        return avg_accuracy, overall_conf_matrix

    def compare_methods(
        self, 
        cca_features: Dict[str, NDArray], 
        fbcca_features: Dict[str, NDArray], 
        labels_by_subject: Dict[str, NDArray], 
        stim_frequencies: List[float]
    ) -> Dict[str, float]:
        """
        Compare classification performance between CCA and FB-CCA methods.
        
        Parameters
        ----------
        cca_features : dict
            Standard CCA features by subject
        fbcca_features : dict
            FB-CCA features by subject
        labels_by_subject : dict
            Labels by subject
        stim_frequencies : list
            Stimulus frequencies
        
        Returns
        -------
        results : dict
            Accuracy results for all methods
        """
        print('=' * 60)
        print("COMPARING CLASSIFICATION METHODS")
        print('=' * 60)
        
        results = {}
        
        # Standard CCA - Max selection
        print("\n--- Standard CCA (Max Selection) ---")
        results['cca_max'], _ = self.classify_max_cca(cca_features, labels_by_subject, stim_frequencies)
        
        # Standard CCA - KNN
        print("\n--- Standard CCA (KNN) ---")
        results['cca_knn'], _ = self.classify_knn(cca_features, labels_by_subject)
        
        # FB-CCA - Max selection
        print("\n--- FB-CCA (Max Selection) ---")
        results['fbcca_max'], _ = self.classify_max_fbcca(fbcca_features, labels_by_subject, stim_frequencies)
        
        # FB-CCA - KNN
        print("\n--- FB-CCA (KNN) ---")
        results['fbcca_knn'], _ = self.classify_knn(fbcca_features, labels_by_subject)
        
        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print('=' * 60)
        for name, acc in results.items():
            print(f"{name:<20} {acc:.4f}")
        print('=' * 60)
        
        return results

    # -------------------- Model Persistence --------------------
    
    def train_and_store(
        self, 
        features_by_subject: Dict[str, NDArray], 
        labels_by_subject: Dict[str, NDArray],
        stim_frequencies: List[float],
        method: str = 'knn'
    ) -> 'EEGClassifier':
        """
        Train a classifier on all data and store for later use.
        
        Parameters
        ----------
        features_by_subject : dict
            Dictionary of subject-wise features
        labels_by_subject : dict
            Dictionary of subject-wise labels
        stim_frequencies : list
            List of stimulus frequencies
        method : str
            Classification method ('knn' or 'max')
            
        Returns
        -------
        self : EEGClassifier
            Self with trained model stored
        """
        # Combine all subjects' data
        X_train = np.vstack(list(features_by_subject.values()))
        y_train = np.concatenate(list(labels_by_subject.values()))
        
        if method.lower() == 'knn':
            self._trained_model = KNeighborsClassifier(n_neighbors=self._n_neighbors)
            self._trained_model.fit(X_train, y_train)
        
        self._model_config = {
            'method': method,
            'n_neighbors': self._n_neighbors,
            'stim_frequencies': stim_frequencies,
            'n_classes': len(stim_frequencies),
            'feature_dim': X_train.shape[1],
            'n_training_samples': X_train.shape[0],
            'trial_duration': self._trial_duration
        }
        
        print(f"Model trained on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        return self
    
    def predict(
        self, 
        features: NDArray, 
        method: Optional[str] = None
    ) -> NDArray:
        """
        Predict class labels for new features.
        
        Parameters
        ----------
        features : ndarray
            Feature array, shape (n_trials, n_features)
        method : str, optional
            Override stored method ('knn' or 'max')
            
        Returns
        -------
        predictions : ndarray
            Predicted class labels
        """
        method = method or self._model_config.get('method', 'max')
        
        if method.lower() == 'knn':
            if self._trained_model is None:
                raise ValueError("No trained KNN model. Call train_and_store first.")
            return self._trained_model.predict(features)
        
        # Max selection
        return np.argmax(features, axis=1) + 1
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the trained classifier to disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to save the model (creates .pkl and .json files)
        """
        filepath = Path(filepath)
        
        # Save sklearn model
        model_path = filepath.with_suffix('.pkl')
        with model_path.open('wb') as f:
            pickle.dump({
                'model': self._trained_model,
                'n_neighbors': self._n_neighbors,
                'trial_duration': self._trial_duration
            }, f)
        
        # Save configuration as JSON
        config_path = filepath.with_suffix('.json')
        with config_path.open('w', encoding='utf-8') as f:
            json.dump(self._model_config, f, indent=2)
        
        # Save spatial filters if available
        if self._spatial_filters:
            filters_path = filepath.with_name(f'{filepath.stem}_spatial_filters.npz')
            flat_dict = {
                f'{ftype}_{freq}Hz': filt 
                for ftype, filters in self._spatial_filters.items() 
                for freq, filt in filters.items()
            }
            np.savez(filters_path, **flat_dict)
            print(f"Spatial filters saved to: {filters_path}")
        
        print(f"Model saved to: {model_path}")
        print(f"Config saved to: {config_path}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'EEGClassifier':
        """
        Load a trained classifier from disk.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model (without extension)
            
        Returns
        -------
        classifier : EEGClassifier
            Loaded classifier ready for prediction
        """
        filepath = Path(filepath)
        
        # Load sklearn model
        model_path = filepath.with_suffix('.pkl')
        with model_path.open('rb') as f:
            saved_data = pickle.load(f)
        
        # Load configuration
        config_path = filepath.with_suffix('.json')
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Create classifier instance
        classifier = cls(
            n_neighbors=saved_data.get('n_neighbors', 5),
            trial_duration=saved_data.get('trial_duration', 4.0)
        )
        classifier._trained_model = saved_data['model']
        classifier._model_config = config
        
        # Load spatial filters if available
        filters_path = filepath.with_name(f'{filepath.stem}_spatial_filters.npz')
        if filters_path.exists():
            classifier._spatial_filters = dict(np.load(filters_path))
            print(f"Spatial filters loaded from: {filters_path}")
        
        print(f"Model loaded from: {model_path}")
        return classifier
    
    def store_spatial_filters(
        self, 
        filters: Dict[float, NDArray],
        filter_type: str = 'cca'
    ) -> None:
        """
        Store CCA spatial filters for later export.
        
        Parameters
        ----------
        filters : dict
            Dictionary mapping frequencies to spatial filter arrays
        filter_type : str
            Type of filter ('cca', 'fbcca', etc.)
        """
        self._spatial_filters[filter_type] = filters
        print(f"Stored {len(filters)} spatial filters of type '{filter_type}'")
    
    def export_spatial_filters(
        self, 
        filepath: Union[str, Path], 
        output_format: str = 'npz'
    ) -> None:
        """
        Export stored spatial filters to file.
        
        Parameters
        ----------
        filepath : str or Path
            Output file path
        output_format : str
            Output format ('npz', 'mat', or 'csv')
        """
        filepath = Path(filepath)
        
        if output_format == 'npz':
            flat_dict = {
                f'{ftype}_{freq}Hz': filt 
                for ftype, filters in self._spatial_filters.items() 
                for freq, filt in filters.items()
            }
            np.savez(filepath.with_suffix('.npz'), **flat_dict)
            
        elif output_format == 'mat':
            try:
                from scipy.io import savemat
                savemat(filepath.with_suffix('.mat'), self._spatial_filters)
            except ImportError as exc:
                raise ImportError("scipy required for .mat export") from exc
                
        elif output_format == 'csv':
            for ftype, filters in self._spatial_filters.items():
                for freq, filt in filters.items():
                    csv_path = filepath.with_name(f'{filepath.stem}_{ftype}_{freq}Hz.csv')
                    np.savetxt(csv_path, filt, delimiter=',')
        
        print(f"Spatial filters exported to: {filepath}")

    # -------------------- Enhanced Evaluation --------------------
    
    def evaluate_with_metrics(
        self,
        features_by_subject: Dict[str, NDArray],
        labels_by_subject: Dict[str, NDArray],
        stim_frequencies: List[float],
        method: str = 'max',
        return_metrics: bool = True
    ) -> Tuple[float, Optional[EvaluationMetrics]]:
        """
        Evaluate classification with comprehensive metrics including ITR.
        
        Parameters
        ----------
        features_by_subject : dict
            Dictionary of subject-wise features
        labels_by_subject : dict
            Dictionary of subject-wise labels
        stim_frequencies : list
            List of stimulus frequencies
        method : str
            'max' for max selection, 'knn' for leave-subject-out KNN
        return_metrics : bool
            If True, return full EvaluationMetrics object
            
        Returns
        -------
        accuracy : float
            Overall accuracy
        metrics : EvaluationMetrics or None
            Full evaluation metrics (if return_metrics=True)
        """
        subjects = list(features_by_subject.keys())
        all_preds, all_labels = [], []
        n_classes = len(stim_frequencies)
        class_names = [f"{f}Hz" for f in stim_frequencies]
        
        if method.lower() == 'max':
            for subject in subjects:
                features = features_by_subject[subject]
                labels = labels_by_subject.get(subject)
                
                if labels is None or len(labels) == 0:
                    continue
                
                preds = np.argmax(features, axis=1) + 1
                all_preds.extend(preds)
                all_labels.extend(labels.astype(int))
                
        elif method.lower() == 'knn':
            for test_subject in subjects:
                X_train = np.vstack([features_by_subject[s] for s in subjects if s != test_subject])
                y_train = np.concatenate([labels_by_subject[s] for s in subjects if s != test_subject])
                
                X_test = features_by_subject[test_subject]
                y_test = labels_by_subject[test_subject]
                
                knn = KNeighborsClassifier(n_neighbors=self._n_neighbors)
                knn.fit(X_train, y_train)
                
                y_pred = knn.predict(X_test)
                all_preds.extend(y_pred)
                all_labels.extend(y_test.astype(int))
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Create comprehensive metrics
        metrics = EvaluationMetrics.from_predictions(
            y_true=all_labels,
            y_pred=all_preds,
            n_classes=n_classes,
            trial_duration=self._trial_duration,
            class_names=class_names
        )
        
        return (metrics.accuracy, metrics) if return_metrics else (metrics.accuracy, None)
    
    def compare_methods_full(
        self, 
        cca_features: Dict[str, NDArray], 
        fbcca_features: Dict[str, NDArray], 
        labels_by_subject: Dict[str, NDArray], 
        stim_frequencies: List[float],
        print_reports: bool = True
    ) -> Dict[str, EvaluationMetrics]:
        """
        Compare all methods with comprehensive evaluation metrics.
        
        Parameters
        ----------
        cca_features : dict
            Standard CCA features by subject
        fbcca_features : dict
            FB-CCA features by subject
        labels_by_subject : dict
            Labels by subject
        stim_frequencies : list
            Stimulus frequencies
        print_reports : bool
            Whether to print detailed reports
            
        Returns
        -------
        results : dict
            Dictionary mapping method names to EvaluationMetrics objects
        """
        methods = [
            ('cca_max', cca_features, 'max', "Standard CCA (Max Selection)"),
            ('cca_knn', cca_features, 'knn', "Standard CCA (KNN)"),
            ('fbcca_max', fbcca_features, 'max', "FB-CCA (Max Selection)"),
            ('fbcca_knn', fbcca_features, 'knn', "FB-CCA (KNN)")
        ]
        
        results = {}
        for name, features, method, title in methods:
            _, metrics = self.evaluate_with_metrics(
                features, labels_by_subject, stim_frequencies, method=method
            )
            results[name] = metrics
            if print_reports:
                metrics.print_report(title)
        
        # Print comparison summary
        if print_reports:
            print(f"\n{'=' * 70}")
            print("COMPREHENSIVE COMPARISON SUMMARY")
            print('=' * 70)
            print(f"{'Method':<20} {'Accuracy':<12} {'ITR (bpm)':<12} {'F1 (macro)':<12}")
            print('-' * 56)
            for name, m in results.items():
                print(f"{name:<20} {m.accuracy:<12.4f} {m.itr:<12.2f} {m.f1_macro:<12.4f}")
            print('=' * 70)
        
        return results
