"""
EEG dataset loading module for SSVEP analysis.

Handles CNT file loading, channel selection, and preprocessing with:
- Pathlib for robust cross-platform path handling
- Complete type hints
- Class constants using ClassVar
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import mne


class EEGDataset:
    """
    Handles EEG data loading, preprocessing, and concatenation from CNT files.
    
    Designed for multi-subject SSVEP datasets where each subject has
    one or more CNT recording files in their own subdirectory.
    
    Examples
    --------
    >>> dataset = EEGDataset(Path("./data"))
    >>> dataset.load_all_subjects()
    >>> raw = dataset.raw_data['subject01']
    """
    
    # Class-level constants for channel configuration
    NON_EEG_CHANNELS: ClassVar[Dict[str, str]] = {
        'HEOL': 'eog',       # Horizontal EOG left
        'HEOR': 'eog',       # Horizontal EOG right
        'HEOR-L': 'eog',     # Alternative horizontal EOG right-left
        'VEOU': 'eog',       # Vertical EOG upper
        'VEOL-U': 'eog',     # Alternative vertical EOG lower-upper
        'EKG': 'ecg',        # Electrocardiogram
        'Trigger': 'stim'    # Stimulus trigger
    }
    
    # EEG channels targeted for SSVEP analysis (occipital region)
    CHANNELS_OF_INTEREST: ClassVar[List[str]] = ['O1', 'Oz', 'OZ', 'O2']
    
    __slots__ = ('_data_path', '_subjects', '_raw_data')

    def __init__(self, data_path: str | Path) -> None:
        """
        Initialize EEGDataset with path to dataset.

        Parameters
        ----------
        data_path : str or Path
            Path to root dataset directory containing subject subdirectories
        """
        self._data_path = Path(data_path)
        self._subjects = self._find_subjects()
        self._raw_data: Dict[str, mne.io.Raw] = {}
    
    # -------------------- Properties --------------------
    
    @property
    def data_path(self) -> Path:
        """Dataset root path."""
        return self._data_path
    
    @property
    def subjects(self) -> List[str]:
        """List of discovered subject IDs."""
        return self._subjects
    
    @property
    def raw_data(self) -> Dict[str, mne.io.Raw]:
        """Dictionary mapping subject IDs to preprocessed raw EEG data."""
        return self._raw_data
    
    # -------------------- Subject Discovery --------------------
    
    def _find_subjects(self) -> List[str]:
        """Discover all subject directories within the dataset path."""
        return sorted([
            p.name for p in self._data_path.iterdir() 
            if p.is_dir()
        ])
    
    # -------------------- Data Loading --------------------

    def load_all_subjects(self) -> None:
        """Load and preprocess EEG data for all subjects in the dataset."""
        for subject_id in self._subjects:
            if raw := self.load_subject_data(subject_id):
                self._raw_data[subject_id] = raw
                print(f"Successfully loaded data for subject: {subject_id}")

    def load_subject_data(self, subject_id: str) -> Optional[mne.io.Raw]:
        """
        Load and preprocess EEG data for a specific subject.

        Parameters
        ----------
        subject_id : str
            Identifier for the subject's data directory

        Returns
        -------
        raw : mne.io.Raw or None
            Preprocessed EEG data, or None if no valid data found
        """
        subject_dir = self._data_path / subject_id
        cnt_files = list(subject_dir.glob('*.cnt'))

        if not cnt_files:
            print(f"No CNT files found for subject {subject_id}.")
            return None

        combined_raw = self.combine_raw_files(cnt_files)

        # Retain only channels of interest
        valid_channels = [
            ch for ch in self.CHANNELS_OF_INTEREST 
            if ch in combined_raw.ch_names
        ]
        combined_raw.pick_channels(valid_channels)
        print(f"Selected channels for {subject_id}: {combined_raw.ch_names}")

        return combined_raw

    def load_and_preprocess_file(self, file_path: Path) -> mne.io.Raw:
        """
        Load and preprocess an individual CNT file.
        
        Preprocessing steps:
        - Assign correct types to non-EEG channels
        - Apply EEG montage (standard_1005)
        - Re-reference to A1/A2, then drop reference channels

        Parameters
        ----------
        file_path : Path
            Path to the CNT file

        Returns
        -------
        raw : mne.io.Raw
            Preprocessed EEG data
        """
        print(f"Loading raw EEG data from {file_path}...")
        raw = mne.io.read_raw_cnt(str(file_path), preload=True)
        print("Raw EEG data loaded successfully.")

        # Set types for existing non-EEG channels only
        existing_non_eeg = {
            ch: typ for ch, typ in self.NON_EEG_CHANNELS.items() 
            if ch in raw.ch_names
        }
        if existing_non_eeg:
            raw.set_channel_types(existing_non_eeg)
            print("Non-EEG channel types assigned correctly.")

        # Apply standard montage
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        print("Standard EEG montage (standard_1005) applied.")

        # Re-reference to A1 and A2
        raw.set_eeg_reference(ref_channels=['A1', 'A2'], projection=False)
        print("EEG data re-referenced to A1 and A2.")

        # Drop reference channels if present
        ref_to_drop = [ch for ch in ('A1', 'A2') if ch in raw.ch_names]
        if ref_to_drop:
            raw.drop_channels(ref_to_drop)
            print(f"Dropped reference channels: {ref_to_drop}")
            print(f"Remaining channels: {raw.info['ch_names']}")

        return raw

    def combine_raw_files(self, file_paths: List[Path]) -> mne.io.Raw:
        """
        Load, preprocess, and concatenate EEG data from multiple CNT files.

        Parameters
        ----------
        file_paths : list of Path
            Paths to CNT files for a single subject

        Returns
        -------
        raw_combined : mne.io.Raw
            Concatenated raw EEG data
        """
        # Load and preprocess each file
        raw_files = [self.load_and_preprocess_file(path) for path in file_paths]

        # Ensure calibration factors match
        common_calibration = raw_files[0]._cals.copy()
        for raw in raw_files:
            raw._cals = common_calibration

        # Concatenate all files
        raw_combined = mne.concatenate_raws(raw_files, preload=True)
        print(f"Concatenated {len(file_paths)} CNT files.")

        return raw_combined
