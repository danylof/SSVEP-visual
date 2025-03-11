import os
import mne
from typing import List

class EEGDataset:
    """Handles EEG data loading, preprocessing, and concatenation from multiple subjects located within a dataset directory."""

    # Mapping for expected non-EEG channels and their correct types.
    NON_EEG_CHANNELS = {
        'HEOL': 'eog',       # Horizontal EOG left
        'HEOR': 'eog',       # Horizontal EOG right
        'HEOR-L': 'eog',     # Alternative horizontal EOG right-left
        'VEOU': 'eog',       # Vertical EOG upper
        'VEOL-U': 'eog',     # Alternative vertical EOG lower-upper
        'EKG': 'ecg',        # Electrocardiogram
        'Trigger': 'stim'    # Stimulus trigger
    }

    # EEG channels specifically targeted for analysis.
    CHANNELS_OF_INTEREST = ['O1', 'Oz', 'OZ', 'O2']

    def __init__(self, data_path: str):
        """
        Initialize EEGDataset with the path to dataset.

        Parameters:
        - data_path (str): Path to the root dataset directory containing subject-specific subdirectories.
        """
        self.data_path = data_path
        self.subjects = self._find_subjects()
        self.raw_data = {}  # Dictionary mapping subject IDs to their preprocessed raw EEG data.

    def _find_subjects(self) -> List[str]:
        """Discover and list all subject directories within the dataset path."""
        return [subj for subj in os.listdir(self.data_path)
                if os.path.isdir(os.path.join(self.data_path, subj))]

    def load_all_subjects(self):
        """Iteratively load and preprocess EEG data for all subjects in the dataset."""
        for subject_id in self.subjects:
            raw_subject_data = self.load_subject_data(subject_id)
            if raw_subject_data:
                self.raw_data[subject_id] = raw_subject_data
                print(f"Successfully loaded and preprocessed data for subject: {subject_id}")

    def load_subject_data(self, subject_id: str) -> mne.io.Raw:
        """
        Load and preprocess EEG data for a specific subject.

        Parameters:
        - subject_id (str): Identifier corresponding to the subject's data directory.

        Returns:
        - raw (mne.io.Raw): Preprocessed EEG data object or None if no valid data found.
        """
        subject_dir = os.path.join(self.data_path, subject_id)
        cnt_files = [file for file in os.listdir(subject_dir) if file.endswith('.cnt')]

        if not cnt_files:
            print(f"No CNT files found for subject {subject_id}.")
            return None

        file_paths = [os.path.join(subject_dir, file) for file in cnt_files]
        combined_raw = self.combine_raw_files(file_paths)

        # Retain only specified EEG channels prior to further processing.
        valid_channels = [ch for ch in self.CHANNELS_OF_INTEREST if ch in combined_raw.ch_names]
        combined_raw.pick_channels(valid_channels)
        print(f"Selected channels for subject {subject_id}: {combined_raw.ch_names}")

        return combined_raw

    def load_and_preprocess_file(self, file_path: str) -> mne.io.Raw:
        """
        Load and preprocess an individual CNT file, performing standard EEG preprocessing steps:
        - Assign correct types to non-EEG channels.
        - Apply EEG montage used during recording (standard_1005).
        - Re-reference EEG data using reference channels (A1, A2), then drop them.

        Parameters:
        - file_path (str): Full path to the CNT file.

        Returns:
        - raw (mne.io.Raw): Preprocessed EEG data.
        """
        print(f"Loading raw EEG data from {file_path}...")
        raw = mne.io.read_raw_cnt(file_path, preload=True)
        print("Raw EEG data loaded successfully.")

        # Set types for existing non-EEG channels only.
        existing_non_eeg_channels = {ch: typ for ch, typ in self.NON_EEG_CHANNELS.items() if ch in raw.ch_names}
        raw.set_channel_types(existing_non_eeg_channels)
        print("Non-EEG channel types assigned correctly.")

        # Apply the standard montage for spatial referencing.
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        print("Standard EEG montage (standard_1005) applied.")

        # Re-reference EEG signals to channels A1 and A2.
        raw.set_eeg_reference(ref_channels=['A1', 'A2'], projection=False)
        print("EEG data re-referenced to channels A1 and A2.")

        # Drop reference channels if present in the data.
        ref_channels_to_drop = [ch for ch in ['A1', 'A2'] if ch in raw.ch_names]
        if ref_channels_to_drop:
            raw.drop_channels(ref_channels_to_drop)
            print(f"Dropped reference channels: {ref_channels_to_drop}. Remaining channels:")
            print(raw.info['ch_names'])
        else:
            print("No reference channels found to drop.")

        return raw

    def combine_raw_files(self, file_paths: List[str]) -> mne.io.Raw:
        """
        Load, preprocess, and concatenate EEG data from multiple CNT files.

        Parameters:
        - file_paths (List[str]): List containing paths to CNT files for a single subject.

        Returns:
        - raw_combined (mne.io.Raw): Concatenated raw EEG data from all provided files.
        """
        # Load and preprocess each individual CNT file.
        raw_files = [self.load_and_preprocess_file(path) for path in file_paths]

        # Ensure calibration factors match across all files for consistency.
        common_calibration = raw_files[0]._cals.copy()
        for raw in raw_files:
            raw._cals = common_calibration

        # Concatenate all preprocessed raw EEG files into a single data object.
        raw_combined = mne.concatenate_raws(raw_files, preload=True)
        print("Successfully concatenated EEG data from multiple CNT files.")

        return raw_combined