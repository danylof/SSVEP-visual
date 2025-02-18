import os
import mne
from typing import List, Dict

class EEGDataset:
    """Handles EEG data loading, preprocessing, and concatenation from multiple subjects in a dataset folder."""

    # Global dictionary for non-EEG channel types.
    NON_EEG_CHANNELS = {
        'HEOL': 'eog',
        'HEOR': 'eog',
        'HEOR-L': 'eog',
        'VEOU': 'eog',
        'VEOL-U': 'eog',
        'EKG': 'ecg',
        'Trigger': 'stim'
    }

    def __init__(self, data_path: str):
        """
        Initialize the dataset manager.

        Parameters:
        - data_path (str): Path to the dataset containing subject folders.
        """
        self.data_path = data_path
        self.subjects = self._find_subjects()
        self.raw_data = {}  # Dictionary to store each subject's raw EEG data

    def _find_subjects(self) -> List[str]:
        """Find all subject folders in the dataset path."""
        return [f for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]

    def load_all_subjects(self):
        """Load and preprocess all subjects from the dataset folder."""
        for subject in self.subjects:
            raw = self.load_subject_data(subject)
            if raw:
                self.raw_data[subject] = raw
                print(f"Successfully loaded and preprocessed {subject}.")

    def load_subject_data(self, subject_id: str) -> mne.io.Raw:
        """
        Load and preprocess EEG data for a given subject.

        Parameters:
        - subject_id (str): Subject folder name.

        Returns:
        - raw (mne.io.Raw): Preprocessed raw EEG data.
        """
        subject_path = os.path.join(self.data_path, subject_id)
        files = [f for f in os.listdir(subject_path) if f.endswith('.cnt')]

        if not files:
            print(f"No CNT files found for subject {subject_id}.")
            return None

        file_paths = [os.path.join(subject_path, f) for f in files]
        raw = self.combine_raw_files(file_paths)
        return raw

    def load_and_preprocess_file(self, file_name: str) -> mne.io.Raw:
        """
        Load a CNT file and perform standard preprocessing:
          - Set channel types for non-EEG channels if they exist.
          - Apply a standard montage ('standard_1005').
          - Re-reference using A1 and A2, then drop these channels.

        Parameters:
        - file_name (str): Path to the CNT file.

        Returns:
        - raw (mne.io.Raw): Preprocessed raw EEG data.
        """
        print(f"Loading raw data from {file_name}...")
        raw = mne.io.read_raw_cnt(file_name, preload=True)
        print("Raw data loaded successfully!")
        
        # Filter NON_EEG_CHANNELS to only include channels that exist in raw data
        existing_channels = {ch: typ for ch, typ in self.NON_EEG_CHANNELS.items() if ch in raw.ch_names}
        raw.set_channel_types(existing_channels)
        print("Channel types adjusted for non-EEG channels.")

        # Apply a standard EEG montage
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, match_case=False, on_missing='ignore')
        print("Montage set for raw EEG data.")

        # Re-reference EEG data using A1 and A2
        raw.set_eeg_reference(ref_channels=['A1', 'A2'], projection=False)
        print("EEG data re-referenced using A1 and A2.")

        # Drop reference channels if they exist
        ref_channels = ['A1', 'A2']
        existing_refs = [ch for ch in ref_channels if ch in raw.ch_names]
        if existing_refs:
            raw.drop_channels(existing_refs)
            print(f"Reference channels {existing_refs} dropped. Remaining channels:")
            print(raw.info['ch_names'])
        else:
            print("No reference channels found to drop.")

        return raw


    def combine_raw_files(self, file_names: List[str]) -> mne.io.Raw:
        """
        Load and concatenate raw data from a list of CNT files.

        Parameters:
        - file_names (List[str]): List of paths to CNT files.

        Returns:
        - raw_combined (mne.io.Raw): Concatenated EEG data.
        """
        raw_list = [self.load_and_preprocess_file(f) for f in file_names]

        # Ensure calibration factors (_cals) match across files.
        common_cals = raw_list[0]._cals.copy()
        for raw in raw_list:
            raw._cals = common_cals

        raw_combined = mne.concatenate_raws(raw_list, preload=True)
        print("All files have been concatenated.")
        return raw_combined
