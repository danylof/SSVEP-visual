import mne
import numpy as np
import json
from typing import List, Tuple, Dict


class EEGPreprocessor:
    """Handles EEG preprocessing including resampling, filtering, epoching, and standardizing event IDs across subjects."""

    def __init__(self, sfreq: int = 256, l_freq: float = None, h_freq: float = None, 
                 event_mapping_path: str = "event_mapping.json", invalid_keys: List[str] = None):
        """
        Initialize EEG preprocessing parameters.

        Parameters:
        - sfreq (int): Target sampling frequency for EEG data resampling (default: 256 Hz).
        - l_freq (float, optional): Low cutoff frequency for bandpass filtering.
        - h_freq (float, optional): High cutoff frequency for bandpass filtering.
        - event_mapping_path (str): File path for storing global event ID mappings.
        - invalid_keys (List[str], optional): Event keys to exclude from analysis and standardization.
        """
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.event_mapping_path = event_mapping_path
        self.invalid_keys = invalid_keys if invalid_keys else ['100', '99', '36']
        self.standard_event_map = self.load_event_mapping()

    def load_event_mapping(self) -> Dict[str, int]:
        """Load standardized event mappings from a JSON file or initialize an empty mapping."""
        try:
            with open(self.event_mapping_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_event_mapping(self):
        """Save the standardized event mappings to a JSON file for future use."""
        with open(self.event_mapping_path, "w") as f:
            json.dump(self.standard_event_map, f, indent=4)

    def resample_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Resample EEG data to ensure consistent sampling frequency across all datasets."""
        if raw.info['sfreq'] != self.sfreq:
            raw_resampled = raw.copy().resample(self.sfreq, npad="auto")
            print(f"Data resampled to {self.sfreq} Hz.")
            return raw_resampled
        return raw

    def filter_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply a bandpass filter to remove unwanted frequency components from EEG signals."""
        raw_filtered = raw.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin', verbose=False)
        print(f"Applied bandpass filter from {self.l_freq} Hz to {self.h_freq} Hz.")
        return raw_filtered

    def get_event_mapping(self, raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extract event information and mapping from EEG annotations.

        Parameters:
        - raw (mne.io.Raw): Raw EEG data.

        Returns:
        - events (np.ndarray): Array containing event timestamps and IDs.
        - event_id (Dict[str, int]): Mapping of event names to numeric IDs.
        """
        events, event_id = mne.events_from_annotations(raw)

        # Retain only event IDs present in the data
        unique_event_codes = set(events[:, 2])
        filtered_event_id = {key: value for key, value in event_id.items() if value in unique_event_codes}

        print(f"Extracted {len(events)} events. Filtered event IDs: {filtered_event_id}")
        return events, filtered_event_id

    def standardize_event_ids(self, events: np.ndarray, event_id: Dict[str, int]) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Standardize event IDs across datasets, removing invalid or irrelevant keys.

        Parameters:
        - events (np.ndarray): Original events array.
        - event_id (Dict[str, int]): Original event-to-ID mapping.

        Returns:
        - standardized_events (np.ndarray): Events with standardized numeric IDs.
        - standardized_event_id (Dict[str, int]): Global standardized event mapping.
        """
        # Filter out invalid event keys
        valid_event_id = {key: val for key, val in event_id.items() if key not in self.invalid_keys}

        # Initialize global standardized event map if empty
        if not self.standard_event_map:
            self.standard_event_map = {key: idx + 1 for idx, key in enumerate(sorted(valid_event_id.keys()))}

        # Update events with standardized IDs
        standardized_events = events.copy()
        standardized_event_id = {key: self.standard_event_map[key] for key in valid_event_id.keys()}

        for i, event in enumerate(standardized_events):
            event_code = event[2]
            matching_keys = [key for key, val in event_id.items() if val == event_code]

            if matching_keys and matching_keys[0] in standardized_event_id:
                standardized_events[i, 2] = standardized_event_id[matching_keys[0]]

        # Persist updated global event mappings
        self.save_event_mapping()
        return standardized_events, standardized_event_id

    def create_epochs_from_raw(self, raw: mne.io.Raw, tmin: float = 0.5, tmax: float = 4.5,
                               valid_keys: List[str] = None) -> mne.Epochs:
        """
        Generate epochs from EEG data based on standardized events.

        Parameters:
        - raw (mne.io.Raw): Raw EEG data.
        - tmin (float): Epoch start time (in seconds).
        - tmax (float): Epoch end time (in seconds).
        - valid_keys (List[str], optional): Specific event keys to include in epochs.
        - channels_of_interest (List[str], optional): Specific EEG channels to retain.

        Returns:
        - epochs (mne.Epochs): EEG data segmented into epochs.
        """
        raw_resampled = self.resample_data(raw)

        events, event_id = self.get_event_mapping(raw_resampled)
        standardized_events, event_id = self.standardize_event_ids(events, event_id)

        if valid_keys is None:
            valid_keys = list(event_id.keys())

        valid_codes = [event_id[k] for k in valid_keys if k in event_id]

        trial_events = standardized_events[np.isin(standardized_events[:, 2], valid_codes)]

        if len(trial_events) == 0:
            print("No valid events found. Returning None.")
            return None

        epochs = mne.Epochs(raw_resampled, events=trial_events, event_id={key: event_id[key] for key in valid_keys if key in event_id},
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)

        return epochs
