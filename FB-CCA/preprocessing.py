import mne
import numpy as np
from typing import List, Tuple, Dict


class EEGPreprocessor:
    """Handles EEG preprocessing including filtering and epoching."""

    def __init__(self, sfreq: int, l_freq: float = None, h_freq: float = None):
        """
        Parameters:
        - sfreq (int): Sampling frequency of the EEG data.
        - l_freq (float, optional): Low cutoff frequency for filtering.
        - h_freq (float, optional): High cutoff frequency for filtering.
        """
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq

    def filter_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply bandpass filter to the raw data."""
        raw_filtered = raw.copy().filter(l_freq=self.l_freq, h_freq=self.h_freq, fir_design='firwin', verbose=False)
        print("Data filtered with bandpass:", self.l_freq, "-", self.h_freq, "Hz")
        return raw_filtered

    def get_event_mapping(self, raw: mne.io.Raw) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extract events and event ID mappings from raw EEG data annotations.

        Parameters:
        - raw (mne.io.Raw): EEG data.

        Returns:
        - events (np.ndarray): Extracted event array.
        - event_id (Dict[str, int]): Dictionary mapping event names to event codes.
        """
        events, event_id = mne.events_from_annotations(raw)

        # Keep only event IDs that actually have events in the data
        unique_event_codes = set(events[:, 2])  # Get event codes that exist in the data
        filtered_event_id = {key: value for key, value in event_id.items() if value in unique_event_codes}

        print(f"Extracted {len(events)} events. Filtered Event IDs:", filtered_event_id)
        return events, filtered_event_id

    def create_epochs_from_raw(self, raw: mne.io.Raw, tmin: float = 0.5, tmax: float = 4.5,
                               valid_keys: List[str] = None, channels_of_interest: List[str] = None) -> mne.Epochs:
        """
        Extract events from raw data and create epochs.

        Parameters:
        - raw (mne.io.Raw): Raw EEG data.
        - tmin (float): Start time for each epoch.
        - tmax (float): End time for each epoch.
        - valid_keys (List[str], optional): List of valid event annotation keys.
        - channels_of_interest (List[str], optional): List of channels to keep for analysis.

        Returns:
        - epochs (mne.Epochs): Processed epochs data.
        """
        events, event_id = self.get_event_mapping(raw)

        print("Events extracted from annotations:")
        print(events)
        print("Event IDs mapping:")
        print(event_id)

        # If no valid_keys are provided, use all keys except non-stimulation ones (99 and 100)
        if valid_keys is None:
            valid_keys = [k for k in event_id.keys() if k not in ['100', '99']]

        # Convert valid keys to event codes, keeping only existing events
        valid_codes = [event_id[k] for k in valid_keys if k in event_id]
        print("Using valid event codes:", valid_codes)

        # Filter events to only include valid ones
        trial_events = events[np.isin(events[:, 2], valid_codes)]
        print("Filtered trial events:")
        print(trial_events)

        # If there are no valid events, return an empty Epochs object
        if len(trial_events) == 0:
            print("No valid events found. Returning empty Epochs object.")
            return None

        # Create epochs
        epochs = mne.Epochs(raw, events=trial_events, event_id={key: event_id[key] for key in valid_keys if key in event_id},
                            tmin=tmin, tmax=tmax, baseline=None, preload=True)

        print(f"Epochs created with {tmax - tmin} seconds duration for each trial.")

        # Pick only channels of interest
        if channels_of_interest:
            existing_channels = [ch for ch in channels_of_interest if ch in epochs.ch_names]
            epochs.pick_channels(existing_channels)
            print("Selected channels for analysis:", epochs.ch_names)

        return epochs
