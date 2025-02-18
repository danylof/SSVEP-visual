import mne

def get_event_mapping(raw):
    """Retrieve event dictionary from raw annotations."""
    events, event_id = mne.events_from_annotations(raw)
    return events, event_id