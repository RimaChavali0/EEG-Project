import mne

# Load the same data as before
raw = mne.datasets.eegbci.load_data(subjects=1, runs=[4])
raw_data = mne.io.read_raw_edf(raw[0], preload=True)

# Find the labeled events in the recording (rest, left hand, right hand)
events, event_id = mne.events_from_annotations(raw_data)
print("Event labels found:", event_id)

# Filter the signal to the frequency range where motor imagery shows up (8-30 Hz)
raw_filtered = raw_data.copy().filter(l_freq=8, h_freq=30)

# Cut the continuous recording into short clips centered on each event
# tmin/tmax define the time window around each event marker (in seconds)
epochs = mne.Epochs(raw_filtered, events, event_id=event_id,
                     tmin=-1, tmax=4, baseline=None, preload=True)

print(epochs)

# Keep only left-hand (T1) and right-hand (T2) trials — drop rest (T0)
epochs_motor = epochs['T1', 'T2']
print(epochs_motor)

# Get the actual signal data as a NumPy array, and the labels
X = epochs_motor.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_motor.events[:, -1]  # the event code for each epoch

print("Data shape:", X.shape)
print("Labels:", y)