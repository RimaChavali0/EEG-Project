import mne

# Download and load one subject's EEG recording
raw = mne.datasets.eegbci.load_data(subjects=1, runs=[4])
raw_data = mne.io.read_raw_edf(raw[0], preload=True)

print(raw_data.info)
raw_data.plot(duration=5, n_channels=10, block=True)