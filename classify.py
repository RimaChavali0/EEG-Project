import mne
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import CSP

# Reload everything (keeps this file self-contained and runnable on its own)
raw = mne.datasets.eegbci.load_data(subjects=1, runs=[4])
raw_data = mne.io.read_raw_edf(raw[0], preload=True)
raw_filtered = raw_data.copy().filter(l_freq=8, h_freq=30)

events, event_id = mne.events_from_annotations(raw_data)
epochs = mne.Epochs(raw_filtered, events, event_id=event_id,
                     tmin=-1, tmax=4, baseline=None, preload=True)
epochs_motor = epochs['T1', 'T2']

X = epochs_motor.get_data()
y = epochs_motor.events[:, -1]

# CSP finds spatial patterns (combinations of electrodes) that best separate
# the two classes — this is the standard feature extraction method for motor imagery
csp = CSP(n_components=4, reg=None, log=True)

# A pipeline chains steps together: extract CSP features, scale them,
# then classify with logistic regression
clf = Pipeline([
    ('CSP', csp),
    ('Scaler', StandardScaler()),
    ('LogisticRegression', LogisticRegression())
])

# Cross-validation: since you only have 15 examples, this trains/tests
# on different small splits of the data multiple times and averages the result,
# which is more honest than a single train/test split on this little data
scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')

print("Individual fold scores:", scores)
print(f"Mean accuracy: {scores.mean():.2f} (chance level is 0.50)")