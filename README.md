# EEG Motor Imagery Classification

A pipeline that decodes imagined left-hand vs. right-hand movement 
from raw EEG signals, using the PhysioNet EEG Motor Movement/Imagery dataset.

## What this does

Given a raw EEG recording (64 channels, one subject performing a motor imagery 
task), this pipeline:

1. Loads and filters the signal to the 8–30 Hz band, where motor imagery signal 
   is known to live (mu/beta rhythms)
2. Segments the continuous recording into labeled epochs around each imagined 
   movement event
3. Extracts spatial features using Common Spatial Patterns (CSP)
4. Trains a logistic regression classifier on those features
5. Evaluates performance with 3-fold cross-validation

## Results

**Mean cross-validated accuracy: 73%** (chance level: 50%), on 15 labeled trials 
(8 left-hand, 7 right-hand imagery).

Fold scores: [0.6, 0.6, 1.0]

## Honest limitations

This is a small, single-subject, single-run dataset. 15 examples are not enough 
to draw strong conclusions, so the 1.0 fold score in particular should be read 
with caution given the small test size. The consistent above-chance performance 
across all three folds is the more meaningful signal here: it suggests CSP is 
picking up a real, decodable difference rather than noise, but this would need 
more subjects and runs to validate rigorously.

## Why I built this

Building technical fluency in EEG/BCI signal processing using the same tools 
and methods (MNE-Python, CSP, cross-validated evaluation) that are standard 
in actual BCI research and industry work.

## Stack

Python, MNE-Python, scikit-learn, NumPy, SciPy, Matplotlib

## Running it

```
pip install mne numpy scipy matplotlib scikit-learn
python3 classify.py
```
