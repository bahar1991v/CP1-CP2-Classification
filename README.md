TESS CP1–CP2 Classification

Python implementation of a machine-learning pipeline for the classification of chemically peculiar stars using TESS photometric light curves and XGBoost.

The project focuses on distinguishing between:

* CP1 (Am) stars
* CP2 (Ap) stars

Pipeline

The workflow consists of two main stages:

1. Feature Extraction
    * Statistical features
    * Autocorrelation (ACF)
    * Fast Fourier Transform (FFT)
    * Wavelet decomposition
    * 40 photometric features in total
2. XGBoost Classification
    * TIC-based 90/10 training-validation split
    * 100 independent iterations
    * Standardized features
    * Accuracy, Precision, Recall, F1, TSS and HSS
    * External prediction on LAMOST-associated data
    * CP1/CP2 classification using an 85% confidence threshold

Files

feature_extraction.py
xgboost_classification.py

Requirements

pip install numpy pandas scipy statsmodels PyWavelets scikit-learn xgboost openpyxl

Data

The observational datasets are not included in this repository.
The scripts expect the input data to be organized according to the paths defined in the code.

Author

Bahar Vasiq
PhD Researcher in Astrophysics
