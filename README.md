Final Training Catalog

The Training_Catalog_Final file contains the initial catalog of CP1 and CP2 stars selected for preparing the training dataset used in the XGBoost classification pipeline. This catalog serves as the reference list for identifying and organizing the labeled stellar samples used for model training and validation.


1. Feature Extraction from TESS Light Curves

Overview

This module extracts numerical features from detrended TESS light curves of chemically peculiar (CP) stars. The extracted features describe the statistical properties, temporal correlations, frequency-domain characteristics, and multiscale structure of the photometric signals.

The extracted features are saved in a CSV file for subsequent machine-learning classification.

Input Data

The input consists of NPZ files containing detrended photometric time series. The pipeline expects the following directory structure:

data/
└── augmented_npz/
    ├── CP1/
    │   ├── TIC123456789_sector_1.npz
    │   └── ...
    └── CP2/
        ├── TIC987654321_sector_2.npz
        └── ...

Each NPZ file must contain an array named magnitude_detrended.

Both original and augmented samples can be processed. Files containing _aug in their filenames are identified as augmented samples.

Extracted Features

The pipeline extracts up to 40 features from each light curve, organized into four groups:

1. Statistical Features

Nine statistical descriptors characterize the distribution of photometric measurements:

* Standard deviation (std_mag)
* Minimum (min_mag)
* Maximum (max_mag)
* Median (median_mag)
* 25th percentile (q25_mag)
* 75th percentile (q75_mag)
* Interquartile range (iqr_mag)
* Skewness (skew_mag)
* Kurtosis (kurtosis_mag)

2. Autocorrelation Features

Five autocorrelation coefficients (acf_1 to acf_5) characterize temporal dependencies in the photometric measurements.

3. Fourier Transform Features

The magnitudes of the first ten coefficients of the Fast Fourier Transform (FFT), normalized by the number of measurements, are extracted as fft_0 to fft_9.

4. Wavelet Features

A discrete wavelet decomposition using the Daubechies-4 (db4) wavelet is performed at level 3.

For each of the four coefficient arrays (approximation coefficients a3 and detail coefficients d3, d2, and d1), four statistical descriptors are calculated:

* Mean
* Standard deviation
* Maximum
* Minimum

This produces 16 wavelet-based features.

Data Processing

The pipeline performs the following operations:

1. Reads NPZ files from the CP1 and CP2 directories.
2. Extracts the TIC identifier and TESS sector from each filename.
3. Identifies augmented samples using the filename.
4. Removes NaN values from the photometric time series.
5. Skips light curves containing fewer than ten valid measurements.
6. Calculates statistical, autocorrelation, FFT, and wavelet features.
7. Assigns numerical class labels:
    * 0: CP1
    * 1: CP2
8. Removes constant or all-zero feature columns.
9. Saves the resulting feature table as a CSV file.

If wavelet decomposition fails, the implementation substitutes zeros for the 16 wavelet features.

Output

The extracted features are saved to:

results/features1_augmented.csv

The output table contains sample metadata, including TIC identifiers, TESS sectors, class labels, augmentation status, and filenames, together with the retained numerical features.

The resulting dataset is used as input to the XGBoost classification pipeline.

2. XGBoost Classification and External Validation

Overview

This module implements an XGBoost-based binary classification pipeline to distinguish between CP1 and CP2 stars using features extracted from TESS photometric light curves.

The pipeline evaluates classification performance through repeated random train-validation splits and applies the trained models to an external dataset of stars with unknown classes.

Input Data

The classification pipeline requires two CSV files:

1. Training Dataset

data/features/features1_augmented.csv

This file contains the extracted photometric features and the corresponding class labels.

The class encoding is:

* 0: CP1
* 1: CP2

Metadata columns, including tic, sector, is_augmented, and filename, are excluded from the model input.

2. External Dataset

data/external/fall_features_lamost14686_matched.csv

This file contains the photometric features of the external stellar sample.

The external dataset must contain the same feature columns used to train the model.

Data Preparation

Before model training, the pipeline:

1. Loads the training and external datasets.
2. Identifies the TIC identifiers and TESS sectors in the external dataset.
3. Checks that each training TIC identifier belongs to only one class.
4. Identifies overlapping TIC identifiers between the training and external datasets.
5. Removes overlapping TICs from the external dataset to reduce data leakage.
6. Verifies that all required training features are available in the external dataset.

A StandardScaler is fitted to the training data and then applied to the validation and external datasets.

Model Configuration

The classifier is implemented using XGBClassifier with the following parameters:

Parameter	Value
n_estimators	400
max_depth	6
learning_rate	0.05
subsample	0.8
colsample_bytree	0.8
objective	binary:logistic
eval_metric	logloss

The scale_pos_weight parameter is calculated from the class distribution in each training split as the ratio of the number of CP1 samples to the number of CP2 samples.

Repeated Training and Validation

The classification procedure is repeated 100 times.

In each iteration:

1. The unique TIC identifiers are randomly shuffled separately for CP1 and CP2.
2. Approximately 90% of the TIC identifiers from each class are selected for training.
3. The remaining approximately 10% are reserved for internal validation.
4. The XGBoost classifier is trained on the selected training samples.
5. Predictions are generated for the held-out validation samples.
6. Classification performance metrics are calculated.
7. The trained model estimates the CP2 probabilities of the external samples.

Splitting the data by TIC identifier ensures that observations belonging to the same TIC are assigned to the same partition.

The external dataset remains fixed across the 100 iterations, while the training-validation split changes in each iteration.

Evaluation Metrics

The following metrics are calculated for internal validation:

* Accuracy
* Precision for CP1 and CP2
* Recall for CP1 and CP2
* F1-score for CP1 and CP2
* True Skill Statistic (TSS)
* Heidke Skill Score (HSS)

The confusion-matrix counts (TP, TN, FP, and FN) are also recorded.

The mean and standard deviation of the validation metrics across the 100 iterations are reported to summarize the variability in classification performance.

External Classification

For each iteration, the model calculates the probability that each external star belongs to CP2.

The CP2 probabilities from all 100 iterations are then combined for each TIC. The following statistics are calculated:

* Mean CP2 probability
* Standard deviation of CP2 probability
* Minimum and maximum CP2 probabilities
* Number of iterations in which the star was evaluated

The mean CP2 probability is used to assign the final classification with a confidence threshold of 85%.

The classification rules are:

* CP2: Mean CP2 probability ≥ 0.85
* CP1: Mean CP2 probability ≤ 0.15
* Uncertain: Mean CP2 probability is between 0.15 and 0.85

Output Files

The pipeline generates three principal output files:

File	Description
100_iterations_external_probabilities.csv	CP2 probabilities for individual external samples across all iterations
100_iterations_validation_metrics.csv	Validation metrics calculated during each iteration
TIC_probability_mean_std_100_iterations.xlsx	Per-TIC probability summaries and final classifications

All output files are saved in:

results/xgb_external/

Purpose

This pipeline provides a reproducible framework for evaluating the photometric classification of CP stars and applying the trained classifier to an external sample with unknown class labels.

3. Spectroscopic Classification Using Gaussian Probability Distributions

Overview

This module implements a rule-based spectroscopic classification procedure for chemically peculiar stars using equivalent-width (EW) measurements.

The method compares the mean equivalent widths associated with CP1 and CP2 spectral-line groups and applies Gaussian cumulative distribution functions to calculate probability values for the classification decisions.

Unlike the photometric XGBoost pipeline, this module uses spectroscopic measurements and predefined decision rules rather than a machine-learning classifier trained on photometric features.

Input Data

The procedure uses two Excel files:

* FINAL_COMBINED_unique.xlsx: Contains the combined stellar sample, including TIC identifiers, LAMOST observation identifiers, and reference CP classifications.
* EW_COMPLETE_2008.xlsx: Contains the summed equivalent widths for the CP1 and CP2 spectral-line groups.

The file paths must be configured for the local computing environment before execution.

Equivalent-Width Features

The mean equivalent width for each spectral-line group is calculated as:

[
\overline{EW}{CP1} =
\frac{\sum EW{CP1}}{10}
]

[
\overline{EW}{CP2} =
\frac{\sum EW{CP2}}{14}
]

Here, the CP1 group contains 10 spectral lines, while the CP2 group contains 14 spectral lines.

Samples are retained only when both mean equivalent-width values are less than or equal to 1.0.

The input datasets are matched using cleaned observation identifiers, and duplicate TIC identifiers are removed during data preparation.

Gaussian Probability Model

The classification procedure uses the following Gaussian parameters:

Parameter	Value
(\mu_1)	0.3185
(\sigma_1)	0.0796
(\mu_2)	0.1559
(\sigma_2)	0.0627

These parameters represent the fitted means and standard deviations associated with the two equivalent-width distributions used by the implementation.

The Gaussian cumulative distribution function is evaluated as:

[
F(x;\mu,\sigma) =
\Phi\left(\frac{x-\mu}{\sigma}\right)
]

where (\Phi) denotes the standard normal cumulative distribution function.

Classification Rules

The function rec_cp assigns a classification code and a probability value according to the relative equivalent widths and the specified data ranges.

The implemented rules are:

* CP1 (rec_y = 0): The CP1 mean equivalent width is greater than the CP2 mean equivalent width and lies within the specified CP1 range.
* CP2 (rec_y = 1): The CP2 mean equivalent width is greater than the CP1 mean equivalent width and lies within the specified CP2 range.
* Ambiguous (rec_y = 10): The two mean equivalent widths differ by no more than approximately 5% relative to one another, according to the implemented comparison.
* Out of Range (rec_y = -1): No classification rule is satisfied.

For CP1 and CP2 decisions, the probability is calculated using the corresponding Gaussian cumulative distribution function. Ambiguous cases are assigned a probability of 0.5.

The numerical classification codes are converted into descriptive labels for reporting.

Evaluation

The script compares the predicted labels with the reference CP_Type classifications.

It reports:

* The number and percentage of stars assigned to each classification category
* The mean and standard deviation of the calculated probability values for each category
* A confusion matrix for all classification outcomes
* Classification accuracy for samples assigned to CP1 or CP2
* Precision, recall, and F1-score for both classes
* True Skill Statistic (TSS)
* Heidke Skill Score (HSS)

The accuracy calculated for the CP1 and CP2 decisions excludes samples classified as ambiguous or out of range.

Output

The final results are saved to:

F:\LAMOST_EW_analysis_results\rec_cp_results_new_params.xlsx

The output file includes stellar identifiers, reference classifications, summed and mean equivalent widths, numerical classification codes, calculated probabilities, and final classification labels.

Purpose

This module provides a spectroscopic classification approach based on equivalent-width measurements and Gaussian probability distributions. Its results can be compared with reference classifications to assess the performance of the implemented decision rules.
