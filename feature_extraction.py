import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft
from statsmodels.tsa.stattools import acf
import pywt
import pandas as pd
import os
import re
import glob

# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = "data/augmented_npz"
OUTPUT_FILE = "results/features1_augmented.csv"

# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features_simple(mag_series):
    mag_series = np.array(mag_series)
    mag_series = mag_series[~np.isnan(mag_series)]

    if len(mag_series) < 10:
        return None

    N = len(mag_series)
    acf_vals = acf(mag_series, nlags=5, fft=True)

    features = [
        np.std(mag_series),
        np.min(mag_series),
        np.max(mag_series),
        np.median(mag_series),
        np.percentile(mag_series, 25),
        np.percentile(mag_series, 75),
        np.percentile(mag_series, 75) - np.percentile(mag_series, 25),
        skew(mag_series),
        kurtosis(mag_series),
        *acf_vals[1:6],
        *(np.abs(fft(mag_series)) / N)[:10]
    ]

    try:
        coeffs = pywt.wavedec(mag_series, wavelet='db4', level=3)
        for c in coeffs:
            features.extend([np.mean(c), np.std(c), np.max(c), np.min(c)])
    except:
        features.extend([0] * 16)

    return features


feature_names = [
    'std_mag', 'min_mag', 'max_mag', 'median_mag',
    'q25_mag', 'q75_mag', 'iqr_mag', 'skew_mag', 'kurtosis_mag',
    'acf_1', 'acf_2', 'acf_3', 'acf_4', 'acf_5',
    'fft_0', 'fft_1', 'fft_2', 'fft_3', 'fft_4',
    'fft_5', 'fft_6', 'fft_7', 'fft_8', 'fft_9',
    'mean_a3', 'std_a3', 'max_a3', 'min_a3',
    'mean_d3', 'std_d3', 'max_d3', 'min_d3',
    'mean_d2', 'std_d2', 'max_d2', 'min_d2',
    'mean_d1', 'std_d1', 'max_d1', 'min_d1'
]

print(f"Total features: {len(feature_names)}")
print("=" * 60)

# =========================================================
# PIPELINE — Read from NPZ files
# =========================================================

rows = []
name_pattern = re.compile(
    r"TIC[_]?(\d+)_?[Ss]ector[_]?(\d+)",
    re.IGNORECASE
)

for class_name in ["CP1", "CP2"]:
    class_dir = os.path.join(BASE_DIR, class_name)

    if not os.path.exists(class_dir):
        print(f"WARNING: Directory {class_dir} does not exist")
        continue

    label = 0 if class_name == "CP1" else 1

    files = glob.glob(os.path.join(class_dir, "*.npz"))
    print(f"\nProcessing {class_name}: {len(files)} files")

    for file_path in files:
        fname = os.path.basename(file_path)
        m = name_pattern.search(fname)

        if not m:
            print(f"WARNING: Filename pattern did not match: {fname}")
            continue

        tic = m.group(1)
        sector = m.group(2)

        # Determine whether the file is original or augmented
        is_augmented = "_aug" in fname

        try:
            data = np.load(file_path)

            if "magnitude_detrended" not in data:
                print(f"WARNING: Required column not found in {fname}")
                continue

            signal = data["magnitude_detrended"]

            feats = extract_features_simple(signal)

            if feats is None:
                continue

            row = {
                "tic": tic,
                "sector": sector,
                "class": label,
                "is_augmented": is_augmented,
                "filename": fname,
            }

            for i, f_name in enumerate(feature_names):
                row[f_name] = feats[i] if i < len(feats) else 0

            rows.append(row)

            if len(rows) % 500 == 0:
                print(f"   Progress: {len(rows)} files processed...")

        except Exception as e:
            print(f"WARNING: Error in {fname}: {e}")
            continue

# =========================================================
# SAVE
# =========================================================

if rows:
    df_out = pd.DataFrame(rows)

    meta_cols = ['tic', 'sector', 'class', 'is_augmented', 'filename']
    feature_cols_only = [
        c for c in df_out.columns
        if c not in meta_cols
    ]

    cols_to_remove = []

    for col in feature_cols_only:
        min_val = df_out[col].min()
        max_val = df_out[col].max()

        if df_out[col].nunique() == 1:
            cols_to_remove.append(col)
            print(f"Removing constant column: {col}")

        elif min_val == 0 and max_val == 0:
            cols_to_remove.append(col)
            print(f"Removing all-zero column: {col}")

    if cols_to_remove:
        final_cols = [
            c for c in df_out.columns
            if c not in cols_to_remove
        ]

        df_out = df_out[final_cols].copy()

        print(
            f"\nRemoved {len(cols_to_remove)} "
            "constant/all-zero columns"
        )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df_out.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Samples: {len(df_out)}")
    print(f"Features: {len(df_out.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nSample distribution:")
    print(df_out["class"].value_counts())

    print("\nOriginal vs. augmented:")
    print(df_out["is_augmented"].value_counts())

    print("\nSample of extracted features:")
    print(df_out.head())

else:
    print("WARNING: No data was processed!")
