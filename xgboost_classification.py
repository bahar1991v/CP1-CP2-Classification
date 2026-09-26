# -*- coding: utf-8 -*-

import os
import re
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================================================
# SETTINGS
# =========================================================

TRAIN_FILE = "data/features/features1_augmented.csv"

# Matched external feature file
EXTERNAL_FILE = "data/external/fall_features_lamost14686_matched.csv"

OUTPUT_DIR = "results/xgb_external"

N_ITERATIONS = 100
TRAIN_FRACTION = 0.90
RANDOM_STATE = None

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD TRAIN DATA
# =========================================================

df = pd.read_csv(TRAIN_FILE)

print("=" * 100)
print("TRAIN DATA LOADED (FIRST FILE)")
print("=" * 100)

print(f"Total rows: {len(df)}")
print(
    f"CP1 rows: {len(df[df['class'] == 0])} | "
    f"CP2 rows: {len(df[df['class'] == 1])}"
)
print(
    f"CP1 unique TICs: "
    f"{df[df['class'] == 0]['tic'].nunique()}"
)
print(
    f"CP2 unique TICs: "
    f"{df[df['class'] == 1]['tic'].nunique()}"
)
print(f"Total unique TICs: {df['tic'].nunique()}")


# =========================================================
# FEATURES
# =========================================================

exclude_cols = [
    'class',
    'tic',
    'sector',
    'is_augmented',
    'filename'
]

feature_cols = [
    c for c in df.columns
    if c not in exclude_cols
]

print(f"\nNumber of features: {len(feature_cols)}")


# =========================================================
# CHECK: EACH TIC HAS ONLY ONE CLASS
# =========================================================

tic_class_check = df.groupby('tic')['class'].nunique()
problematic_tics = tic_class_check[tic_class_check > 1]

if len(problematic_tics) > 0:
    print(
        f"\nWARNING: {len(problematic_tics)} TICs "
        "have more than one class!"
    )
    raise ValueError(
        "Some TICs belong to multiple classes."
    )

else:
    print("\nEvery TIC belongs to exactly one class.")


# =========================================================
# UNIQUE TICs PER CLASS
# =========================================================

cp1_tics = (
    df.loc[df['class'] == 0, 'tic']
    .drop_duplicates()
    .values
)

cp2_tics = (
    df.loc[df['class'] == 1, 'tic']
    .drop_duplicates()
    .values
)

print(
    f"\nUnique TICs → CP1 = {len(cp1_tics)} | "
    f"CP2 = {len(cp2_tics)}"
)


# =========================================================
# LOAD EXTERNAL DATA (SECOND FILE — MATCHED)
# =========================================================

print("\n" + "=" * 100)
print("EXTERNAL DATA LOADED (SECOND FILE — MATCHED)")
print("=" * 100)

external_df_raw = pd.read_csv(EXTERNAL_FILE)

print(f"External rows: {len(external_df_raw)}")

first_col = external_df_raw.columns[0]


def extract_tic(filename):
    m = re.match(
        r'TIC(\d+)_',
        str(filename).strip()
    )
    return int(m.group(1)) if m else np.nan


def extract_sector(filename):
    m = re.search(
        r'sector_?(\d+)',
        str(filename).strip()
    )
    return int(m.group(1)) if m else np.nan


external_df = external_df_raw.copy()

external_df['tic'] = (
    external_df[first_col]
    .apply(extract_tic)
)

external_df['sector'] = (
    external_df[first_col]
    .apply(extract_sector)
)

external_df = (
    external_df
    .dropna(subset=['tic'])
    .reset_index(drop=True)
)

external_df['tic'] = external_df['tic'].astype(int)
external_df['sector'] = external_df['sector'].astype(int)

print(
    f"External rows with valid TIC: "
    f"{len(external_df)}"
)

print(
    f"External unique TICs: "
    f"{external_df['tic'].nunique()}"
)


# Check overlap
train_tic_set = set(df['tic'].unique())
ext_tic_set = set(external_df['tic'].unique())

overlap = train_tic_set & ext_tic_set

if len(overlap) > 0:

    print(
        f"\nWARNING: {len(overlap)} TICs overlap "
        "→ removed from external to avoid leakage."
    )

    external_df = (
        external_df[
            ~external_df['tic'].isin(overlap)
        ]
        .reset_index(drop=True)
    )

    print(
        f"External rows after removal: "
        f"{len(external_df)}"
    )

    print(
        f"External unique TICs after removal: "
        f"{external_df['tic'].nunique()}"
    )

else:
    print(
        "\nNo TIC overlap between train and external."
    )


# Check features
missing_feats = [
    c for c in feature_cols
    if c not in external_df.columns
]

if missing_feats:
    raise ValueError(
        f"Missing features in external: {missing_feats}"
    )

print(
    f"\nAll {len(feature_cols)} "
    "features exist in external."
)


# =========================================================
# FUNCTIONS: TSS / HSS / METRICS
# =========================================================

def calculate_tss_hss(cm):

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    TPR = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0
    )

    FPR = (
        FP / (FP + TN)
        if (FP + TN) > 0
        else 0
    )

    TSS = TPR - FPR

    total = TP + TN + FP + FN

    if total == 0:
        return 0, 0, TP, TN, FP, FN

    expected_acc = (
        (
            (TP + FN) * (TP + FP)
            + (FP + TN) * (FN + TN)
        )
        / (total * total)
    )

    accuracy = (TP + TN) / total

    HSS = (
        (accuracy - expected_acc)
        / (1 - expected_acc)
        if (1 - expected_acc) > 0
        else 0
    )

    return TSS, HSS, TP, TN, FP, FN


def compute_metrics(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    TSS, HSS, TP, TN, FP, FN = (
        calculate_tss_hss(cm)
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    prec = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=[0, 1],
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=[0, 1],
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=[0, 1],
        zero_division=0
    )

    return {
        'accuracy': acc,
        'precision_cp1': prec[0],
        'precision_cp2': prec[1],
        'recall_cp1': rec[0],
        'recall_cp2': rec[1],
        'f1_cp1': f1[0],
        'f1_cp2': f1[1],
        'tss': TSS,
        'hss': HSS,
        'tp': TP,
        'tn': TN,
        'fp': FP,
        'fn': FN
    }


# =========================================================
# STORAGE
# =========================================================

all_probability_records = []
all_metrics = []


# =========================================================
# 100 ITERATIONS
# =========================================================

print("\n" + "=" * 100)
print(f"STARTING {N_ITERATIONS} ITERATIONS")
print("=" * 100)


for iteration in range(1, N_ITERATIONS + 1):

    print("\n" + "-" * 100)
    print(
        f"ITERATION {iteration}/{N_ITERATIONS}"
    )
    print("-" * 100)

    # New random split for each iteration

    rng = np.random.default_rng()

    # CP1
    n_cp1_train = int(
        round(len(cp1_tics) * TRAIN_FRACTION)
    )

    shuffled_cp1 = rng.permutation(cp1_tics)

    train_cp1_tics = (
        shuffled_cp1[:n_cp1_train]
    )

    val_cp1_tics = (
        shuffled_cp1[n_cp1_train:]
    )

    # CP2
    n_cp2_train = int(
        round(len(cp2_tics) * TRAIN_FRACTION)
    )

    shuffled_cp2 = rng.permutation(cp2_tics)

    train_cp2_tics = (
        shuffled_cp2[:n_cp2_train]
    )

    val_cp2_tics = (
        shuffled_cp2[n_cp2_train:]
    )

    train_tics = np.concatenate(
        [train_cp1_tics, train_cp2_tics]
    )

    val_tics = np.concatenate(
        [val_cp1_tics, val_cp2_tics]
    )

    # Build train and validation sets

    train_df = df[
        df['tic'].isin(train_tics)
    ].copy()

    val_df = df[
        df['tic'].isin(val_tics)
    ].copy()

    print(
        f"Train TICs (random)      : "
        f"{train_df['tic'].nunique()}"
    )

    print(
        f"Train rows               : "
        f"{len(train_df)}"
    )

    print(
        f"Validation TICs (10%)    : "
        f"{val_df['tic'].nunique()}"
    )

    print(
        f"Validation rows (10%)    : "
        f"{len(val_df)}"
    )

    print(
        f"External TICs (fixed)    : "
        f"{external_df['tic'].nunique()}"
    )

    print(
        f"External rows (fixed)    : "
        f"{len(external_df)}"
    )

    # X / y

    X_train = train_df[
        feature_cols
    ].values

    y_train = train_df[
        'class'
    ].values

    X_val = val_df[
        feature_cols
    ].values

    y_val = val_df[
        'class'
    ].values

    X_external = external_df[
        feature_cols
    ].values

    # Scaling

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(X_train)
    )

    X_val_scaled = (
        scaler.transform(X_val)
    )

    X_external_scaled = (
        scaler.transform(X_external)
    )

    # Class weight

    n_train_cp1 = np.sum(
        y_train == 0
    )

    n_train_cp2 = np.sum(
        y_train == 1
    )

    scale_pos_weight = (
        n_train_cp1 / n_train_cp2
        if n_train_cp2 > 0
        else 1
    )

    # XGBoost

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=None
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    # Prediction on held-out 10%

    y_pred_val = model.predict(
        X_val_scaled
    )

    val_metrics = compute_metrics(
        y_val,
        y_pred_val
    )

    val_metrics['iteration'] = iteration

    # Print validation metrics

    print(
        "\n  -- Internal Validation "
        "(10% held-out TICs) --"
    )

    print(
        f"    Accuracy     : "
        f"{val_metrics['accuracy']:.4f}"
    )

    print(
        f"    Recall CP1   : "
        f"{val_metrics['recall_cp1']:.4f}"
    )

    print(
        f"    Recall CP2   : "
        f"{val_metrics['recall_cp2']:.4f}"
    )

    print(
        f"    Precision CP1: "
        f"{val_metrics['precision_cp1']:.4f}"
    )

    print(
        f"    Precision CP2: "
        f"{val_metrics['precision_cp2']:.4f}"
    )

    print(
        f"    F1 CP1       : "
        f"{val_metrics['f1_cp1']:.4f}"
    )

    print(
        f"    F1 CP2       : "
        f"{val_metrics['f1_cp2']:.4f}"
    )

    print(
        f"    TSS          : "
        f"{val_metrics['tss']:.4f}"
    )

    print(
        f"    HSS          : "
        f"{val_metrics['hss']:.4f}"
    )

    print(
        f"    TP={val_metrics['tp']}  "
        f"TN={val_metrics['tn']}  "
        f"FP={val_metrics['fp']}  "
        f"FN={val_metrics['fn']}"
    )

    all_metrics.append(
        val_metrics
    )

    # Prediction on external data

    prob_external = (
        model.predict_proba(
            X_external_scaled
        )[:, 1]
    )

    iteration_prob_df = (
        external_df[
            ['tic', 'sector']
        ].copy()
    )

    iteration_prob_df[
        'Probability_CP2'
    ] = prob_external

    iteration_prob_df[
        'Iteration'
    ] = iteration

    all_probability_records.append(
        iteration_prob_df
    )

    print(
        "\n  -- External (unknown data) --"
    )

    print(
        f"    Mean prob CP2 = "
        f"{prob_external.mean():.4f} | "
        f"min = {prob_external.min():.4f} | "
        f"max = {prob_external.max():.4f}"
    )


# =========================================================
# COMBINE
# =========================================================

print("\n" + "=" * 100)
print("COMBINING ALL 100 ITERATIONS")
print("=" * 100)

all_probability_df = pd.concat(
    all_probability_records,
    ignore_index=True
)


# =========================================================
# SAVE RAW PROBABILITIES -> CSV
# =========================================================

raw_probability_path = os.path.join(
    OUTPUT_DIR,
    "100_iterations_external_probabilities.csv"
)

all_probability_df.to_csv(
    raw_probability_path,
    index=False,
    encoding='utf-8-sig'
)

print(
    f"\nRaw probabilities saved (CSV):\n"
    f"{raw_probability_path}"
)


# =========================================================
# SAVE VALIDATION METRICS
# =========================================================

metrics_df = pd.DataFrame(
    all_metrics
)

metrics_path = os.path.join(
    OUTPUT_DIR,
    "100_iterations_validation_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding='utf-8-sig'
)

print(
    f"Validation metrics saved:\n"
    f"{metrics_path}"
)


# =========================================================
# PER-TIC SUMMARY (external)
# =========================================================

tic_probability_summary = (
    all_probability_df
    .groupby(['tic'])
    .agg(
        Mean_Prob_CP2=(
            'Probability_CP2',
            'mean'
        ),
        Std_Prob_CP2=(
            'Probability_CP2',
            'std'
        ),
        N_Test_Iterations=(
            'Probability_CP2',
            'count'
        ),
        Min_Prob_CP2=(
            'Probability_CP2',
            'min'
        ),
        Max_Prob_CP2=(
            'Probability_CP2',
            'max'
        ),
    )
    .reset_index()
)

tic_probability_summary[
    'Std_Prob_CP2'
] = (
    tic_probability_summary[
        'Std_Prob_CP2'
    ].fillna(0)
)

tic_probability_summary[
    'Mean_Prob_CP1'
] = (
    1 - tic_probability_summary[
        'Mean_Prob_CP2'
    ]
)


# =========================================================
# LABEL WITH 85% THRESHOLD
# =========================================================

CONFIDENCE_THRESHOLD = 0.85


def assign_label(prob):

    if prob >= CONFIDENCE_THRESHOLD:
        return 'CP2'

    elif (1 - prob) >= CONFIDENCE_THRESHOLD:
        return 'CP1'

    else:
        return 'Uncertain'


tic_probability_summary[
    'Predicted_Label_85pct'
] = (
    tic_probability_summary[
        'Mean_Prob_CP2'
    ].apply(assign_label)
)

tic_probability_summary = (
    tic_probability_summary
    .sort_values('Mean_Prob_CP2')
    .reset_index(drop=True)
)


# =========================================================
# SAVE PER-TIC SUMMARY
# =========================================================

tic_summary_path = os.path.join(
    OUTPUT_DIR,
    "TIC_probability_mean_std_100_iterations.xlsx"
)

tic_probability_summary.to_excel(
    tic_summary_path,
    index=False
)

print(
    f"\nPer-TIC summary saved:\n"
    f"{tic_summary_path}"
)


# =========================================================
# FINAL DISTRIBUTION
# =========================================================

print("\n" + "=" * 100)
print(
    "FINAL PREDICTION DISTRIBUTION "
    "(EXTERNAL)"
)
print("=" * 100)

distribution = (
    tic_probability_summary[
        'Predicted_Label_85pct'
    ].value_counts()
)

print(distribution)

n_cp1 = int(
    distribution.get('CP1', 0)
)

n_cp2 = int(
    distribution.get('CP2', 0)
)

n_unc = int(
    distribution.get('Uncertain', 0)
)

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"CP1       : {n_cp1}")
print(f"CP2       : {n_cp2}")
print(f"Uncertain : {n_unc}")
print(
    f"Total     : "
    f"{len(tic_probability_summary)}"
)


# =========================================================
# SUMMARY OF VALIDATION METRICS
# =========================================================

print("\n" + "=" * 100)
print(
    "VALIDATION METRICS — MEAN ± STD "
    "OVER 100 ITERATIONS"
)
print("=" * 100)

for col, name in [
    ('accuracy', 'Accuracy'),
    ('recall_cp1', 'Recall CP1'),
    ('recall_cp2', 'Recall CP2'),
    ('precision_cp1', 'Precision CP1'),
    ('precision_cp2', 'Precision CP2'),
    ('f1_cp1', 'F1 CP1'),
    ('f1_cp2', 'F1 CP2'),
    ('tss', 'TSS'),
    ('hss', 'HSS'),
]:

    print(
        f"{name:<16}: "
        f"{metrics_df[col].mean():.4f} "
        f"± "
        f"{metrics_df[col].std():.4f}"
    )


# =========================================================
# TOP 20 EXTERNAL TICs
# =========================================================

print("\n" + "=" * 100)
print(
    "TOP 20 TICs WITH HIGHEST MEAN "
    "PROB CP2 (EXTERNAL)"
)
print("=" * 100)

print(
    tic_probability_summary
    .tail(20)
    [
        [
            'tic',
            'Mean_Prob_CP2',
            'Std_Prob_CP2',
            'N_Test_Iterations',
            'Predicted_Label_85pct'
        ]
    ]
    .to_string(index=False)
)


# =========================================================
# DONE
# =========================================================

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

print("\nFiles created:")

print(
    f"1) Raw probabilities (CSV)     : "
    f"{raw_probability_path}"
)

print(
    f"2) Validation metrics (CSV)    : "
    f"{metrics_path}"
)

print(
    f"3) Per-TIC summary (Excel)     : "
    f"{tic_summary_path}"
)
