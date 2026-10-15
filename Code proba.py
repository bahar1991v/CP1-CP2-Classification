# -*- coding: utf-8 -*-
# =========================================================
# 
# =========================================================

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# =========================================================
# paths 
# =========================================================
COMBINED = r"F:\FINAL_COMBINED_unique"
EW_ALL   = r"F:\EW_COMPLETE_2008.xlsx"
OUT_DIR  = r"F:\LAMOST_EW_analysis_results"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================================================

Number lines# =========================================================
N_CP1_LINES = 10
N_CP2_LINES = 14
EW_MAX = 1.0

# =========================================================
# function normal_cdf
# =========================================================
def normal_cdf(x, mu, sigma):
    return norm.cdf(x, loc=mu, scale=sigma)

# =========================================================
# function rec_cp
# =========================================================
def rec_cp(mu1, sigma1, mu2, sigma2, ew1, ew2,
           min_ew1, max_ew1, min_ew2, max_ew2, disew):
    y = -1
    prob = 0.0

    if ((ew1 > ew2) & (ew1 >= min_ew1) & (ew1 <= max_ew1)):
        y = 0
        prob = normal_cdf(ew1, mu1, sigma1)      # ✅ modified : sigma1

    if ((ew2 > ew1) & (ew2 >= min_ew2) & (ew2 <= max_ew2)):
        y = 1
        prob = normal_cdf(ew2, mu2, sigma2)

    if (ew1 <= 1.05 * ew2) & (ew1 >= 0.95 * ew2):
        prob = 0.5
        y = 10

    return y, prob

def clean_key(x):
    try:
        return str(int(float(x)))
    except Exception:
        return None

# =========================================================
# read data
# =========================================================
print("=" * 70)
print("📊 ا rec_cp new parameters for(2005 stars)")
print("=" * 70)

df_main = pd.read_excel(COMBINED)
df_main["tic_clean"]   = df_main["TIC"].apply(clean_key)
df_main["obsid_clean"] = df_main["LAMOST_ID"].apply(clean_key)
df_main = df_main.drop_duplicates(subset=["tic_clean"]).reset_index(drop=True)

df_ew = pd.read_excel(EW_ALL)
df_ew["obsid_clean"] = df_ew["obsid_clean"].apply(clean_key)
df_ew = df_ew.drop_duplicates(subset=["obsid_clean"])
df_ew = df_ew[["obsid_clean", "Sum_EW_CP1_Am", "Sum_EW_CP2_Ap"]]

df = df_main.merge(df_ew, on="obsid_clean", how="inner")
df = df.drop_duplicates(subset=["tic_clean"]).reset_index(drop=True)
df = df[df["CP_Type"].isin(["CP1", "CP2"])].copy()

prin(f"number all: {len(df)}")
print(f": {df['CP_Type'].value_counts().to_dict()}")

# =========================================================
# compute means EW
# =========================================================
df["Mean_EW_CP1_Am"] = df["Sum_EW_CP1_Am"] / N_CP1_LINES
df["Mean_EW_CP2_Ap"] = df["Sum_EW_CP2_Ap"] / N_CP2_LINES

# ⬇️ filter Mean EW ≤ 1
df = df[(df["Mean_EW_CP1_Am"] <= EW_MAX) &
        (df["Mean_EW_CP2_Ap"] <= EW_MAX)].copy()

print(f"filter: EW ≤ {EW_MAX}: {len(df)}")
print(f"   CP1: {(df['CP_Type']=='CP1').sum()}")
print(f"   CP2: {(df['CP_Type']=='CP2').sum()}")

# =========================================================
# ✅ new parameters 
# =========================================================
# fits:
mu1     = 0.3185      # EW1 برای CP1 (fit)
sigma1  = 0.0796
mu2     = 0.1559      # EW2 برای CP2 (fit)
sigma2  = 0.0627

# limited of data
min_ew1 = df["Mean_EW_CP1_Am"].min()
max_ew1 = df["Mean_EW_CP1_Am"].max()
min_ew2 = df["Mean_EW_CP2_Ap"].min()
max_ew2 = df["Mean_EW_CP2_Ap"].max()

disew   = 0.05

print(f"\n📊 (new parameters with fit):")
print(f"   mu1     = {mu1}")
print(f"   sigma1  = {sigma1}")
print(f"   mu2     = {mu2}")
print(f"   sigma2  = {sigma2}")
print(f"   min_ew1 = {min_ew1:.4f}")
print(f"   max_ew1 = {max_ew1:.4f}")
print(f"   min_ew2 = {min_ew2:.4f}")
print(f"   max_ew2 = {max_ew2:.4f}")

# =========================================================
# function
# =========================================================
results_y = []
results_prob = []


for i, row in df.iterrows():
    ew1 = row["Mean_EW_CP1_Am"]
    ew2 = row["Mean_EW_CP2_Ap"]

    y, prob = rec_cp(
        mu1=mu1, sigma1=sigma1,
        mu2=mu2, sigma2=sigma2,
        ew1=ew1, ew2=ew2,
        min_ew1=min_ew1, max_ew1=max_ew1,
        min_ew2=min_ew2, max_ew2=max_ew2,
        disew=disew
    )

    results_y.append(y)
    results_prob.append(prob)

df["rec_y"] = results_y
df["rec_prob"] = results_prob

# =========================================================
# classify
# =========================================================
def label_from_y(y):
    if y == -1: return "Out of Range"
    if y == 0:  return "CP1"
    if y == 1:  return "CP2"
    if y == 10: return "Ambiguous"
    return "Unknown"

df["rec_label"] = df["rec_y"].apply(label_from_y)

# =========================================================
# report
# =========================================================
print("\n" + "=" * 70)
print(" data")
print("=" * 70)

counts = df["rec_label"].value_counts()
for label, count in counts.items():
    pct = 100 * count / len(df)
    print(f"   {label:15s}: {count:5d}  ({pct:6.2f}%)")

print("\n" + "=" * 70)
print("📊 prob")
print("=" * 70)
for label in counts.index:
    sub = df[df["rec_label"] == label]
    print(f"\n🔹 {label} (n={len(sub)}):")
    print(f"   prob — mean={sub['rec_prob'].mean():.4f}, "
          f"std={sub['rec_prob'].std():.4f}")

# =========================================================
# Confusion Matrix + accuracy 
# =========================================================
print("\n" + "=" * 70)
print("📊 Confusion Matrix (all)")
print("=" * 70)
ct = pd.crosstab(df["CP_Type"], df["rec_label"],
                 margins=True, margins_name="Total")
print(ct)
#
df_dec = df[df["rec_label"].isin(["CP1", "CP2"])]
if len(df_dec) > 0:
    correct = (df_dec["rec_label"] == df_dec["CP_Type"]).sum()
    total   = len(df_dec)
    acc     = 100 * correct / total
    print(f"\n✅accuracy: {correct}/{total} = {acc:.2f}%")

    # Confusion Matrix 
    print("\n📊 Confusion Matrix ()")
    ct_dec = pd.crosstab(df_dec["CP_Type"], df_dec["rec_label"],
                         margins=True, margins_name="Total")
    print(ct_dec)

    # metrics
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    cm = confusion_matrix(df_dec["CP_Type"], df_dec["rec_label"],
                          labels=["CP1", "CP2"])
    TP, FN = int(cm[0, 0]), int(cm[0, 1])
    FP, TN = int(cm[1, 0]), int(cm[1, 1])

    acc_s = accuracy_score(df_dec["CP_Type"], df_dec["rec_label"])
    prec  = precision_score(df_dec["CP_Type"], df_dec["rec_label"],
                            average=None, labels=["CP1", "CP2"], zero_division=0)
    rec   = recall_score(df_dec["CP_Type"], df_dec["rec_label"],
                         average=None, labels=["CP1", "CP2"], zero_division=0)
    f1    = f1_score(df_dec["CP_Type"], df_dec["rec_label"],
                     average=None, labels=["CP1", "CP2"], zero_division=0)

    TPR1 = TP / (TP + FN); TPR2 = TN / (TN + FP)
    TSS  = TPR1 + TPR2 - 1
    total_cm = TP + TN + FP + FN
    exp  = ((TP+FN)*(TP+FP) + (FP+TN)*(FN+TN)) / (total_cm**2)
    HSS  = (acc_s - exp) / (1 - exp) if (1 - exp) > 0 else 0

    print(f"\n📊 metrics:")
    print(f"   Accuracy       : {acc_s:.4f}")
    print(f"   Precision CP1  : {prec[0]:.4f}")
    print(f"   Precision CP2  : {prec[1]:.4f}")
    print(f"   Recall CP1     : {rec[0]:.4f}")
    print(f"   Recall CP2     : {rec[1]:.4f}")
    print(f"   F1 CP1         : {f1[0]:.4f}")
    print(f"   F1 CP2         : {f1[1]:.4f}")
    print(f"   TSS            : {TSS:.4f}")
    print(f"   HSS            : {HSS:.4f}")

# =========================================================
# save
# =========================================================
out_path = os.path.join(OUT_DIR, "rec_cp_results_new_params.xlsx")
df[["TIC", "LAMOST_ID", "CP_Type",
    "Sum_EW_CP1_Am", "Sum_EW_CP2_Ap",
    "Mean_EW_CP1_Am", "Mean_EW_CP2_Ap",
    "rec_y", "rec_prob", "rec_label"]].to_excel(out_path, index=False)
print(f"\n✅ saved: {out_path}")