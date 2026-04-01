"""
Customer Churn Analysis — IBM Telco Dataset (Real Data)
Author : Shaik Roshan Basha
Tools  : Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv (7,043 records)
Goal   : Identify key churn drivers and build a predictive model to
         help the business reduce customer attrition.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import json, os, warnings
warnings.filterwarnings("ignore")

DATA = "/home/claude/churn_project/data/telco_churn.csv"
OUT  = "/home/claude/churn_project/outputs"
os.makedirs(OUT, exist_ok=True)

BLUE  = "#1B3A6B"
ACCENT= "#2E86C1"
RED   = "#C0392B"
GREEN = "#1E8449"
GRAY  = "#BDC3C7"
BG    = "#F8F9FA"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans", "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11
})

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────
print("=" * 55)
print("SECTION 1 — DATA LOADING & CLEANING")
print("=" * 55)

df = pd.read_csv(DATA)
print(f"\nRaw shape         : {df.shape}")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
blanks = df["TotalCharges"].isna().sum()
print(f"Blank TotalCharges: {blanks} rows dropped")
df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
print(f"Clean shape       : {df.shape}")
print(f"Remaining nulls   : {df.isnull().sum().sum()}")

N          = len(df)
churn_rate = (df.Churn == "Yes").mean() * 100
print(f"\nTotal customers   : {N:,}")
print(f"Churned           : {(df.Churn=='Yes').sum():,}  ({churn_rate:.1f}%)")
print(f"Retained          : {(df.Churn=='No').sum():,}  ({100-churn_rate:.1f}%)")
print(f"Avg Monthly Charge: ${df.MonthlyCharges.mean():.2f}")
print(f"Avg Tenure        : {df.tenure.mean():.1f} months")

# ─────────────────────────────────────────────
# 2. EDA PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 55)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Customer Churn Overview — IBM Telco Dataset",
             fontsize=15, fontweight="bold", color=BLUE, y=1.01)

churn_counts = df.Churn.value_counts()
axes[0].bar(["Retained","Churned"], churn_counts.values,
            color=[GREEN, RED], width=0.5, edgecolor="white", linewidth=1.5)
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v+40, f"{v:,}\n({v/N*100:.1f}%)",
                 ha="center", fontsize=11, fontweight="bold")
axes[0].set_title("Churn Distribution")
axes[0].set_ylabel("Number of Customers")
axes[0].set_ylim(0, max(churn_counts.values)*1.15)

axes[1].hist(df[df.Churn=="No"].tenure,  bins=30, alpha=0.7, color=GREEN, label="Retained")
axes[1].hist(df[df.Churn=="Yes"].tenure, bins=30, alpha=0.7, color=RED,   label="Churned")
axes[1].set_title("Tenure Distribution by Churn Status")
axes[1].set_xlabel("Tenure (months)")
axes[1].set_ylabel("Count")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/01_churn_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] 01_churn_overview.png")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Churn Rate by Key Business Segments",
             fontsize=15, fontweight="bold", color=BLUE)
cats = [
    ("Contract",        "Churn Rate by Contract Type"),
    ("InternetService", "Churn Rate by Internet Service"),
    ("PaymentMethod",   "Churn Rate by Payment Method"),
    ("SeniorCitizen",   "Churn Rate: Senior vs Non-Senior"),
]
for ax, (col, title) in zip(axes.flat, cats):
    rates = df.groupby(col)["Churn"].apply(
        lambda x: (x=="Yes").mean()*100).sort_values()
    bars = ax.barh(rates.index.astype(str), rates.values,
                   color=[RED if v > churn_rate else ACCENT for v in rates.values],
                   edgecolor="white", linewidth=1)
    for bar, val in zip(bars, rates.values):
        ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.axvline(churn_rate, color=GRAY, linestyle="--",
               linewidth=1.2, label=f"Avg {churn_rate:.1f}%")
    ax.set_title(title)
    ax.set_xlabel("Churn Rate (%)")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/02_churn_by_segments.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] 02_churn_by_segments.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Charges & Tenure Analysis", fontsize=15, fontweight="bold", color=BLUE)
axes[0].hist(df[df.Churn=="No"].MonthlyCharges,  bins=30, alpha=0.7, color=GREEN, label="Retained")
axes[0].hist(df[df.Churn=="Yes"].MonthlyCharges, bins=30, alpha=0.7, color=RED,   label="Churned")
axes[0].set_title("Monthly Charges Distribution by Churn")
axes[0].set_xlabel("Monthly Charges ($)")
axes[0].set_ylabel("Count")
axes[0].legend()

tenure_bins  = pd.cut(df.tenure, bins=[0,6,12,24,48,72],
                      labels=["0-6m","7-12m","13-24m","25-48m","49-72m"])
tenure_churn = df.groupby(tenure_bins, observed=True)["Churn"].apply(
    lambda x: (x=="Yes").mean()*100)
axes[1].plot(tenure_churn.index.astype(str), tenure_churn.values,
             marker="o", color=BLUE, linewidth=2.5, markersize=8)
axes[1].fill_between(range(len(tenure_churn)), tenure_churn.values,
                     alpha=0.15, color=BLUE)
axes[1].set_title("Churn Rate by Tenure Band")
axes[1].set_xlabel("Tenure Band")
axes[1].set_ylabel("Churn Rate (%)")
axes[1].set_xticks(range(len(tenure_churn)))
axes[1].set_xticklabels(tenure_churn.index.astype(str))
plt.tight_layout()
plt.savefig(f"{OUT}/03_charges_tenure.png", dpi=150, bbox_inches="tight")
plt.close()
print("[Saved] 03_charges_tenure.png")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 3 — PREPROCESSING")
print("=" * 55)

df_model = df.drop("customerID", axis=1).copy()
df_model["Churn"] = (df_model["Churn"] == "Yes").astype(int)
le = LabelEncoder()
for col in df_model.select_dtypes("object").columns:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"\nFeatures: {X.shape[1]}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 4 — MODEL TRAINING & EVALUATION")
print("=" * 55)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_prob = lr.predict_proba(X_test_s)[:,1]
lr_acc  = accuracy_score(y_test, lr_pred)
lr_auc  = roc_auc_score(y_test, lr_prob)

rf = RandomForestClassifier(n_estimators=200, random_state=42,
                             class_weight="balanced", n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:,1]
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf_prob)

print(f"\nLogistic Regression — Accuracy: {lr_acc*100:.1f}%  |  AUC: {lr_auc:.3f}")
print(f"Random Forest       — Accuracy: {rf_acc*100:.1f}%  |  AUC: {rf_auc:.3f}")
print(f"\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, target_names=["Retained","Churned"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Evaluation", fontsize=15, fontweight="bold", color=BLUE)
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Retained","Churned"],
            yticklabels=["Retained","Churned"])
axes[0].set_title("Confusion Matrix — Random Forest")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

for prob, label, color in [
    (lr_prob, f"Logistic Reg (AUC={lr_auc:.3f})", ACCENT),
    (rf_prob, f"Random Forest (AUC={rf_auc:.3f})", BLUE),
]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[1].plot(fpr, tpr, color=color, linewidth=2, label=label)
axes[1].plot([0,1],[0,1],"k--",linewidth=1)
axes[1].set_title("ROC Curve Comparison")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(fontsize=9)

fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(10)
axes[2].barh(fi.index, fi.values, color=BLUE, edgecolor="white")
axes[2].set_title("Top 10 Feature Importances")
axes[2].set_xlabel("Importance Score")
plt.tight_layout()
plt.savefig(f"{OUT}/04_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[Saved] 04_model_evaluation.png")

# ─────────────────────────────────────────────
# 5. INSIGHTS & SAVE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("SECTION 5 — BUSINESS INSIGHTS")
print("=" * 55)

mtm_churn    = (df[df.Contract=="Month-to-month"]["Churn"]=="Yes").mean()*100
fiber_churn  = (df[df.InternetService=="Fiber optic"]["Churn"]=="Yes").mean()*100
echeck_churn = (df[df.PaymentMethod=="Electronic check"]["Churn"]=="Yes").mean()*100
new_churn    = (df[df.tenure<=6]["Churn"]=="Yes").mean()*100
loyal_churn  = (df[df.tenure>=49]["Churn"]=="Yes").mean()*100

print(f"""
KEY FINDINGS (Real IBM Telco Data):
  Overall churn rate          : {churn_rate:.1f}%
  Month-to-month churn        : {mtm_churn:.1f}%  (+{mtm_churn-churn_rate:.1f} pts above avg)
  Fiber optic churn           : {fiber_churn:.1f}%  (+{fiber_churn-churn_rate:.1f} pts above avg)
  Electronic check churn      : {echeck_churn:.1f}%  (+{echeck_churn-churn_rate:.1f} pts above avg)
  New customers (0–6mo) churn : {new_churn:.1f}%
  Loyal customers (49+mo)     : {loyal_churn:.1f}%
""")

summary = {
    "total_customers": int(N),
    "churned"        : int((df.Churn=="Yes").sum()),
    "retained"       : int((df.Churn=="No").sum()),
    "churn_rate"     : round(churn_rate, 1),
    "avg_monthly"    : round(df.MonthlyCharges.mean(), 2),
    "avg_tenure"     : round(df.tenure.mean(), 1),
    "rf_accuracy"    : round(rf_acc*100, 1),
    "rf_auc"         : round(rf_auc, 3),
    "lr_auc"         : round(lr_auc, 3),
    "mtm_churn"      : round(mtm_churn, 1),
    "fiber_churn"    : round(fiber_churn, 1),
    "echeck_churn"   : round(echeck_churn, 1),
    "new_cust_churn" : round(new_churn, 1),
    "loyal_churn"    : round(loyal_churn, 1),
}
with open(f"{OUT}/summary_stats.json","w") as f:
    json.dump(summary, f, indent=2)

contract_churn = df.groupby("Contract")["Churn"].apply(
    lambda x: round((x=="Yes").mean()*100,1)).to_dict()
internet_churn = df.groupby("InternetService")["Churn"].apply(
    lambda x: round((x=="Yes").mean()*100,1)).to_dict()
t_bins = pd.cut(df.tenure, bins=[0,6,12,24,48,72],
                labels=["0-6m","7-12m","13-24m","25-48m","49-72m"])
tenure_churn = df.groupby(t_bins, observed=True)["Churn"].apply(
    lambda x: round((x=="Yes").mean()*100,1)).to_dict()
payment_churn = df.groupby("PaymentMethod")["Churn"].apply(
    lambda x: round((x=="Yes").mean()*100,1)).to_dict()

chart_data = {
    "contract_churn" : contract_churn,
    "internet_churn" : internet_churn,
    "tenure_churn"   : {str(k): v for k,v in tenure_churn.items()},
    "payment_churn"  : payment_churn,
    "churn_rate"     : round(churn_rate,1),
}
with open(f"{OUT}/chart_data.json","w") as f:
    json.dump(chart_data, f, indent=2)

print("[Saved] summary_stats.json + chart_data.json")
print("\nAnalysis complete. Real IBM Telco data used.")
