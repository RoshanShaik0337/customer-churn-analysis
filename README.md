# Customer Churn Analysis — Telco Dataset

**Tools:** Python · SQL (MySQL) · Power BI · scikit-learn  
**Author:** Shaik Roshan Basha  
**Dataset:** IBM Telco Customer Churn (7,043 records)

---

## Project Overview

This end-to-end data analytics project identifies the key drivers of customer churn for a telecommunications company and builds a machine learning model to predict at-risk customers. The final output includes actionable business recommendations to reduce churn and protect revenue.

**Business Problem:** The company is losing 35.4% of customers annually. Leadership needs to understand *why* customers leave and *which* customers are most at risk so the retention team can act proactively.

---

## Key Findings

| Driver | Churn Rate | vs Overall Avg (35.4%) |
|---|---|---|
| Month-to-month contract | 52.3% | +16.9 pts |
| Fiber optic internet | 47.3% | +11.9 pts |
| Electronic check payment | 45.1% | +9.7 pts |
| New customers (0–6 months) | 40.4% | +5.0 pts |
| Loyal customers (49+ months) | 29.0% | −6.4 pts |

---

## Business Recommendations

**R1 — Contract Conversion Campaign**  
Offer 15–20% discount to Month-to-month customers who upgrade to annual contracts. Estimated 18–22% churn reduction if 30% convert.

**R2 — Fiber Optic Quality Review**  
Deploy NPS survey to Fiber users. Fast-track top complaints with a 60-day SLA improvement target.

**R3 — Auto-Pay Incentive**  
Offer $5/month discount for switching to bank transfer or credit card auto-pay. Targets the 45.1% churn rate in Electronic check users.

**R4 — 90-Day Onboarding Program**  
Proactive support touchpoints at day 7, 30, and 90 for new customers to address the 40.4% early churn rate.

---

## Project Structure

```
customer-churn-analysis/
│
├── data/
│   └── telco_churn.csv           # Dataset (7,043 records)
│
├── scripts/
│   ├── churn_analysis.py         # EDA + preprocessing + ML pipeline
│   └── churn_queries.sql         # 12 analytical SQL queries
│
├── outputs/
│   ├── 01_churn_overview.png     # Churn distribution + tenure histogram
│   ├── 02_churn_by_segments.png  # Churn by contract, internet, payment, age
│   ├── 03_charges_tenure.png     # Monthly charges + tenure band analysis
│   ├── 04_model_evaluation.png   # Confusion matrix, ROC curve, feature importance
│   ├── summary_stats.json        # Key metrics
│   └── chart_data.json           # Segment-level churn rates
│
├── dashboard.html                # Interactive browser dashboard (no server needed)
└── README.md
```

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Data Generation & EDA | Python (pandas, NumPy, Matplotlib, Seaborn) | Exploration, cleaning, visualisation |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest) | Churn prediction model |
| SQL Analysis | MySQL | Aggregations, segmentation, window functions |
| Dashboard | HTML + Chart.js | Interactive browser-based dashboard |
| Version Control | Git + GitHub | Project management |

---

## ML Model Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Logistic Regression | 68.3% | 0.752 |
| **Random Forest** | **70.5%** | **0.774** |

**Best model:** Random Forest with 200 estimators  
**Top predictors:** Monthly Charges, Tenure, Total Charges, Contract Type, Internet Service

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/roshan-shaik0337/customer-churn-analysis.git
cd customer-churn-analysis

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# 3. Run the full analysis
cd scripts
python churn_analysis.py

# 4. Open the dashboard
open ../dashboard.html   # or double-click the file in your browser

# 5. Run SQL queries
# Import telco_churn.csv into MySQL, then run churn_queries.sql
```

---

## Dataset

The dataset structure mirrors the publicly available [IBM Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn) on Kaggle.

**Key columns:** `customerID`, `gender`, `SeniorCitizen`, `tenure`, `Contract`, `InternetService`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn`

---

## Skills Demonstrated

- Exploratory Data Analysis (EDA) with Python
- Data cleaning and feature engineering
- SQL aggregations, CTEs, and window functions
- Binary classification with Logistic Regression and Random Forest
- Model evaluation: confusion matrix, ROC-AUC, classification report
- Interactive dashboard design
- Translating analytical findings into business recommendations
