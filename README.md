# Telco Customer Churn Prediction

Predicting which telecom customers are likely to cancel their subscription, using the
IBM Telco Customer Churn dataset. The project covers the full workflow: data cleaning,
exploratory data analysis, feature engineering, and a comparison of three machine
learning models.

## Problem

Every customer who churns is lost monthly revenue. If a model can flag the customers
most likely to leave, the retention team can target them with offers *before* they cancel,
instead of spending the same budget on everyone.

## Dataset

- **Source:** IBM Telco Customer Churn (`data/telco_churn.csv`)
- **Size:** 7,043 customers × 21 columns (7,032 rows after cleaning)
- **Target:** `Churn` (Yes/No) — overall churn rate is **26.6%**, so the classes are imbalanced
- Features include contract type, tenure, monthly/total charges, internet service,
  add-on services, and payment method.

## Approach

**Cleaning**
- `TotalCharges` was stored as text with blank values for 11 brand-new customers
  (tenure = 0) — converted it to numeric and dropped those 11 rows.
- Dropped the `customerID` column and encoded the target as 1/0.

**Key EDA findings**
- Month-to-month customers churn at **43%**, vs 11% for one-year and 3% for two-year contracts.
- Fibre-optic users churn at **42%** — the highest of any internet service — and also pay the most.
- Electronic-check payers churn at **45%**, vs under 20% for automatic payment methods.
- Churn is mostly a new-customer problem: the median churner leaves after ~10 months.

**Feature engineering**
- `tenure_group`: tenure bucketed into 0–12, 13–24, 25–48, 49–72 months.
- `num_addons`: count of the six add-on services each customer subscribes to (0–6).
- `avg_charge_per_month`: total charges divided by tenure.
- All categorical columns one-hot encoded; stratified 80/20 train/test split.
  Features are scaled for Logistic Regression only (tree models don't need it).

**Models compared:** Logistic Regression, Random Forest, Gradient Boosting.

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.794 | 0.636 | 0.527 | 0.576 | 0.835 |
| Random Forest | 0.789 | 0.626 | 0.511 | 0.563 | 0.821 |
| **Gradient Boosting** | **0.797** | **0.646** | 0.521 | **0.577** | **0.840** |

Gradient Boosting performed best overall. Its most important features were:

1. **tenure** — how long the customer has been with the company
2. **InternetService (Fibre optic)**
3. **PaymentMethod (Electronic check)**

## Key Insights

- **Contract length is the strongest lever.** Getting a month-to-month customer onto a
  yearly contract cuts their churn risk by roughly 4×, so migration offers are the
  highest-value retention spend.
- **Fibre-optic customers need attention** — highest prices *and* highest churn suggests
  a price/expectation problem worth investigating.
- **Auto-pay is protective.** Customers on automatic payments churn at less than half the
  rate of electronic-check payers; a small incentive to switch could pay off.
- **Watch customers with zero add-ons in their first year** — they are the highest-risk
  segment and the cheapest to target early.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

Run the notebook top to bottom. It loads `data/telco_churn.csv`, reproduces all charts
and model results, and writes the final metrics to `metrics.json`.

## Tech Stack

- Python, Pandas, NumPy — data cleaning and feature engineering
- Matplotlib, Seaborn — EDA visualisations
- Scikit-learn — Logistic Regression, Random Forest, Gradient Boosting, evaluation metrics
- Jupyter Notebook
