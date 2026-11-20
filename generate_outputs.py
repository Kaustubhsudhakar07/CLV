"""
generate_outputs.py
-------------------
Run this script ONCE to process the dataset, train models, and generate all
output CSV files and pickled models that the Streamlit dashboard will load.

Usage:  python generate_outputs.py
"""

import pandas as pd
import numpy as np
import dill as pickle
import os
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "online_retail_II.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND COMBINE DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

df1 = pd.read_excel(DATA_PATH, sheet_name="Year 2009-2010")
df2 = pd.read_excel(DATA_PATH, sheet_name="Year 2010-2011")
df = pd.concat([df1, df2], ignore_index=True)
print(f"  Combined dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Cleaning data...")
print("=" * 60)

initial_rows = len(df)

# 2a. Remove rows with missing Customer ID
df = df.dropna(subset=["Customer ID"])
print(f"  After removing missing Customer ID: {len(df):,} rows (removed {initial_rows - len(df):,})")

# 2b. Remove cancelled invoices (Invoice starts with 'C')
before = len(df)
df = df[~df["Invoice"].astype(str).str.startswith("C")]
print(f"  After removing cancelled invoices:  {len(df):,} rows (removed {before - len(df):,})")

# 2c. Remove non-positive quantities
before = len(df)
df = df[df["Quantity"] > 0]
print(f"  After removing non-positive qty:    {len(df):,} rows (removed {before - len(df):,})")

# 2d. Remove zero/negative prices
before = len(df)
df = df[df["Price"] > 0]
print(f"  After removing zero/negative price: {len(df):,} rows (removed {before - len(df):,})")

# 2e. Convert types
df["Customer ID"] = df["Customer ID"].astype(int)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# 2f. Create TotalPrice
df["TotalPrice"] = df["Quantity"] * df["Price"]

print(f"\n  Final cleaned dataset: {len(df):,} rows")

# Save cleaned data for the dashboard
df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
print("  Saved: outputs/cleaned_data.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: CUSTOMER-LEVEL SUMMARY (LIFETIMES)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Building customer-level RFM summary...")
print("=" * 60)

from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Create the RFM summary table that lifetimes expects
summary = summary_data_from_transaction_data(
    df,
    customer_id_col="Customer ID",
    datetime_col="InvoiceDate",
    monetary_value_col="TotalPrice",
)

print(f"  Customers in summary: {len(summary):,}")
print(f"  Customers with repeat purchases (frequency > 0): {(summary['frequency'] > 0).sum():,}")
print(f"\n  Sample:")
print(summary.head())

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: BG/NBD MODEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Training BG/NBD model...")
print("=" * 60)

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(summary["frequency"], summary["recency"], summary["T"])
print("  BG/NBD model fitted successfully.")
print(f"  Model parameters: {bgf.summary}")

# Predict expected purchases in next 6 months (180 days)
summary["predicted_purchases_6m"] = bgf.conditional_expected_number_of_purchases_up_to_time(
    180,  # 6 months = 180 days
    summary["frequency"],
    summary["recency"],
    summary["T"],
)

# Probability that customer is still alive
summary["prob_alive"] = bgf.conditional_probability_alive(
    summary["frequency"],
    summary["recency"],
    summary["T"],
)

# Save BG/NBD model
with open(os.path.join(MODEL_DIR, "bg_nbd_model.pkl"), "wb") as f:
    pickle.dump(bgf, f)
print("  Saved: models/bg_nbd_model.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: GAMMA-GAMMA MODEL
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training Gamma-Gamma model...")
print("=" * 60)

# Gamma-Gamma only works for customers with at least 1 repeat purchase
repeat_customers = summary[summary["frequency"] > 0]

ggf = GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(repeat_customers["frequency"], repeat_customers["monetary_value"])
print("  Gamma-Gamma model fitted successfully.")

# Predicted average monetary value per transaction
summary["predicted_monetary_value"] = ggf.conditional_expected_average_profit(
    summary["frequency"],
    summary["monetary_value"],
)

# Save Gamma-Gamma model
with open(os.path.join(MODEL_DIR, "gamma_gamma_model.pkl"), "wb") as f:
    pickle.dump(ggf, f)
print("  Saved: models/gamma_gamma_model.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: CLV PREDICTION (6 MONTHS)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Calculating CLV for next 6 months...")
print("=" * 60)

# CLV = predicted_purchases * predicted_avg_monetary_value (adjusted by discount rate)
# The lifetimes library has a built-in method that combines BG/NBD + Gamma-Gamma
summary["predicted_clv_6m"] = ggf.customer_lifetime_value(
    bgf,
    summary["frequency"],
    summary["recency"],
    summary["T"],
    summary["monetary_value"],
    time=6,           # 6 months
    discount_rate=0.01,  # 1% monthly discount rate
)

# Clip negative CLV values to 0
# Negative CLV can appear for one-time buyers due to the discount rate math.
# In a non-contractual retail setting, a customer's future value cannot be negative.
summary["predicted_clv_6m"] = summary["predicted_clv_6m"].clip(lower=0)

print(f"  Average predicted CLV (6 months): £{summary['predicted_clv_6m'].mean():,.2f}")
print(f"  Total predicted CLV (6 months):   £{summary['predicted_clv_6m'].sum():,.2f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Segmenting customers by CLV...")
print("=" * 60)

# Use quantiles to segment customers into 3 groups
clv_33 = summary["predicted_clv_6m"].quantile(0.33)
clv_66 = summary["predicted_clv_6m"].quantile(0.66)

def assign_segment(clv):
    if clv <= clv_33:
        return "Low Value"
    elif clv <= clv_66:
        return "Medium Value"
    else:
        return "High Value"

summary["segment"] = summary["predicted_clv_6m"].apply(assign_segment)

print("  Segment distribution:")
print(summary["segment"].value_counts().to_string())
print(f"\n  Average CLV by segment:")
print(summary.groupby("segment")["predicted_clv_6m"].mean().to_string())

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: SAVE CUSTOMER CLV AND SEGMENTS
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8: Saving output files...")
print("=" * 60)

# Reset index so Customer ID is a column
customer_clv = summary.reset_index()
customer_clv = customer_clv.rename(columns={"Customer ID": "customer_id"})

# Add historical revenue from cleaned data
historical_revenue = df.groupby("Customer ID")["TotalPrice"].sum().reset_index()
historical_revenue.columns = ["customer_id", "historical_revenue"]

customer_clv = customer_clv.merge(historical_revenue, on="customer_id", how="left")

# Add country info (most frequent country for each customer)
customer_country = df.groupby("Customer ID")["Country"].agg(lambda x: x.mode()[0]).reset_index()
customer_country.columns = ["customer_id", "country"]
customer_clv = customer_clv.merge(customer_country, on="customer_id", how="left")

# Save customer CLV
customer_clv.to_csv(os.path.join(OUTPUT_DIR, "customer_clv.csv"), index=False)
print("  Saved: outputs/customer_clv.csv")

# Save customer segments
segments_df = customer_clv[["customer_id", "segment", "predicted_clv_6m", "historical_revenue", "country"]]
segments_df.to_csv(os.path.join(OUTPUT_DIR, "customer_segments.csv"), index=False)
print("  Saved: outputs/customer_segments.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: MODEL VALIDATION (CALIBRATION / HOLDOUT SPLIT)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9: Model validation with time-based split...")
print("=" * 60)

# Split: Use first ~21 months for training, last 3 months for testing
end_date = df["InvoiceDate"].max()
split_date = end_date - pd.DateOffset(months=3)
print(f"  Data range:        {df['InvoiceDate'].min().date()} to {end_date.date()}")
print(f"  Calibration end:   {split_date.date()}")
print(f"  Holdout start:     {(split_date + pd.DateOffset(days=1)).date()}")

calibration_df = df[df["InvoiceDate"] <= split_date]
holdout_df = df[df["InvoiceDate"] > split_date]

print(f"  Calibration rows:  {len(calibration_df):,}")
print(f"  Holdout rows:      {len(holdout_df):,}")

# Build calibration summary
cal_summary = summary_data_from_transaction_data(
    calibration_df,
    customer_id_col="Customer ID",
    datetime_col="InvoiceDate",
    monetary_value_col="TotalPrice",
    observation_period_end=split_date,
)

# Build holdout actual spending
holdout_actual = holdout_df.groupby("Customer ID")["TotalPrice"].sum().reset_index()
holdout_actual.columns = ["Customer ID", "actual_spend"]

# Train lifetimes models on calibration data
bgf_cal = BetaGeoFitter(penalizer_coef=0.001)
bgf_cal.fit(cal_summary["frequency"], cal_summary["recency"], cal_summary["T"])

ggf_cal = GammaGammaFitter(penalizer_coef=0.001)
cal_repeat = cal_summary[cal_summary["frequency"] > 0]
ggf_cal.fit(cal_repeat["frequency"], cal_repeat["monetary_value"])

# Predict CLV for holdout period (3 months)
cal_summary["lifetimes_predicted_spend"] = ggf_cal.customer_lifetime_value(
    bgf_cal,
    cal_summary["frequency"],
    cal_summary["recency"],
    cal_summary["T"],
    cal_summary["monetary_value"],
    time=3,
    discount_rate=0.01,
)

# Merge with actual holdout spending
validation_df = cal_summary.reset_index().merge(holdout_actual, on="Customer ID", how="left")
validation_df["actual_spend"] = validation_df["actual_spend"].fillna(0)

# Calculate Lifetimes metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

lifetimes_mae = mean_absolute_error(validation_df["actual_spend"], validation_df["lifetimes_predicted_spend"])
lifetimes_rmse = np.sqrt(mean_squared_error(validation_df["actual_spend"], validation_df["lifetimes_predicted_spend"]))

print(f"\n  Lifetimes Model (BG/NBD + Gamma-Gamma):")
print(f"    MAE:  £{lifetimes_mae:,.2f}")
print(f"    RMSE: £{lifetimes_rmse:,.2f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: XGBOOST MODEL (COMPARISON)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 10: Training XGBoost model for comparison...")
print("=" * 60)

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Create features from calibration data
features = cal_summary.reset_index().copy()

# Add extra features from calibration transactions
# Average days between purchases
def avg_days_between(dates_list):
    if len(dates_list) > 1:
        dates_sorted = sorted(dates_list)
        diffs = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
        return np.mean(diffs)
    return 0

customer_dates = calibration_df.groupby("Customer ID")["InvoiceDate"].apply(list)
features["avg_days_between_purchases"] = features["Customer ID"].map(
    customer_dates.apply(avg_days_between)
).fillna(0)

# Standard deviation of spending
spend_std = calibration_df.groupby("Customer ID")["TotalPrice"].std().fillna(0)
features["spend_std"] = features["Customer ID"].map(spend_std).fillna(0)

# Number of unique products purchased
unique_products = calibration_df.groupby("Customer ID")["StockCode"].nunique()
features["unique_products"] = features["Customer ID"].map(unique_products).fillna(0)

# Merge with holdout actual
xgb_df = features.merge(holdout_actual, on="Customer ID", how="left")
xgb_df["actual_spend"] = xgb_df["actual_spend"].fillna(0)

# Prepare X and y
feature_cols = ["frequency", "recency", "T", "monetary_value",
                "avg_days_between_purchases", "spend_std", "unique_products"]
X = xgb_df[feature_cols]
y = xgb_df["actual_spend"]

# Split for validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    early_stopping_rounds=10,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predict on ALL customers
xgb_df["xgb_predicted_spend"] = xgb_model.predict(X)

# Calculate XGBoost metrics
xgb_mae = mean_absolute_error(xgb_df["actual_spend"], xgb_df["xgb_predicted_spend"])
xgb_rmse = np.sqrt(mean_squared_error(xgb_df["actual_spend"], xgb_df["xgb_predicted_spend"]))

print(f"  XGBoost Model:")
print(f"    MAE:  £{xgb_mae:,.2f}")
print(f"    RMSE: £{xgb_rmse:,.2f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: SAVE METRICS AND VALIDATION DATA
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 11: Saving model metrics...")
print("=" * 60)

# Save model comparison metrics
metrics = pd.DataFrame({
    "Model": ["Lifetimes (BG/NBD + Gamma-Gamma)", "XGBoost"],
    "MAE": [round(lifetimes_mae, 2), round(xgb_mae, 2)],
    "RMSE": [round(lifetimes_rmse, 2), round(xgb_rmse, 2)],
    "Holdout_Period": ["3 months", "3 months"],
    "Best_For": ["Long-term strategic CLV estimation", "Short-term spend forecasting"],
})
metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)
print("  Saved: outputs/model_metrics.csv")

# Save validation data for the dashboard
val_save = xgb_df[["Customer ID", "actual_spend", "xgb_predicted_spend"]].copy()
val_save = val_save.merge(
    validation_df[["Customer ID", "lifetimes_predicted_spend"]],
    on="Customer ID",
    how="left",
)
val_save.to_csv(os.path.join(OUTPUT_DIR, "validation_data.csv"), index=False)
print("  Saved: outputs/validation_data.csv")

# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ALL DONE! Generated files:")
print("=" * 60)
print("  outputs/cleaned_data.csv")
print("  outputs/customer_clv.csv")
print("  outputs/customer_segments.csv")
print("  outputs/model_metrics.csv")
print("  outputs/validation_data.csv")
print("  models/bg_nbd_model.pkl")
print("  models/gamma_gamma_model.pkl")
print("\nYou can now run the Streamlit dashboard:  streamlit run app.py")
