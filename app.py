"""
Streamlit Dashboard – Customer Lifetime Value (CLV) Analytics
=============================================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load Data (cached) ────────────────────────────────────────────────────
@st.cache_data
def load_cleaned_data():
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"))
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df

@st.cache_data
def load_customer_clv():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "customer_clv.csv"))

@st.cache_data
def load_segments():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "customer_segments.csv"))

@st.cache_data
def load_metrics():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"))

@st.cache_data
def load_validation():
    return pd.read_csv(os.path.join(OUTPUT_DIR, "validation_data.csv"))

# Load all data
try:
    df = load_cleaned_data()
    customer_clv = load_customer_clv()
    segments = load_segments()
    metrics = load_metrics()
    validation = load_validation()
except FileNotFoundError as e:
    st.error(f"⚠️ Required data files not found. Please run `python generate_outputs.py` first.\n\nError: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("📊 CLV Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Executive Overview", "👥 Customer Analytics", "💰 CLV Prediction",
     "🎯 Customer Segmentation", "📈 Model Performance"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Info**")
st.sidebar.markdown(f"📅 {df['InvoiceDate'].min().strftime('%b %Y')} – {df['InvoiceDate'].max().strftime('%b %Y')}")
st.sidebar.markdown(f"👥 {customer_clv['customer_id'].nunique():,} Customers")
st.sidebar.markdown(f"📦 {df['Invoice'].nunique():,} Transactions")

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def metric_card(label, value, delta=None):
    """Display a styled metric."""
    st.metric(label=label, value=value, delta=delta)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
if page == "🏠 Executive Overview":
    st.title("🏠 Executive Overview")
    st.markdown("Key performance indicators and business health at a glance.")
    st.markdown("---")

    # KPI Row
    total_revenue = df["TotalPrice"].sum()
    total_customers = df["Customer ID"].nunique()
    total_transactions = df["Invoice"].nunique()
    avg_order_value = df.groupby("Invoice")["TotalPrice"].sum().mean()
    avg_clv = customer_clv["predicted_clv_6m"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Revenue", f"£{total_revenue:,.0f}")
    with col2:
        st.metric("Total Customers", f"{total_customers:,}")
    with col3:
        st.metric("Total Transactions", f"{total_transactions:,}")
    with col4:
        st.metric("Avg Order Value", f"£{avg_order_value:,.2f}")
    with col5:
        st.metric("Avg CLV (6 months)", f"£{avg_clv:,.2f}")

    st.markdown("---")

    # Monthly Revenue Trend
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("📈 Monthly Revenue Trend")
        monthly_rev = df.groupby(df["InvoiceDate"].dt.to_period("M"))["TotalPrice"].sum().reset_index()
        monthly_rev["InvoiceDate"] = monthly_rev["InvoiceDate"].astype(str)
        fig = px.line(monthly_rev, x="InvoiceDate", y="TotalPrice",
                      labels={"InvoiceDate": "Month", "TotalPrice": "Revenue (£)"},
                      markers=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("👥 Customer Acquisition Trend")
        # First purchase month per customer
        first_purchase = df.groupby("Customer ID")["InvoiceDate"].min().reset_index()
        first_purchase["month"] = first_purchase["InvoiceDate"].dt.to_period("M").astype(str)
        acq_trend = first_purchase.groupby("month").size().reset_index(name="new_customers")
        fig2 = px.bar(acq_trend, x="month", y="new_customers",
                      labels={"month": "Month", "new_customers": "New Customers"},
                      color_discrete_sequence=["#636EFA"])
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Segmentation Distribution
    st.subheader("🎯 Customer Segmentation Distribution")
    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = segments["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig3 = px.pie(seg_counts, values="Count", names="Segment",
                      color="Segment",
                      color_discrete_map={"High Value": "#2ecc71", "Medium Value": "#f39c12", "Low Value": "#e74c3c"})
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        seg_clv = segments.groupby("segment")["predicted_clv_6m"].mean().reset_index()
        seg_clv.columns = ["Segment", "Avg CLV (£)"]
        fig4 = px.bar(seg_clv, x="Segment", y="Avg CLV (£)",
                      color="Segment",
                      color_discrete_map={"High Value": "#2ecc71", "Medium Value": "#f39c12", "Low Value": "#e74c3c"})
        fig4.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER ANALYTICS
# ══════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Analytics":
    st.title("👥 Customer Analytics")
    st.markdown("Deep dive into customer purchasing behavior.")
    st.markdown("---")

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        countries = ["All"] + sorted(df["Country"].unique().tolist())
        selected_country = st.selectbox("Filter by Country", countries)
    with col_f2:
        date_range = st.date_input(
            "Date Range",
            value=(df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()),
            min_value=df["InvoiceDate"].min().date(),
            max_value=df["InvoiceDate"].max().date(),
        )

    # Apply filters
    filtered_df = df.copy()
    if selected_country != "All":
        filtered_df = filtered_df[filtered_df["Country"] == selected_country]
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["InvoiceDate"].dt.date >= date_range[0])
            & (filtered_df["InvoiceDate"].dt.date <= date_range[1])
        ]

    st.markdown(f"**Showing {len(filtered_df):,} transactions** | {filtered_df['Customer ID'].nunique():,} customers")
    st.markdown("---")

    # Row 1: Purchase Frequency & Revenue by Customer
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Purchase Frequency Distribution")
        cust_freq = filtered_df.groupby("Customer ID")["Invoice"].nunique().reset_index()
        cust_freq.columns = ["Customer ID", "Num_Purchases"]
        # Cap at 50 for readability
        cust_freq_capped = cust_freq.copy()
        cust_freq_capped["Num_Purchases"] = cust_freq_capped["Num_Purchases"].clip(upper=50)
        fig = px.histogram(cust_freq_capped, x="Num_Purchases", nbins=50,
                           labels={"Num_Purchases": "Number of Purchases"},
                           color_discrete_sequence=["#636EFA"])
        fig.update_layout(height=350, yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Top 10 Customers by Revenue")
        top_cust = filtered_df.groupby("Customer ID")["TotalPrice"].sum().nlargest(10).reset_index()
        top_cust.columns = ["Customer ID", "Revenue (£)"]
        top_cust["Customer ID"] = top_cust["Customer ID"].astype(str)
        fig2 = px.bar(top_cust, x="Revenue (£)", y="Customer ID", orientation="h",
                      color_discrete_sequence=["#2ecc71"])
        fig2.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: New vs Returning & Country-wise Revenue
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🔄 New vs Returning Customers")
        first_buy = filtered_df.groupby("Customer ID")["InvoiceDate"].min()
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy["first_purchase"] = filtered_df_copy["Customer ID"].map(first_buy)
        filtered_df_copy["customer_type"] = np.where(
            filtered_df_copy["InvoiceDate"] == filtered_df_copy["first_purchase"], "New", "Returning"
        )
        type_rev = filtered_df_copy.groupby("customer_type")["TotalPrice"].sum().reset_index()
        fig3 = px.pie(type_rev, values="TotalPrice", names="customer_type",
                      color_discrete_sequence=["#3498db", "#e74c3c"])
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("🌍 Top 10 Countries by Revenue")
        country_rev = filtered_df.groupby("Country")["TotalPrice"].sum().nlargest(10).reset_index()
        country_rev.columns = ["Country", "Revenue (£)"]
        fig4 = px.bar(country_rev, x="Revenue (£)", y="Country", orientation="h",
                      color_discrete_sequence=["#f39c12"])
        fig4.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, use_container_width=True)

    # Customer Retention Trends
    st.markdown("---")
    st.subheader("📅 Monthly Active Customers")
    monthly_active = filtered_df.groupby(filtered_df["InvoiceDate"].dt.to_period("M"))["Customer ID"].nunique().reset_index()
    monthly_active["InvoiceDate"] = monthly_active["InvoiceDate"].astype(str)
    monthly_active.columns = ["Month", "Active Customers"]
    fig5 = px.line(monthly_active, x="Month", y="Active Customers", markers=True,
                   color_discrete_sequence=["#9b59b6"])
    fig5.update_layout(height=350)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: CLV PREDICTION
# ══════════════════════════════════════════════════════════════════════════
elif page == "💰 CLV Prediction":
    st.title("💰 CLV Prediction")
    st.markdown("Predicted Customer Lifetime Value using BG/NBD + Gamma-Gamma models.")
    st.markdown("---")

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg CLV (6 months)", f"£{customer_clv['predicted_clv_6m'].mean():,.2f}")
    with col2:
        st.metric("Total Predicted CLV", f"£{customer_clv['predicted_clv_6m'].sum():,.0f}")
    with col3:
        high_val = len(segments[segments["segment"] == "High Value"])
        st.metric("High Value Customers", f"{high_val:,}")
    with col4:
        avg_alive = customer_clv["prob_alive"].mean()
        st.metric("Avg Prob. Alive", f"{avg_alive:.1%}")

    st.markdown("---")

    # CLV Distribution & Top Customers
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 CLV Distribution")
        # Cap CLV for better visualization
        clv_capped = customer_clv["predicted_clv_6m"].clip(upper=customer_clv["predicted_clv_6m"].quantile(0.95))
        fig = px.histogram(clv_capped, nbins=50,
                           labels={"value": "Predicted CLV (£)"},
                           color_discrete_sequence=["#2ecc71"])
        fig.update_layout(height=350, showlegend=False, yaxis_title="Number of Customers")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🏆 Top 10 Customers by Predicted CLV")
        top_clv = customer_clv.nlargest(10, "predicted_clv_6m")[["customer_id", "predicted_clv_6m", "segment"]]
        top_clv_disp = top_clv.copy()
        top_clv_disp["customer_id"] = top_clv_disp["customer_id"].astype(str)
        fig2 = px.bar(top_clv_disp, x="predicted_clv_6m", y="customer_id", orientation="h",
                      color="segment",
                      color_discrete_map={"High Value": "#2ecc71", "Medium Value": "#f39c12", "Low Value": "#e74c3c"},
                      labels={"predicted_clv_6m": "Predicted CLV (£)", "customer_id": "Customer ID"})
        fig2.update_layout(height=350, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

    # Segment counts
    st.subheader("🎯 Customer Counts by Segment")
    seg_counts = segments["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    col_c, col_d, col_e = st.columns(3)
    for i, row in seg_counts.iterrows():
        col = [col_c, col_d, col_e][i % 3]
        with col:
            color_map = {"High Value": "🟢", "Medium Value": "🟡", "Low Value": "🔴"}
            emoji = color_map.get(row["Segment"], "⚪")
            st.metric(f"{emoji} {row['Segment']}", f"{row['Count']:,}")

    # Customer Search
    st.markdown("---")
    st.subheader("🔍 Customer Lookup")
    customer_ids = sorted(customer_clv["customer_id"].unique())
    selected_customer = st.selectbox("Select a Customer ID", customer_ids)

    if selected_customer:
        cust_data = customer_clv[customer_clv["customer_id"] == selected_customer].iloc[0]
        col_x, col_y, col_z = st.columns(3)
        with col_x:
            st.markdown("**Purchasing Summary**")
            st.write(f"- **Recency:** {cust_data['recency']:.0f} days")
            st.write(f"- **Frequency:** {cust_data['frequency']:.0f} purchases")
            st.write(f"- **Customer Age (T):** {cust_data['T']:.0f} days")
            st.write(f"- **Avg Transaction Value:** £{cust_data['monetary_value']:,.2f}")
        with col_y:
            st.markdown("**Predictions**")
            st.write(f"- **Predicted CLV (6 months):** £{cust_data['predicted_clv_6m']:,.2f}")
            st.write(f"- **Expected Purchases (6 months):** {cust_data['predicted_purchases_6m']:.1f}")
            st.write(f"- **Predicted Avg Transaction:** £{cust_data['predicted_monetary_value']:,.2f}")
            st.write(f"- **Probability Alive:** {cust_data['prob_alive']:.1%}")
        with col_z:
            st.markdown("**Segment & History**")
            st.write(f"- **Segment:** {cust_data['segment']}")
            st.write(f"- **Historical Revenue:** £{cust_data['historical_revenue']:,.2f}")
            st.write(f"- **Country:** {cust_data['country']}")

    # Full CLV Table
    st.markdown("---")
    st.subheader("📋 Customer CLV Table")
    display_cols = ["customer_id", "frequency", "recency", "T", "monetary_value",
                    "predicted_clv_6m", "predicted_purchases_6m", "prob_alive", "segment", "country"]
    available_cols = [c for c in display_cols if c in customer_clv.columns]
    st.dataframe(
        customer_clv[available_cols].sort_values("predicted_clv_6m", ascending=False).head(100),
        use_container_width=True,
        height=400,
    )


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════
elif page == "🎯 Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    st.markdown("Customers segmented into High, Medium, and Low value tiers based on predicted CLV.")
    st.markdown("---")

    # Segment Overview
    col1, col2, col3 = st.columns(3)
    for i, seg in enumerate(["High Value", "Medium Value", "Low Value"]):
        seg_data = segments[segments["segment"] == seg]
        avg_clv = seg_data["predicted_clv_6m"].mean()
        count = len(seg_data)
        total_rev = seg_data["historical_revenue"].sum()
        emoji = ["🟢", "🟡", "🔴"][i]
        col = [col1, col2, col3][i]
        with col:
            st.markdown(f"### {emoji} {seg}")
            st.metric("Customers", f"{count:,}")
            st.metric("Avg CLV (6 months)", f"£{avg_clv:,.2f}")
            st.metric("Total Historical Revenue", f"£{total_rev:,.0f}")

    st.markdown("---")

    # Charts
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📊 Avg CLV by Segment")
        seg_avg = segments.groupby("segment")["predicted_clv_6m"].mean().reset_index()
        seg_avg.columns = ["Segment", "Avg CLV (£)"]
        fig = px.bar(seg_avg, x="Segment", y="Avg CLV (£)",
                     color="Segment",
                     color_discrete_map={"High Value": "#2ecc71", "Medium Value": "#f39c12", "Low Value": "#e74c3c"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("💰 Revenue Contribution by Segment")
        seg_rev = segments.groupby("segment")["historical_revenue"].sum().reset_index()
        seg_rev.columns = ["Segment", "Total Revenue (£)"]
        fig2 = px.pie(seg_rev, values="Total Revenue (£)", names="Segment",
                      color="Segment",
                      color_discrete_map={"High Value": "#2ecc71", "Medium Value": "#f39c12", "Low Value": "#e74c3c"})
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Segment Purchase Patterns
    st.markdown("---")
    st.subheader("📈 Segment-wise Purchase Patterns")
    seg_patterns = customer_clv.groupby("segment").agg(
        avg_frequency=("frequency", "mean"),
        avg_recency=("recency", "mean"),
        avg_monetary=("monetary_value", "mean"),
        avg_prob_alive=("prob_alive", "mean"),
    ).reset_index()

    st.dataframe(seg_patterns.style.format({
        "avg_frequency": "{:.1f}",
        "avg_recency": "{:.0f} days",
        "avg_monetary": "£{:,.2f}",
        "avg_prob_alive": "{:.1%}",
    }), use_container_width=True)

    # Marketing Recommendations
    st.markdown("---")
    st.subheader("💡 Marketing Recommendations")

    with st.expander("🟢 High Value Customers – Retention Strategy", expanded=True):
        st.markdown("""
        - **Loyalty Programs**: Offer VIP rewards, early access to new products, and exclusive discounts.
        - **Personal Account Management**: Assign dedicated account managers for the top customers.
        - **Feedback Loops**: Regularly collect feedback to improve their experience.
        - **Cross-sell and Upsell**: Recommend complementary products based on purchase history.
        """)

    with st.expander("🟡 Medium Value Customers – Growth Strategy"):
        st.markdown("""
        - **Targeted Promotions**: Offer quantity-based discounts to increase order size.
        - **Engagement Campaigns**: Send personalised email campaigns highlighting popular products.
        - **Referral Programs**: Incentivise referrals to expand their network and yours.
        - **Bundle Offers**: Create product bundles to increase average order value.
        """)

    with st.expander("🔴 Low Value Customers – Re-engagement Strategy"):
        st.markdown("""
        - **Win-back Campaigns**: Send re-engagement emails with special offers.
        - **Survey for Feedback**: Understand why they purchase infrequently.
        - **Entry-level Offers**: Provide low-commitment offers to encourage repeat purchases.
        - **Content Marketing**: Share useful content to maintain brand awareness.
        """)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Comparison of BG/NBD + Gamma-Gamma (Lifetimes) and XGBoost models.")
    st.markdown("---")

    # Metrics Table
    st.subheader("📊 Model Comparison")
    st.dataframe(metrics.style.format({"MAE": "£{:,.2f}", "RMSE": "£{:,.2f}"}), use_container_width=True)

    # Visual comparison
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("MAE Comparison")
        fig = px.bar(metrics, x="Model", y="MAE",
                     color="Model",
                     color_discrete_sequence=["#3498db", "#2ecc71"],
                     labels={"MAE": "Mean Absolute Error (£)"})
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("RMSE Comparison")
        fig2 = px.bar(metrics, x="Model", y="RMSE",
                      color="Model",
                      color_discrete_sequence=["#3498db", "#2ecc71"],
                      labels={"RMSE": "Root Mean Squared Error (£)"})
        fig2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Actual vs Predicted Scatter
    st.markdown("---")
    st.subheader("🔍 Actual vs Predicted Spending (Holdout Period)")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Lifetimes Model**")
        # Sample to avoid overplotting
        sample_val = validation.sample(min(2000, len(validation)), random_state=42)
        fig3 = px.scatter(sample_val, x="actual_spend", y="lifetimes_predicted_spend",
                          labels={"actual_spend": "Actual Spend (£)", "lifetimes_predicted_spend": "Predicted Spend (£)"},
                          opacity=0.4, color_discrete_sequence=["#3498db"])
        # Add diagonal line
        max_val = max(sample_val["actual_spend"].max(), sample_val["lifetimes_predicted_spend"].max())
        fig3.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines",
                                  line=dict(dash="dash", color="red"), name="Perfect prediction"))
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown("**XGBoost Model**")
        fig4 = px.scatter(sample_val, x="actual_spend", y="xgb_predicted_spend",
                          labels={"actual_spend": "Actual Spend (£)", "xgb_predicted_spend": "Predicted Spend (£)"},
                          opacity=0.4, color_discrete_sequence=["#2ecc71"])
        max_val2 = max(sample_val["actual_spend"].max(), sample_val["xgb_predicted_spend"].max())
        fig4.add_trace(go.Scatter(x=[0, max_val2], y=[0, max_val2], mode="lines",
                                  line=dict(dash="dash", color="red"), name="Perfect prediction"))
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # Model Limitations
    st.markdown("---")
    st.subheader("⚠️ Model Limitations")

    st.markdown("""
    **BG/NBD + Gamma-Gamma (Lifetimes):**
    - Assumes customers are either "alive" or permanently "dead" — no temporary inactivity.
    - Requires non-contractual, continuous transaction settings.
    - Gamma-Gamma model requires that purchase frequency and monetary value are independent.
    - Works best with longer observation windows and repeat purchasers.

    **XGBoost:**
    - Requires careful feature engineering to avoid data leakage.
    - Performance heavily depends on the quality of features.
    - Random train/test split within the dataset was used for early stopping, though the overall split is chronological.
    - Prone to overfitting on small datasets.

    **General:**
    - Both models predict spending, not churn directly (probability alive is an estimate, not a classification).
    - External factors (economic conditions, competition) are not captured.
    - Predictions are most reliable for the near-term (3–6 months).
    """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")
st.sidebar.markdown("CLV Analytics Dashboard v1.0")
