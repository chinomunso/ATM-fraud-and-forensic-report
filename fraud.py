import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title="EFCC ATM Fraud - ML Version", layout="wide")
st.title("🏧 ATM Fraud Tracker - ML + Rules Hybrid")
st.markdown("For Financial Analyst (EFCC) - Uses Isolation Forest to detect fraud")

# Sidebar
st.sidebar.header("Model Settings")
contamination = st.sidebar.slider("Expected % of fraud in data", 0.01, 0.3, 0.1)
max_amount_rule = st.sidebar.number_input("Rule: Flag if amount >", 200000)

uploaded = st.file_uploader("Upload ATM Transactions CSV", type=["csv"])
st.caption("Required columns: account_no, amount, time, atm_id, location (optional)")

# Load data
if uploaded is None:
    st.info("Using sample data with hidden fraud patterns")
    np.random.seed(42)
    normal_amounts = np.random.normal(10000, 3000, 90)
    fraud_amounts = np.random.normal(250000, 50000, 10)
    all_amounts = np.concatenate([normal_amounts, fraud_amounts])
    times = pd.date_range("2026-10-29 08:00:00", periods=100, freq="15min")
    df = pd.DataFrame({
        "account_no": np.random.choice(["001122", "004455", "009988", "007766"], 100),
        "amount": all_amounts.astype(int),
        "time": times,
        "atm_id": np.random.choice(["ATM001", "ATM002", "ATM010"], 100),
        "location": np.random.choice(["Benin", "Lagos", "Abuja"], 100)
    })
else:
    df = pd.read_csv(uploaded)

df['time'] = pd.to_datetime(df['time'])
st.dataframe(df.head(10))

if st.button("🔍 Run ML Fraud Detection"):
    # --- Feature Engineering (Financial Modeling Part) ---
    df['hour'] = df['time'].dt.hour
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 23 or x <= 4 else 0)
    
    # Frequency feature
    df['date'] = df['time'].dt.date
    daily_counts = df.groupby(['account_no', 'date']).size().reset_index(name='txn_count')
    df = df.merge(daily_counts, on=['account_no', 'date'], how='left')

    # Amount deviation from account average (Forensic Accounting)
    avg_amount = df.groupby('account_no')['amount'].mean().reset_index(name='avg_amount')
    df = df.merge(avg_amount, on='account_no', how='left')
    df['amount_vs_avg'] = df['amount'] / df['avg_amount']

    # --- ML Model ---
    features = df[['amount', 'hour', 'txn_count', 'amount_vs_avg', 'is_night']]
    features = features.fillna(0)

    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(features)  # -1 = fraud, 1 = normal
    df['fraud_score'] = model.decision_function(features) # lower = more fraudulent

    df['ml_status'] = df['anomaly'].apply(lambda x: "🔴 ML FLAGGED" if x == -1 else "🟢 Normal")
    df['rule_status'] = df['amount'].apply(lambda x: "🔴 Rule Flag" if x > max_amount_rule else "OK")

    # Final Verdict - Hybrid
    df['final_status'] = np.where((df['anomaly'] == -1) | (df['amount'] > max_amount_rule), "🔴 HIGH RISK - Investigate", "🟢 OK")

    st.success(f"Checked {len(df)} transactions")
    
    # --- Visuals ---
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x='time', y='amount', color='final_status', 
                         hover_data=['account_no', 'fraud_score'], title="Fraud vs Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fraud_only = df[df['final_status'].str.contains("HIGH RISK")]
        st.metric("Total High Risk", len(fraud_only))
        st.metric("Total Amount at Risk", f"₦{fraud_only['amount'].sum():,}")
        st.dataframe(fraud_only[['account_no', 'amount', 'time', 'location', 'amount_vs_avg', 'final_status']].sort_values('amount', ascending=False))

    # Full report
    st.subheader("Forensic Report for EFCC Case File")
    st.dataframe(df.sort_values('fraud_score'))

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full ML Report", csv, "efcc_ml_fraud_report.csv", "text/csv")

    st.markdown("---")
    st.info("""
    **How this works (for interview):**
    1.  **Feature Engineering:** We create `amount_vs_avg` to see if user withdraws 10x his normal.
    2.  **Isolation Forest:** Algorithm isolates outliers. Fraudsters behave differently, so they are easy to isolate.
    3.  **fraud_score:** The lower (more negative), the more suspicious.
    4.  **Hybrid:** We combine ML + Human Rule (amount > limit) because EFCC needs evidence, not just AI guess.
    """)
