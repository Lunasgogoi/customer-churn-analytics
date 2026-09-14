"""Run from the project root with: streamlit run app/app.py."""

from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.churn import RAW_FEATURES, RISK_LABELS, add_features, risk_categories

st.set_page_config(page_title="Customer Churn Analytics", page_icon="📊", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/tuned_gradient_boosting.joblib")


@st.cache_data
def load_data():
    processed = ROOT / "data/processed"
    return (
        pd.read_csv(processed / "telco_churn_cleaned.csv"),
        pd.read_csv(processed / "customer_risk_segments.csv"),
        pd.read_csv(processed / "gradient_boosting_importance.csv"),
        pd.read_csv(processed / "high_risk_customers.csv"),
        json.loads((ROOT / "models/metadata.json").read_text(encoding="utf-8")),
    )


def bar_chart(values, title, xlabel, ylabel, percent=False):
    fig, ax = plt.subplots(figsize=(7, 3.7))
    ax.barh(values.index.astype(str), values.values, color="#328482")
    ax.invert_yaxis()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if percent:
        ax.set_xlim(0, 100)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


st.title("Customer Churn Analytics")
st.caption("IBM Telco sample · Understand churn patterns and explore customer risk")
try:
    model = load_model()
    customers, risks, importance, priority, metadata = load_data()
except (FileNotFoundError, ModuleNotFoundError) as error:
    st.error("Project artifacts are missing. Run notebooks 01–06 in order, then restart the app.")
    st.caption(str(error))
    st.stop()

analytics, prediction = st.tabs(["Analytics Dashboard", "Customer Churn Prediction"])

with analytics:
    metrics = metadata["metrics"][0]
    high_count = int(risks["RiskCategory"].eq("High Risk").sum())
    cards = st.columns(4)
    cards[0].metric("Total Customers", f"{len(customers):,}")
    cards[1].metric("Overall Churn Rate", f"{customers['ChurnFlag'].mean():.2%}")
    cards[2].metric("High Risk Customers (test set)", f"{high_count:,}")
    cards[3].metric("Final Model ROC-AUC (test set)", f"{metrics['ROC-AUC']:.4f}")
    st.caption(
        f"EDA uses all {len(customers):,} historical customers. Risk segments and model metrics "
        f"use only {len(risks):,} held-out customers. This is a historical demonstration, not a live campaign list."
    )
    left, right = st.columns(2)
    with left:
        bar_chart(customers["Churn"].value_counts(), "Churn distribution", "Customers", "Churn")
        bar_chart(customers.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values(ascending=False),
                  "Churn by contract", "Churn rate (%)", "Contract", True)
        bar_chart(customers.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).sort_values(ascending=False),
                  "Churn by payment method", "Churn rate (%)", "Payment method", True)
    with right:
        tenure_order = ["0-12 months", "13-24 months", "25-48 months", "49-60 months", "61+ months"]
        tenure_rates = customers.assign(TenureGroup=add_features(customers)["TenureGroup"]).groupby(
            "TenureGroup")["ChurnFlag"].mean().mul(100).reindex(tenure_order)
        bar_chart(tenure_rates, "Churn by tenure", "Churn rate (%)", "Tenure", True)
        bar_chart(customers.groupby("InternetService")["ChurnFlag"].mean().mul(100).sort_values(ascending=False),
                  "Churn by internet service", "Churn rate (%)", "Internet service", True)
        bar_chart(risks["RiskCategory"].value_counts().reindex(RISK_LABELS, fill_value=0),
                  "Held-out risk distribution", "Customers", "Risk category")
    st.subheader("Risk segments")
    summary = risks.groupby("RiskCategory").agg(
        Customers=("customerID", "size"), ObservedChurnRate=("ActualChurn", "mean")
    ).reindex(RISK_LABELS)
    summary["Share"] = summary["Customers"] / len(risks)
    st.dataframe(summary.style.format({"ObservedChurnRate": "{:.1%}", "Share": "{:.1%}"}), width="stretch")
    st.caption("Demo bands: Low <30%; Medium 30–<60%; High ≥60%. Validate thresholds against budget, error costs and customer lifetime value.")
    left, right = st.columns(2)
    with left:
        bar_chart(importance.head(12).set_index("Feature")["Importance"],
                  "Top churn drivers — Gradient Boosting", "Feature importance", "Feature")
    with right:
        st.markdown("""
**What the analysis supports**

- Month-to-month customers have higher observed churn than longer-contract customers.
- Churned customers have shorter average tenure and higher average monthly charges.
- Fiber optic and electronic-check groups have higher observed churn.
- Customers with Online Security or Tech Support have lower observed churn than those without.

Feature importance measures predictive usefulness; it provides neither direction nor
causal effects. Correlated charges and service choices complicate coefficient interpretation.
These findings suggest retention experiments, not proven interventions.
""")
    st.subheader("Highest-risk customers")
    st.dataframe(priority.head(20).style.format({"ChurnProbability": "{:.1%}", "MonthlyCharges": "{:.2f}"}), hide_index=True, width="stretch")
    st.download_button("Download high-risk queue", priority.to_csv(index=False),
                       "high_risk_customers.csv", "text/csv")
    st.download_button("Download held-out risk scores", risks.to_csv(index=False),
                       "customer_risk_segments.csv", "text/csv")

with prediction:
    st.subheader("Explore an individual customer's churn risk")
    st.caption("Enter original customer details. Tenure groups and service counts are calculated automatically.")
    details = {}
    first, second, third = st.columns(3)
    with first:
        st.markdown("**Customer and account**")
        details["gender"] = st.selectbox("Gender", ["Female", "Male"], key="gender")
        details["SeniorCitizen"] = int(st.selectbox("Senior citizen", ["No", "Yes"], key="SeniorCitizen") == "Yes")
        for field in ["Partner", "Dependents"]:
            details[field] = st.selectbox(field, ["No", "Yes"], key=field)
        details["tenure"] = st.number_input("Tenure (months)", 0, 72, 12, key="tenure")
        details["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="Contract")
    with second:
        st.markdown("**Services**")
        details["PhoneService"] = st.selectbox("Phone service", ["Yes", "No"], key="PhoneService")
        phone_options = ["No", "Yes"] if details["PhoneService"] == "Yes" else ["No phone service"]
        details["MultipleLines"] = st.selectbox("Multiple lines", phone_options, key="MultipleLines")
        details["InternetService"] = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"], key="InternetService")
        service_options = ["No", "Yes"] if details["InternetService"] != "No" else ["No internet service"]
        for field, label in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                             ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
                             ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]:
            details[field] = st.selectbox(label, service_options, key=field)
    with third:
        st.markdown("**Billing**")
        details["PaperlessBilling"] = st.selectbox("Paperless billing", ["Yes", "No"], key="PaperlessBilling")
        details["PaymentMethod"] = st.selectbox("Payment method", ["Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"], key="PaymentMethod")
        details["MonthlyCharges"] = st.number_input("Monthly charges", min_value=0.0, value=65.0, step=0.05, key="MonthlyCharges")
        details["TotalCharges"] = st.number_input("Total charges", min_value=0.0, value=780.0, step=0.05, key="TotalCharges")
        st.caption("Use the customer's actual accumulated total; tenure × monthly charges may differ after plan changes.")
    if st.button("Predict churn", type="primary"):
        row = pd.DataFrame([details], columns=RAW_FEATURES)
        probability = float(model.predict_proba(row)[0, 1])
        category = str(risk_categories([probability])[0])
        cards = st.columns(3)
        cards[0].metric("Churn probability", f"{probability:.1%}")
        cards[1].metric("Predicted class", "Churn" if probability >= 0.50 else "No churn")
        cards[2].metric("Risk category", category.upper())
        st.caption("Class labels use 50%; High Risk starts at 60%. Model scores are estimates and have not been calibrated for business use.")
        factors = []
        for condition, label in [
            (details["Contract"] == "Month-to-month", "a month-to-month contract"),
            (details["tenure"] <= 12, "short tenure"),
            (details["InternetService"] == "Fiber optic", "fiber optic internet"),
            (details["OnlineSecurity"] == "No", "no online security"),
            (details["TechSupport"] == "No", "no technical support"),
            (details["PaymentMethod"] == "Electronic check", "electronic-check payment"),
        ]:
            if condition:
                factors.append(label)
        if factors:
            st.info("This customer shares characteristics associated with higher observed churn in the dataset: "
                    + ", ".join(factors) + ".")
        else:
            st.info("This customer does not match the highlighted higher-churn patterns. The model still considers all supplied features.")
        st.caption("This context uses global dataset findings and customer characteristics; it is not a local feature attribution or causal explanation.")
        st.markdown("Treat risk bands as a starting point. Set campaign thresholds using retention budget, false-positive/false-negative costs and customer lifetime value.")
