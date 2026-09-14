"""Shared feature preparation and demonstration risk bands."""

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]
SERVICE_FEATURES = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]


def add_features(customers):
    """Reproduce notebook 04, including internet in the service count."""
    result = customers.loc[:, RAW_FEATURES].copy()
    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    new_customer = result["tenure"].eq(0) & result["TotalCharges"].isna()
    result.loc[new_customer, "TotalCharges"] = 0
    result["TenureGroup"] = np.select(
        [result["tenure"] <= 12, result["tenure"] <= 24,
         result["tenure"] <= 48, result["tenure"] <= 60],
        ["0-12 months", "13-24 months", "25-48 months", "49-60 months"],
        default="61+ months",
    )
    result["TotalServices"] = (
        result[SERVICE_FEATURES].eq("Yes").sum(axis=1)
        + result["InternetService"].ne("No").astype(int)
    )
    return result


def risk_categories(probabilities):
    """Fixed demonstration thresholds; no thresholds fitted to test labels."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")
    return pd.Categorical(
        np.select([values < 0.30, values < 0.60], RISK_LABELS[:2],
                  default=RISK_LABELS[2]),
        categories=RISK_LABELS, ordered=True,
    )
