"""
Trains an Isolation Forest anomaly detector on synthetic wallet behavior data,
then validates it correctly separates normal from bot-like wallets.

Run after generate_synthetic_wallets.py has produced wallet_behavior_dataset.csv.
"""

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "tx_frequency",
    "timing_regularity",
    "gas_variance",
    "interaction_diversity",
    "account_age_days",
    "activity_burst_ratio",
]


def train_model(csv_path: str = "ml/wallet_behavior_dataset.csv"):
    df = pd.read_csv(csv_path)

    # Train ONLY on normal wallets -- Isolation Forest learns what "normal" looks like,
    # then flags anything that deviates as anomalous (this is the correct usage pattern:
    # we don't want the model to "learn" bot behavior as acceptable).
    normal_df = df[df["label"] == "normal"]
    X_train = normal_df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,  # expect ~5% noise even within "normal" data
        random_state=42,
    )
    model.fit(X_train_scaled)

    joblib.dump(model, "ml/isolation_forest_model.pkl")
    joblib.dump(scaler, "ml/feature_scaler.pkl")
    print("Model and scaler saved to ml/")

    # Validate: check how well it separates normal vs bot-like on the FULL dataset
    X_all_scaled = scaler.transform(df[FEATURE_COLUMNS])
    predictions = model.predict(X_all_scaled)  # 1 = normal, -1 = anomaly
    df["predicted_anomaly"] = predictions == -1

    print("\n--- Validation ---")
    for label in ["normal", "bot_like"]:
        subset = df[df["label"] == label]
        flagged_pct = subset["predicted_anomaly"].mean() * 100
        print(f"{label:10s}: {flagged_pct:.1f}% flagged as anomalous ({subset['predicted_anomaly'].sum()}/{len(subset)})")

    return model, scaler


def get_anomaly_score(wallet_features: dict) -> float:
    """
    Loads the trained model and returns an anomaly score for a single wallet.

    Args:
        wallet_features: dict with keys matching FEATURE_COLUMNS

    Returns:
        float in roughly [-0.5, 0.5] -- more negative = more anomalous.
        (This matches sklearn's IsolationForest.decision_function convention.)
    """
    model = joblib.load("ml/isolation_forest_model.pkl")
    scaler = joblib.load("ml/feature_scaler.pkl")

    row = pd.DataFrame([wallet_features])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    score = model.decision_function(row_scaled)[0]
    return float(score)


if __name__ == "__main__":
    train_model()

    print("\n--- Example single-wallet scoring ---")
    normal_example = {
        "tx_frequency": 5.2,
        "timing_regularity": 0.25,
        "gas_variance": 4.1,
        "interaction_diversity": 6,
        "account_age_days": 150,
        "activity_burst_ratio": 1.05,
    }
    bot_example = {
        "tx_frequency": 48.0,
        "timing_regularity": 0.92,
        "gas_variance": 0.1,
        "interaction_diversity": 1,
        "account_age_days": 8,
        "activity_burst_ratio": 6.5,
    }
    print("Normal-looking wallet score:", get_anomaly_score(normal_example))
    print("Bot-looking wallet score:   ", get_anomaly_score(bot_example))
    print("(more negative = more anomalous)")
