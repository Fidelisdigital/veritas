"""
Generates synthetic wallet-behavior data for training Veritas's
Sybil/bot anomaly detector.

Features per wallet:
  - tx_frequency: transactions per day
  - timing_regularity: 0 (irregular/human-like) to 1 (perfectly regular/bot-like)
  - gas_variance: variance in gas price used across txs (low = bot-like)
  - interaction_diversity: number of distinct function types called
  - account_age_days: how old the wallet is
  - activity_burst_ratio: recent activity vs. historical average (high = suspicious spike)

Normal wallets: irregular timing, varied gas, diverse interactions, gradual activity.
Bot-like wallets: regular timing, near-identical gas, narrow interactions, sudden bursts.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def generate_normal_wallets(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "tx_frequency": np.random.gamma(shape=2.0, scale=3.0, size=n),
        "timing_regularity": np.random.beta(2, 5, size=n),  # skewed low = irregular
        "gas_variance": np.random.gamma(shape=3.0, scale=2.0, size=n),
        "interaction_diversity": np.random.poisson(lam=5, size=n) + 1,
        "account_age_days": np.random.gamma(shape=4.0, scale=40.0, size=n),
        "activity_burst_ratio": np.random.normal(loc=1.0, scale=0.3, size=n).clip(min=0.1),
    })


def generate_bot_like_wallets(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "tx_frequency": np.random.gamma(shape=6.0, scale=8.0, size=n),  # much higher
        "timing_regularity": np.random.beta(8, 2, size=n),              # skewed high = regular
        "gas_variance": np.random.gamma(shape=0.5, scale=0.3, size=n),  # near-identical gas
        "interaction_diversity": np.random.poisson(lam=1, size=n) + 1,  # narrow behavior
        "account_age_days": np.random.gamma(shape=1.0, scale=10.0, size=n),  # newer accounts
        "activity_burst_ratio": np.random.gamma(shape=5.0, scale=1.0, size=n),  # sudden spikes
    })


def generate_dataset(n_normal: int = 800, n_bot: int = 150) -> pd.DataFrame:
    normal = generate_normal_wallets(n_normal)
    normal["label"] = "normal"

    bots = generate_bot_like_wallets(n_bot)
    bots["label"] = "bot_like"

    df = pd.concat([normal, bots], ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("ml/wallet_behavior_dataset.csv", index=False)
    print(f"Generated {len(df)} synthetic wallet records")
    print(f"  Normal: {(df['label'] == 'normal').sum()}")
    print(f"  Bot-like: {(df['label'] == 'bot_like').sum()}")
    print("\nSample rows:")
    print(df.head())
