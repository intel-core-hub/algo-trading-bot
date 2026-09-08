"""モメンタム戦略: 直近N期間のリターンが閾値を超えたらその方向にロング/ショート。"""

import pandas as pd


def generate_signal(close: pd.Series, lookback: int = 24, threshold: float = 0.0) -> pd.Series:
    momentum = close.pct_change(lookback)
    signal = pd.Series(0.0, index=close.index)
    signal[momentum > threshold] = 1.0
    signal[momentum < -threshold] = -1.0
    return signal.fillna(0.0)
