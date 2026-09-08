"""ファンディングレート逆張り戦略: 無期限先物の資金調達率が極端に振れたら反対に張る。

資金調達率がプラスに大きい = ロングがショートに支払っている = ロングが過熱している
状態なので、平均回帰的に反落を狙ってショートする(逆もまた然り)。価格そのものではなく
デリバティブ市場のポジショニング情報を使う点が、これまでのテクニカル指標ベース戦略と
質的に異なる。
"""

import pandas as pd


def generate_signal(
    funding_rate: pd.Series,
    window: int = 90,
    entry_z: float = 1.5,
    exit_z: float = 0.5,
) -> pd.Series:
    rolling_mean = funding_rate.rolling(window).mean()
    rolling_std = funding_rate.rolling(window).std()
    zscore = (funding_rate - rolling_mean) / rolling_std

    raw_signal = pd.Series(index=funding_rate.index, dtype=float)
    raw_signal[zscore > entry_z] = -1.0  # ロング過熱 -> ショート
    raw_signal[zscore < -entry_z] = 1.0  # ショート過熱 -> ロング
    raw_signal[zscore.abs() < exit_z] = 0.0

    return raw_signal.ffill().fillna(0.0)
