"""長期トレンドフォロー: 日足換算の移動平均クロス(ゴールデンクロス/デッドクロス相当)。

これまでのma_cross/momentum/mean_reversionは1h足そのものにシグナルを立てるため
取引回数が多く、手数料・スリッページの影響を強く受けていた。本戦略は入力が1h足で
あっても日足にリサンプルしてからクロス判定することで、保有期間を数十〜数百日単位に
伸ばし、取引回数を極端に減らすことでコスト負けを避けられるかを検証する。
"""

import pandas as pd


def generate_signal(close: pd.Series, fast_days: int = 50, slow_days: int = 200) -> pd.Series:
    daily_close = close.resample("1D").last().dropna()
    fast_ma = daily_close.rolling(fast_days).mean()
    slow_ma = daily_close.rolling(slow_days).mean()
    daily_signal = (fast_ma > slow_ma).astype(float) - (fast_ma < slow_ma).astype(float)
    daily_signal = daily_signal.fillna(0.0)

    return daily_signal.reindex(close.index, method="ffill").fillna(0.0)
