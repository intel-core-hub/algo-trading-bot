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

    # resample("1D").last()は「その日の最後に観測できた値」を日付Dのキーに入れるが、
    # それが確定するのは1h足の当日分がすべて出そろうD日の終わり(=23時台)であり、
    # D日の朝の時点ではまだ知り得ない。shift(1)でシグナルの有効化を丸1日遅らせ、
    # D日の確定足で計算したシグナルはD+1日以降にしか使わないようにする
    # (同日中の未確定足を参照する先読みバイアスを防ぐ)。
    daily_signal = daily_signal.shift(1).fillna(0.0)

    return daily_signal.reindex(close.index, method="ffill").fillna(0.0)
