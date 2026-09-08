"""平均回帰戦略: 価格のZスコアが極端に振れたら反対方向に張り、平均に戻ったら手仕舞う。

- Zスコア = (close - rolling_mean) / rolling_std (ボリンジャーバンドと同じ考え方)。
- Zスコアが -entry_z を下回ったらロング(売られ過ぎ、反発を狙う)。
- Zスコアが +entry_z を上回ったらショート(買われ過ぎ、反落を狙う)。
- |Zスコア| が exit_z を下回ったら(平均近くに戻ったら)手仕舞ってノーポジ。
- それ以外のバーはシグナルを維持(状態を持たせるため forward-fill で表現)。
"""

import pandas as pd


def generate_signal(
    close: pd.Series,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    zscore = (close - rolling_mean) / rolling_std

    raw_signal = pd.Series(index=close.index, dtype=float)
    raw_signal[zscore < -entry_z] = 1.0
    raw_signal[zscore > entry_z] = -1.0
    raw_signal[zscore.abs() < exit_z] = 0.0

    return raw_signal.ffill().fillna(0.0)
