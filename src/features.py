"""OHLCVからML用の特徴量とラベルを作る。

前提: 特徴量は時刻tまでのデータのみ(rolling系はshift不要、pandasのrollingは
その時点までのウィンドウなので未来を見ない)。ラベルだけは未来のリターンを使う
(予測対象なので当然)。学習時にラベルを特徴量として使わない限りリークしない。
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "vol_24",
    "vol_72",
    "ma_ratio_20",
    "ma_ratio_50",
    "rsi_14",
    "volume_z_24",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    volume = df["volume"]
    returns = close.pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = close.pct_change(1)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_6"] = close.pct_change(6)
    feat["ret_12"] = close.pct_change(12)
    feat["ret_24"] = close.pct_change(24)
    feat["vol_24"] = returns.rolling(24).std()
    feat["vol_72"] = returns.rolling(72).std()
    feat["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    feat["ma_ratio_50"] = close / close.rolling(50).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = (100 - 100 / (1 + rs)).fillna(50.0)

    feat["volume_z_24"] = (volume - volume.rolling(24).mean()) / volume.rolling(24).std()

    return feat[FEATURE_COLUMNS]


def build_labels(close: pd.Series, horizon: int = 12, threshold: float = 0.005) -> pd.Series:
    """horizon本先までの将来リターンが+threshold超なら1、-threshold未満なら-1、それ以外は0。"""
    forward_return = close.shift(-horizon) / close - 1
    label = pd.Series(0.0, index=close.index, dtype=float)
    label[forward_return > threshold] = 1.0
    label[forward_return < -threshold] = -1.0
    label[forward_return.isna()] = np.nan
    return label
