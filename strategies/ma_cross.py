"""移動平均クロス戦略: 短期MAが長期MAを上抜けでロング、下抜けでショート。"""

import pandas as pd


def generate_signal(close: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(float) - (fast_ma < slow_ma).astype(float)
    return signal.fillna(0.0)
