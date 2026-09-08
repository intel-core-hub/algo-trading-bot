"""ファンディング・キャリー(cash-and-carry): 現物ロング + 無期限先物ショートで
デルタニュートラルを保ちつつ、資金調達率がプラスの間だけそれを受け取り続ける。

これまでのfunding_contrarian(価格の方向を資金調達率から逆張り予測する)とは違い、
本戦略は価格方向を一切予測しない。ポジションはデルタニュートラル(現物と先物の
価格変動はほぼ相殺される)なので、リターンの源泉は資金調達率そのもの
(ロングがショートに支払う分を、ショート側として受け取る)。
資金調達がプラスの間だけ建てて、マイナスに転じたら降りる(逆側=現物ショートは
借株コストなど現実の実行コストが高いため対象にしない)。
"""

import pandas as pd


def generate_signal(funding_rate: pd.Series, window: int = 21, threshold: float = 0.0) -> pd.Series:
    """直近window本(デフォルト21本×8h=7日)の資金調達率の平均が閾値を上回っている間だけ
    キャリーポジション(現物ロング+先物ショート)を建てる。未来の資金調達率は使わない
    (shiftして直近までの情報のみで判断)。
    """
    trailing_avg = funding_rate.rolling(window).mean().shift(1)
    signal = (trailing_avg > threshold).astype(float)
    return signal.fillna(0.0)
