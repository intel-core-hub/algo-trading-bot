"""ポジションサイジング/リスク管理のオーバーレイ。

これまでの検証で「勝てる方向性シグナル」自体は見つかっていないが、自動発注に進む前提として
リスク管理ロジック(ポジションサイズ調整、ドローダウン時の強制フラット化)は
シグナルの優劣によらず独立して必要な基盤なので、ここで整備する。
シグナル(-1〜1)を受け取り、リスク調整後のポジションサイズ(-max_leverage〜max_leverage)を
返す関数群。
"""

import numpy as np
import pandas as pd


def volatility_target_position(
    signal: pd.Series,
    close: pd.Series,
    target_annual_vol: float = 0.3,
    vol_window: int = 24,
    max_leverage: float = 1.0,
    periods_per_year: int = 24 * 365,
    rebalance_every: int = 1,
) -> pd.Series:
    """実現ボラティリティに応じてポジションサイズを調整し、リスク量(年率ボラ)を一定に近づける。

    ボラが高い局面ではサイズを縮小し、低い局面ではmax_leverageまで拡大する
    (ボラが低すぎて割り算が発散しないようmax_leverageで上限を切る)。
    rebalance_everyを1より大きくすると、毎バーではなくNバーごとにしかサイズを更新しない
    (毎バー連続的にサイズを微調整すると、方向を変えていなくても取引コストが積み上がる
    ため、実運用に近づけるには更新頻度を落とす必要がある)。
    """
    realized_vol = close.pct_change().rolling(vol_window).std() * np.sqrt(periods_per_year)
    scale = (target_annual_vol / realized_vol).clip(upper=max_leverage).fillna(0.0)
    sized = (signal * scale).clip(-max_leverage, max_leverage)

    if rebalance_every <= 1:
        return sized

    raw = pd.Series(np.nan, index=sized.index)
    raw.iloc[::rebalance_every] = sized.iloc[::rebalance_every]
    return raw.ffill().fillna(0.0)


def apply_drawdown_stop(
    signal: pd.Series,
    close: pd.Series,
    max_drawdown: float = 0.20,
    cooldown_bars: int = 24,
) -> pd.Series:
    """含み損が max_drawdown を超えたら cooldown_bars 本ノーポジにする、簡易サーキットブレーカー。

    バックテストエンジン(backtest.run_backtest)と同じ「シグナルは翌バーから実効」の
    規約に合わせて、逐次的にエクイティを計算しながらシグナルを間引く。
    """
    price_returns = close.pct_change().fillna(0.0)
    out_signal = signal.copy()

    equity = 1.0
    peak_equity = 1.0
    cooldown_remaining = 0
    prev_out_signal = 0.0  # 前バーの(サーキットブレーカー適用後の)実効シグナル

    for i in range(len(signal.index)):
        # このバーの損益は、前バーまでに決めた実効シグナル(=このバーの実ポジション)で決まる
        equity *= 1.0 + prev_out_signal * price_returns.iloc[i]
        peak_equity = max(peak_equity, equity)
        drawdown = equity / peak_equity - 1.0

        if cooldown_remaining > 0:
            out_signal.iloc[i] = 0.0
            cooldown_remaining -= 1
        elif drawdown < -max_drawdown:
            out_signal.iloc[i] = 0.0
            cooldown_remaining = cooldown_bars
        # else: 元のsignalをそのまま使う(out_signalは既にsignalのコピー)

        prev_out_signal = out_signal.iloc[i]

    return out_signal
