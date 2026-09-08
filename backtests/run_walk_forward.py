"""walk-forwardパラメータ最適化で3戦略(MAクロス/モメンタム/平均回帰)をBTC・ETHで検証する。

各foldでtrain窓のみを使ってパラメータグリッドを探索し、選ばれたパラメータをtest窓に
そのまま適用する(test窓のリターンはout-of-sample)。全foldのtest窓を連結し、
同じ連結期間のbuy-and-holdと比較する。

パラメータ選択は2方式を比較する:
- naive:  train窓全体でのSharpe最大化(単一窓の運の良さにフィットしやすい)
- robust: train窓をサブ期間に分け、(平均Sharpe - 標準偏差)が最大のものを選ぶ
          (どのサブ期間でも安定して機能するパラメータを優先し、過学習を抑える狙い)

Usage:
    python backtests/run_walk_forward.py
data/ に事前に `python src/data.py <symbol> 1h 20000` でキャッシュしたOHLCVが必要。
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "strategies"))

from backtest import run_backtest  # noqa: E402
from walk_forward import naive_selector, robust_selector, walk_forward_optimize  # noqa: E402
import ma_cross  # noqa: E402
import mean_reversion  # noqa: E402
import momentum  # noqa: E402

DATA_FILES = {
    "BTC/USDT": ROOT / "data" / "binance_BTC-USDT_1h.csv",
    "ETH/USDT": ROOT / "data" / "binance_ETH-USDT_1h.csv",
}

TRAIN_BARS = 2000  # 約83日
TEST_BARS = 500  # 約21日

STRATEGIES = {
    "ma_cross": dict(
        signal_fn=ma_cross.generate_signal,
        param_grid={"fast": [10, 20, 30], "slow": [50, 100, 150]},
        param_filter=lambda p: p["fast"] < p["slow"],
    ),
    "momentum": dict(
        signal_fn=momentum.generate_signal,
        param_grid={"lookback": [12, 24, 48, 72], "threshold": [0.0, 0.01, 0.02]},
        param_filter=None,
    ),
    "mean_reversion": dict(
        signal_fn=mean_reversion.generate_signal,
        param_grid={"window": [10, 20, 30], "entry_z": [1.5, 2.0, 2.5], "exit_z": [0.25, 0.5]},
        param_filter=None,
    ),
}


def summarize(label: str, report) -> None:
    result = report.combined_result
    n_folds = len(report.folds)
    win_folds = sum(1 for f in report.folds if f.test_result.total_return > 0)
    top_params = Counter(tuple(sorted(f.best_params.items())) for f in report.folds).most_common(3)

    print(
        f"{label:26s} | oos_return={result.total_return:+.2%} | oos_sharpe={result.sharpe:+.2f} "
        f"| oos_max_dd={result.max_drawdown:.2%} | folds={n_folds} | fold_win_rate={win_folds / n_folds:.0%}"
    )
    print("                           top params across folds:")
    for params, count in top_params:
        print(f"                             {dict(params)} x{count}")


def main() -> None:
    for symbol, path in DATA_FILES.items():
        if not path.exists():
            print(f"skip {symbol}: {path} not found (run src/data.py first)")
            continue

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        print(f"\n### {symbol}: {df.index[0]} -> {df.index[-1]} ({len(df)} bars) ###")

        for name, cfg in STRATEGIES.items():
            naive_report = walk_forward_optimize(
                df,
                signal_fn=cfg["signal_fn"],
                param_grid=cfg["param_grid"],
                param_filter=cfg["param_filter"],
                train_bars=TRAIN_BARS,
                test_bars=TEST_BARS,
                selector=naive_selector(cfg["signal_fn"]),
            )
            summarize(f"{name} (naive)", naive_report)

            robust_report = walk_forward_optimize(
                df,
                signal_fn=cfg["signal_fn"],
                param_grid=cfg["param_grid"],
                param_filter=cfg["param_filter"],
                train_bars=TRAIN_BARS,
                test_bars=TEST_BARS,
                selector=robust_selector(cfg["signal_fn"], n_sub_windows=4),
            )
            summarize(f"{name} (robust)", robust_report)

        # 同じ連結out-of-sample期間でのbuy-and-hold(fold毎のtest窓をそのまま繋げる)
        n = len(df)
        test_slices = []
        start = 0
        while start + TRAIN_BARS + TEST_BARS <= n:
            test_slices.append(df.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS])
            start += TEST_BARS
        combined_close = pd.concat([s["close"] for s in test_slices])
        bh_returns = []
        for s in test_slices:
            bh = run_backtest(s["close"], pd.Series(1.0, index=s.index))
            bh_returns.append(bh.returns)
        bh_combined = pd.concat(bh_returns)
        bh_equity = (1 + bh_combined).cumprod()
        print(
            f"{'buy_and_hold':26s} | oos_return={float(bh_equity.iloc[-1] - 1):+.2%} "
            f"(same {len(test_slices)} test windows, stitched)"
        )


if __name__ == "__main__":
    main()
