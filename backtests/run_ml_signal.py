"""特徴量エンジニアリング + RandomForest分類器ベースのシグナルをwalk-forwardで検証する。

これまでのルールベース戦略(MAクロス/モメンタム/平均回帰/トレンドフォロー)は
過学習を抑えても手数料控除後のエッジが確認できなかった。単純な線形ルールでは
拾えない非線形な特徴量の組み合わせに賭けて、ML分類器で方向(上昇/下落/横ばい)を
予測できるかを検証する。

方式(walk_forward.pyと同じローリング窓の考え方):
- train窓の特徴量・ラベルでRandomForestClassifierを学習
- test窓はモデルの予測をそのままシグナルとして使う(test窓は完全にout-of-sample)
- test窓ごとにモデルを再学習して次のtest窓へロールしていく

Usage:
    python backtests/run_ml_signal.py
data/ に事前に `python src/data.py <symbol> 1h 20000` でキャッシュしたOHLCVが必要。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from backtest import run_backtest  # noqa: E402
from features import FEATURE_COLUMNS, build_features, build_labels  # noqa: E402
from walk_forward import Fold, WalkForwardReport  # noqa: E402

DATA_FILES = {
    "BTC/USDT": ROOT / "data" / "binance_BTC-USDT_1h.csv",
    "ETH/USDT": ROOT / "data" / "binance_ETH-USDT_1h.csv",
}

TRAIN_BARS = 2000  # 約83日
TEST_BARS = 500  # 約21日
HORIZON = 12  # 予測対象: 12時間先までの方向
LABEL_THRESHOLD = 0.005  # 往復コスト(手数料+スリッページ)より高い閾値で有意な方向のみ学習させる


def hold_for_horizon(predictions: np.ndarray, index: pd.DatetimeIndex, horizon: int) -> pd.Series:
    """予測ラベルはhorizon本先までの方向なので、毎バー乗り換えず、horizon本ごとに
    再判定して次の再判定までポジションを保持する(でないと1バー毎の取引コストで溶ける)。"""
    raw = pd.Series(np.nan, index=index)
    raw.iloc[::horizon] = predictions[::horizon]
    return raw.ffill().fillna(0.0)


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def walk_forward_ml(df: pd.DataFrame, train_bars: int, test_bars: int) -> WalkForwardReport:
    feat = build_features(df)
    label = build_labels(df["close"], horizon=HORIZON, threshold=LABEL_THRESHOLD)

    data = feat.copy()
    data["label"] = label
    data["close"] = df["close"]
    data = data.dropna()

    folds: list[Fold] = []
    combined_returns: list[pd.Series] = []

    start = 0
    n = len(data)
    while start + train_bars + test_bars <= n:
        train = data.iloc[start : start + train_bars]
        test = data.iloc[start + train_bars : start + train_bars + test_bars]

        model = make_model()
        model.fit(train[FEATURE_COLUMNS], train["label"])

        train_signal = hold_for_horizon(model.predict(train[FEATURE_COLUMNS]), train.index, HORIZON)
        train_result = run_backtest(train["close"], train_signal)

        test_signal = hold_for_horizon(model.predict(test[FEATURE_COLUMNS]), test.index, HORIZON)
        test_result = run_backtest(test["close"], test_signal)

        importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(3)))
        top_features = dict(sorted(importances.items(), key=lambda kv: -kv[1])[:3])

        folds.append(
            Fold(
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                best_params=top_features,
                train_result=train_result,
                test_result=test_result,
            )
        )
        combined_returns.append(test_result.returns)
        start += test_bars

    if not folds:
        raise ValueError("not enough bars for even a single train/test fold")

    return WalkForwardReport(folds=folds, combined_returns=pd.concat(combined_returns))


def summarize(symbol: str, report: WalkForwardReport) -> None:
    result = report.combined_result
    n_folds = len(report.folds)
    win_folds = sum(1 for f in report.folds if f.test_result.total_return > 0)
    avg_trades = np.mean([f.test_result.n_trades for f in report.folds])

    print(
        f"{symbol:10s} ml_signal | oos_return={result.total_return:+.2%} | oos_sharpe={result.sharpe:+.2f} "
        f"| oos_max_dd={result.max_drawdown:.2%} | folds={n_folds} | fold_win_rate={win_folds / n_folds:.0%} "
        f"| avg_trades/fold={avg_trades:.0f}"
    )
    recurring_features = {}
    for f in report.folds:
        for feat_name in f.best_params:
            recurring_features[feat_name] = recurring_features.get(feat_name, 0) + 1
    top = sorted(recurring_features.items(), key=lambda kv: -kv[1])[:5]
    print(f"           most frequent top-3 features across folds: {top}")


def main() -> None:
    for symbol, path in DATA_FILES.items():
        if not path.exists():
            print(f"skip {symbol}: {path} not found (run src/data.py first)")
            continue

        df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
        print(f"\n### {symbol}: {df.index[0]} -> {df.index[-1]} ({len(df)} bars) ###")

        report = walk_forward_ml(df, TRAIN_BARS, TEST_BARS)
        summarize(symbol, report)

        n = len(df)
        test_slices = []
        start = 0
        while start + TRAIN_BARS + TEST_BARS <= n:
            test_slices.append(df.iloc[start + TRAIN_BARS : start + TRAIN_BARS + TEST_BARS])
            start += TEST_BARS
        bh_returns = [run_backtest(s["close"], pd.Series(1.0, index=s.index)).returns for s in test_slices]
        bh_equity = (1 + pd.concat(bh_returns)).cumprod()
        print(f"{'buy_and_hold':10s} {'':10s}| oos_return={float(bh_equity.iloc[-1] - 1):+.2%}")


if __name__ == "__main__":
    main()
