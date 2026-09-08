# algo-trading-bot

暗号資産市場向けの自動売買戦略の構築・検証プロジェクト。

## スコープと役割分担

- **ここで作るもの**: 相場データ取得、特徴量/シグナル生成、バックテスト、リスク管理ロジック。
- **ユーザー側で行うこと**: 実際の取引所APIキー発行・保管、実資金での発注実行。
  自動発注コードはテストネット/ペーパートレード止まりとし、実弾運用は検証が十分に固まってから
  ユーザー自身の判断・操作で行う。

## 進め方

1. `src/data.py` — ccxtで公開データ(OHLCV)を取得しキャッシュ(ページングして長期履歴も取得可能。
   `python src/data.py <symbol> <timeframe> <total_bars>` でCLI実行可能)
2. `strategies/` — ルールベースの戦略ロジック(MAクロス/モメンタム/平均回帰/トレンドフォロー)
3. `src/backtest.py` — 自前の軽量バックテストエンジン。手数料・スリッページを考慮
4. `backtests/run_baseline.py` — 単純なtrain/test(in-sample/out-of-sample)分割で検証
5. `src/walk_forward.py` / `backtests/run_walk_forward.py` — ローリングwalk-forwardで
   パラメータ再最適化しながらout-of-sample検証(複数銘柄対応)
6. `src/features.py` / `backtests/run_ml_signal.py` — 特徴量エンジニアリング +
   RandomForest分類器ベースのシグナルをwalk-forwardで検証
7. `backtests/run_timeframe_sweep.py` — 同じ戦略群を1h/4h/日足で横並び比較
8. 十分な検証を経てから、テストネットでの自動発注に進む

### 現状の結果 (2026-09-08時点)

これまでに検証した内容(すべてBTC/USDT・ETH/USDT、手数料10bps+スリッページ5bps込み):

1. **単純なtrain/test分割** (`run_baseline.py`, 1h足): MAクロス・モメンタム・平均回帰の
   3戦略とも train/test 両方 buy-and-hold に負けた。特にモメンタムは取引回数が多く
   コスト負けが大きい。
2. **walk-forward最適化** (`walk_forward.py` / `run_walk_forward.py`, 1h足2.3年、
   train2000本/test500本ローリング36 fold): train窓でのグリッドサーチ最良パラメータを
   test窓に適用してもout-of-sample連結リターンは-73%〜-96%とさらに悪化。fold毎の
   ベストパラメータも安定せず、**ナイーブな「train窓のSharpe最大化」は過学習しやすい**
   ことを確認。train窓を4分割し(平均-標準偏差)を最大化する`robust_selector`に
   変えても-65%〜-95%と大勢は変わらず。
3. **長期トレンドフォロー** (`strategies/trend_following.py`, 日足換算50/200日MAクロス):
   取引回数を数回まで激減させ手数料の影響をほぼ消しても、なお buy-and-hold に負けた
   → **コストの問題ではなく方向予測自体が外れている**ことを確認。
4. **MLシグナル** (`src/features.py` / `run_ml_signal.py`, RandomForest 3クラス分類):
   予測ホライズンと保有期間を一致させる修正(最初は毎バー再予測でコスト負けするバグが
   あった)後もout-of-sample -65.95%/-96.47%。方向的中率は train~50%・test~36%
   (ランダムは33%)でほぼ運と同じ、特徴量重要度もボラティリティ系のみが上位。
5. **時間足スイープ** (`run_timeframe_sweep.py`, 1h/4h/日足): 直近850本の日足だけで
   見るとMAクロス・モメンタムが逆にbuy-and-holdに大勝ち(ETHモメンタムtest +250%等)
   したが、**日足履歴を2000本(≈5.5年、2021-2026、複数の強気/弱気相場を含む)に
   拡張して再検証すると勝ちが消え、元の「エッジなし」の結論に戻った**。
   小サンプル(850本≒2年強)でのbacktest結果は簡単に過学習・偶然の勝ちを生む、
   という重要な教訓。

**現時点までの結論**: ルールベース4種、walk-forwardでのnaive/robust再最適化、
RandomForest分類器、3つの時間足のいずれも、手数料控除後に一貫したエッジを示せなかった。
また日足の小サンプルで見えた「勝ち」も長期データでは再現しなかった。単純な価格ベースの
特徴量だけでは、この市場にこの粒度の方向優位性は乗っていない可能性が高い。
次のステップ候補:
(a) オンチェーン/資金調達率(funding rate)/オーダーブック不均衡など価格以外の特徴量、
(b) 「勝つ戦略を探す」から「複数の弱い戦略を組み合わせてリスク管理する」方向への転換、
(c) いったんテクニカル戦略探索を止め、リスク管理・ポジションサイジングなど
    自動発注に必要な周辺基盤の整備を優先する。

## セットアップ

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

APIキーが必要になった場合は `.env` に置く(gitignore済み)。マーケットデータ取得自体は
公開エンドポイントのみを使うため、当面はAPIキー不要で進められる。
