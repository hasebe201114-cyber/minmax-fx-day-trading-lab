# 判定書（Review） - EXP-FX000005（SYS-FX011 ボラティリティ・ブレイク戦略）

> 担当: 品質チーム（adversarial-reviewer / Red Team、本タスク専用に起動された独立エージェント。前回査読とは別インスタンス）
> 方針: 良好結果を積極的に疑う。楽観バイアスの最終砦。
> 対象: 改善ループ第7試行（価格反応型ショック抑制フィルター、4通貨）+ T-01〜T-09（外部レビュー対応）+
> 重複トレード生成バグ修正（`select_non_overlapping_breakout_events()`）+ T-13（出口設計をトレール専業化、`tp_levels=[]`）
> 適用後の最新版。参照データ: `research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json`・
> `vol_breakout_dow_theory_4pairs_v7_trailonly_kpi_evaluation.json`
> **本判定は参考意見であり、本番判定の最終権限は司令塔（人間）に属する。**
> **本査読は前回査読（2026-08-21作成）の指摘事項の解消確認に留まらず、コード・データをゼロベースで独立に再検証した。前回査読者の結論を追認するのではなく、自分で集計・再現した結果に基づく。**

## 最終判定
- [ ] 採用可（参考意見。最終 GO は司令塔が出す）
- [x] **不採用**（理由を該当する論点に明記）
- [ ] 保留（spec の基準は満たさないが、本番運用に資する追加検証が必要な場合）

## 判定の要旨

T-01〜T-09・重複トレード生成バグ修正・T-13（トレール専業化）はいずれもコードを自ら読み、独立にトレード明細を再集計して効果を検証した。**KPI計算パイプライン自体（Sharpe・PF・ペイオフ・DD・最大連敗・スプレッドコスト倍率・permutation p値・実効n）はすべて独立再計算で完全に一致し、数値の捏造・パイプライン不整合は確認されなかった**（後述「検証手順」参照）。T-06（ブロック順列検定）・T-07（実効n二重計上解消）・T-09（コストモデルのスリッページ復活）は実装・効果とも妥当であり、B実装チームの自己発見・自己修正の姿勢（重複トレードバグを含む）も引き続き評価する。

しかし、以下の理由により**依然として不採用**と判定する。

1. **通貨別均質性の分析が最新データで再検証されておらず、独立に再集計した結果、報告されているより深刻な集中依存とサイン反転（正→負）が見つかった**。`currency_homogeneity.json`はT-09+T-06/T-07時点（重複バグ修正前・T-13前、Train619/Validation143/Test204件）のデータで計算されたままで、最終候補（T-13後、Train524/Validation133/Test170件）では一度も再実行されていない。本査読が最終候補データで独立に再集計したところ、Validationは AUD/JPY除外で残存29.4%（報告値49.9%よりさらに悪化）、GBP/JPYはmean_r_netが**+0.034→-0.061と符号が反転**（純負に転落）。Testは USD/JPY除外で残存28.3%（報告値37.7%よりさらに悪化）、EUR/JPYはmean_r_netが**+0.062→-0.181と符号が反転**。4通貨のうち少なくとも1通貨が期間ごとに入れ替わりながら負に転じるという不安定さは、「4通貨で均質なエッジ」という前提を追加で毀損する新規の負の所見である。
2. **証拠金維持率ストレステスト（T-12）も同様に最終候補データで再実行されていない**（`margin_call_stress.json`はdedupfix段階・T-13前のTrain540件で計算）。最大同時保有ポジション数（=4）は最終データでも独立に確認でき変わらないと推定できるが、ロスカット相当水準（Train/Testで約25〜35%）という定量値自体は未検証のまま採否判断の根拠に使われている。
3. **K3m（最大連続損失≤5）がスケール不変でないという外部レビューの指摘（T-08）が、優先順位のトラッキングから漏れたまま一度も対応されていない**。`obs/.../70対応待ち/OBS000008-...md`のチェックリスト（進捗管理表）にはT-08が完了・未着手いずれの欄にも記載がなく、`01_TODO_BREAKDOWN.md`では明確に`[ ]`（未着手）である。外部レビュー自身の分析（i.i.d.想定でTrain観測6のパーセンタイル0.59・Test観測5のパーセンタイル0.28〜0.59）によれば、この基準は本質的にノイズと見分けがつかない。それにもかかわらず「Train6/10・Validation7/10」というKPI達成数の中にK3mがそのまま1票として算入され続けている。
4. **多重検定・過学習のリスクは前回査読時よりむしろ拡大している**。改善ループ第1〜7試行に加え、T-01〜T-13の過程でさらに約10種類の候補（コストモデル3段階・検定方式2種・出口設計2方向等）が試され、Train/Validationの結果を見ながら方向を決める意思決定が繰り返された。Deflated Sharpe等の多重検定補正（外部レビューT-18相当）は依然未実施。
5. **K4m（ペイオフ比≥1.5）はTrain 1.087・Validation 1.180と、T-13で大きく改善したものの依然未達**。K6m（フォワード検証）は一度もフォワードテストが実施されておらず判定不能のまま、レジーム別分析（トレンド/レンジ/高ボラ）も本戦略専用には一度も実施されていない。

以上を総合し、**T-01〜T-13すべて反映後の最新候補（trailonly版）も不採用と判定する**。前回査読と結論は同じだが、根拠は独立であり、むしろ通貨別均質性について前回より深刻な新規の負の所見（GBP/JPY・EUR/JPYの符号反転）を追加で発見した点で、前回査読の単純な追認ではない。

---

## チェックリスト（各項目に証拠）

| # | チェック項目 | 結果 | 証拠 |
|---|---|---|---|
| 1 | 予測単位 × パイプライン評価 | ✅ パイプライン整合性は確認、ただしラベルの誤解を招く残存箇所あり | `research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json`の全trades明細（Train524・Validation133・Test170）を自ら読み込み、`payoff_ratio`（Train1.087/Validation1.180）・`profit_factor`（1.651/2.023）・`spread_cost_multiplier`（2.28/3.12）・`max_consecutive_losses`（6/4）・`max_dd_pct`（peak相対、11.28%/6.52%）・`permutation_p_block`（0.035/0.0989）のすべてを独立にPythonで再計算し、`vol_breakout_dow_theory_4pairs_v7_trailonly_kpi_evaluation.json`の値と完全一致することを確認した（コマンド: `python3 -c "..."`でtrades配列を集計、`src/minmax_fx_dt/backtest/permutation.py`の`permutation_test_block()`を直接呼び出してp値を再現）。数値の捏造・パイプライン不整合は見当たらない。一方、T-13で`tp_levels=[]`にした結果、`exit_reason`ラベルが機能不全化している点を発見: `scripts/backtest_vol_breakout_dow_theory.py:227`の`reason = "SL_INITIAL_NO_TP" if n_levels_hit == 0 else ...`は、TP水準が存在しない現行版では**すべての非週末決済が無条件に`SL_INITIAL_NO_TP`とラベル付けされる**。実際に自分で集計すると、Trainの`SL_INITIAL_NO_TP`516件中332件（64.3%）は`r_gross>0`（最大+6.76R）であり、真の初期逆指値ヒット（-1.0R）ではなくトレーリングストップでの利益確定を含む。コスト計算上は「逆指値決済」として扱われる点自体はT-09の設計（SL・トレールとも逆指値スリッページ1.0pip）と整合するため実害はないが、ラベルの意味が変わったことは`01-trade-scenario-definition.md`§5.4に明記されておらず、今後の診断作業で誤読するリスクがある。 |
| 2 | パラメータ探索と過学習の防止 | ❌ 疑義あり（前回より試行数がさらに増加） | `00-spec.md`を通読し、改善ループ第1〜7試行（うち第4試行はPart A 9通り・Part B 6通りのグリッド、第5試行は4通りの切り分け比較）に加え、T-01〜T-13の過程で新たに約10種類の候補比較（コストモデル3段階=TP指値化→SL/トレール指値化→スリッページ復活、検定方式2種=pair-cluster→day-block、出口設計2方向=段階利確維持→トレール専業）が、いずれも「Trainまたは(Test凍結宣言以降は)Train+Validationの結果を見てから次の一手を選ぶ」形で行われたことを確認した。事前登録の体裁（`00-spec.md`の「事前登録」節）は各試行内では守られているが、**試行間の選択（どの試行を採用しどれを不採用にするか）自体がTrain/Validationの結果に基づく逐次的な意思決定**であり、Deflated Sharpe Ratio等の多重検定補正（外部レビューT-18）は`obs/.../70対応待ち/OBS000008-...md`のチェックリストで依然未実施（`grep -n "T-18"`で本文言及なし、対応記録なし）。 |
| 3 | クロス通貨再検証（特定通貨ペア依存の排除） | ❌ 未達（本査読で新規発見: 報告値より深刻） | `research/method-notes/currency_homogeneity.json`の`generated_at: 2026-08-21T10:58`・`periods.train.all.n: 619`を確認し、これがT-09+T-06/T-07時点（重複バグ修正前・T-13前）のデータであり、最終候補（T-13後、Train524/Validation133/Test170）とは**異なるデータセット**であることを特定した。最終候補の`trades`配列を自ら独立に通貨別集計した結果（コマンド: `python3 -c "..."`でpair別r_net集計・leave-one-out比率を算出）: Validationは**AUD/JPY除外で残存29.4%**（`currency_homogeneity.json`の報告値49.9%よりさらに悪化）、**GBP/JPYのmean_r_netが-0.0607**（報告時点の+0.0344から符号反転、n=23で純負）。Testは**USD/JPY除外で残存28.3%**（報告値37.7%よりさらに悪化）、**EUR/JPYのmean_r_netが-0.1811**（報告時点の+0.062から符号反転、n=23で純負）。4通貨中1通貨が期間ごとに入れ替わりながら負転する不安定さは、報告済みの集中依存（Validation: AUD/JPY・USD/JPY、Test: USD/JPY）よりもさらに強い懸念材料であり、「T-14（通貨拡大）で緩和できる可能性が高い」（`01-trade-scenario-definition.md`§11課題11）という楽観的な記述は現時点で裏付けられていない。 |
| 4 | 統計的堅牢性（permutation p 値・多重比較補正・有効 n） | ⚠️ 検定装置自体はT-06/T-07で改善したが、依然未達かつ別の未修正欠陥（T-08）が残る | `src/minmax_fx_dt/backtest/permutation.py`の`permutation_test_block()`（283〜361行目）を読み、クラスタキー=エントリー日として符号を独立に引くロジックであることを確認。`tests/test_permutation_block.py`の6テストを読み、「4通貨・全勝ケースでもp≥0.05に張り付く（旧clustered版）」「300クラスタ・全勝ならp<0.05に到達できる（新block版）」という完了条件が実際にテストされていることを確認した。さらに最終候補データで`permutation_test_block(r_values, day_clusters, seed=42)`を自ら再実行し、Train p=0.034965(≈0.035、日次クラスタ数118)・Validation p=0.098901(≈0.0989、クラスタ数42)が`vol_breakout_dow_theory_4pairs_v7_trailonly_kpi_evaluation.json`の値と完全一致することを確認した。T-07（`src/minmax_fx_dt/decision/criteria.py:216-221`の`apply_correlation_discount`分岐）も確認し、`apply_correlation_discount=False`指定時はn_effが名目トレード数（=524等）をそのまま返すことをコードで確認した。**Train p=0.035は118クラスタという十分な分解能で得られた値であり検定装置としては機能しているが、有意水準0.05にわずかに滑り込んだに過ぎず頑健とは言い難い**。Validation（p=0.099）・Test（p=0.108、参考）は依然未達。加えて、外部レビューが指摘したK3m（最大連続損失）のスケール不変性問題（T-08）は`obs/.../85外部レビュー/.../01_TODO_BREAKDOWN.md:95`で`### [ ] T-08`のまま一度も対応されておらず、`obs/.../70対応待ち/OBS000008-...md`の進捗チェックリスト（14〜31行目）にも完了・未着手いずれの記載もない（優先順トラッキングから漏れている）。K3mはTrain・Testで6/10・4/10の一部としてFAILに数えられているが、この基準自体がi.i.d.想定下でも通過率6割前後というノイズに近い基準であることは、C査読・意思決定の場に一度も明示的に伝わっていない。 |
| 5 | リーク / バイアス（先読み・データリーク・生存バイアス） | ✅ T-01は独立確認、T-09は独立確認、重複バグ修正も独立確認 | 最終候補データの全827トレード（Train524+Validation133+Test170）についてentry_time/exit_timeのISO週番号を自ら比較し、**週またぎ0件**を確認した（`is_weekend_close_time()`のロジック修正が最終データにも反映されていることを裏付け）。`scripts/backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd.py:146-155`を読み、`remaining_is_stop_triggered = sim["exit_reason"] in ("SL_INITIAL_NO_TP", "TP_THEN_SL_TRAIL")`で`exit_slippage = fraction_remaining * SLIPPAGE_PIPS_STOP_TRIGGERED`（1.0pip）が適用されるロジックを確認し、T-09の事前登録（外部レビュー提案レンジ0.5〜1.0pipの上限）と一致することを確認した。`scripts/backtest_vol_breakout_dow_theory.py:102-150`の`select_non_overlapping_breakout_events()`を読み、`tests/test_dedup_breakout_events.py`の5テスト（同一方向連続イベントの間引き・追跡窓外イベントの保持・反対方向イベントの許可・空入力・長さ不一致エラー）を自ら実行しすべてpassすることを確認した（`PYTHONPATH=src python3 -m pytest tests/ -q`で全88件pass、うち`test_dedup_breakout_events.py`5件・`test_permutation_block.py`6件を含む）。ロジック自体（`active_until[direction]`を選択済みイベントのbreak_timeからのみ更新、方向別に独立管理）はテストケースと整合し、実装は妥当と判断する。なお回帰テスト総数は88件で、タスク依頼文に記載の94件とは一致しなかった（軽微な差異、内容面での不整合は見当たらない）。 |
| 6 | 基準の一貫性（spec の数値基準から逸脱していないか） | ⚠️ 閾値自体は一貫、ただし必須/参考の区別が依然未整備 | `vol_breakout_dow_theory_4pairs_v7_trailonly_kpi_evaluation.json`の`kpi_thresholds`と`00-spec.md`のKPI閾値表（33〜46行目）を照合し完全一致を確認した。前回査読が指摘した「payoff_ratioの二重定義（Rベース/ドルベース）」問題は、`01-trade-scenario-definition.md`§5.2・§8で「ペイオフレシオ(KPI評価値)」と「ペイオフレシオ(生値)」を明示的に区別する記載に改善されており（例: 606行目「payoff_ratio(KPI評価値) 0.787→1.087 / 生値 0.821→1.158」）、この点は前回からの改善として評価する。一方、外部レビューT-11（`obs/.../01_TODO_BREAKDOWN.md:120-121`「hard gateと参考指標を明示的に分ける」）は`obs/.../70対応待ち/OBS000008-...md:31`で依然未着手と確認した。「Train6/10・Validation7/10」という単純な達成数の羅列は、統計的に無意味と判明済みのK3m（項目4参照）や、実効nの重み付けなしの二値判定（min_n_trades_effectiveは299件と301件を区別できない）を、Sharpe・PFのような比較的頑健な指標と同列に1票としてカウントしており、採否判断の根拠として単純化しすぎている。 |
| 7 | 当日クローズ遵守（週末クローズ、K7・Q7相当） | ✅ 最終データで独立に達成を確認 | 項目5の週またぎ0件確認に加え、`exit_reason`の内訳を独立集計（Train: `WEEKEND_NO_TP` 8件・`SL_INITIAL_NO_TP` 516件、Validation: `WEEKEND_NO_TP` 2件・`SL_INITIAL_NO_TP` 131件、Test: `WEEKEND_NO_TP` 4件・`SL_INITIAL_NO_TP` 166件）し、週末強制クローズが実際に発動していることを確認した。T-13で`tp_levels=[]`になった結果、`TP_THEN_WEEKEND`・`TP_FULL`・`TP_THEN_SL_TRAIL`の3ラベルは最終データに一件も出現しない（項目1のラベル問題と同根）が、週末クローズ自体の機能は健全。 |
| 8 | コスト反映（スプレッド + スリッページ 0.5 pip） | ✅ T-09対応済みを独立確認、モデル自体は妥当 | `scripts/backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd.py:70-73`で`SPREAD_PIPS`（USD/JPY 0.3・EUR/JPY 0.5・GBP/JPY 0.7・AUD/JPY 0.6、`CLAUDE.md`記載のUSD/JPY 0.3銭・GBP/JPY 0.7銭と一致）・`SLIPPAGE_PIPS_MARKET_LEG=0.5`・`SLIPPAGE_PIPS_STOP_TRIGGERED=1.0`・`COMMISSION_RATE_ROUND_TRIP=0.00004`を確認し、146〜160行目のコスト按分ロジック（TP到達分=指値スリッページ0、SL・トレール決済分=逆指値スリッページ1.0pip、週末・MAX_HOLD分=成行スリッページ0.5pip、`fraction_via_tp`で按分）を読んだ。`spread_cost_multiplier`（K5m）をtrades明細から独立再計算し、Train 2.28・Validation 3.12・Test 1.95（参考）が完全一致することを確認した。コストモデル自体は外部レビュー・前回C査読の指摘（T-09）に対して妥当に対応済みと判断する。 |
| 9 | 市場レジーム別成績（トレンド/レンジ/高ボラの3種） | ❌ 依然未実施 | `research/EXP-FX000005/`・`research/method-notes/`配下を`grep -rl "regime\|レジーム"`で検索した結果、本戦略専用のレジーム別（トレンド/レンジ/高ボラ）成績分析は依然として存在しないことを確認した（ヒットしたのは`00-prescreen.md`の一般的言及と、`00-spec.md`・`01-trade-scenario-definition.md`・本判定書自身が前回C査読の指摘を引用している箇所のみ）。戦略自体が「高ボラブレイク検出」を前提とする設計であるにもかかわらず、通常ボラ局面での誤検知率・機会コストは一度も定量化されていない。前回査読からの状況変化なし。 |
| 10 | フォワード検証との乖離率（K6m） | 判定不能（未実施、状況変化なし） | `grep -rli forward research/EXP-FX000005/ research/method-notes/*.json`で本戦略に関連するフォワードテストデータの存在有無を確認したが該当なし。`00-spec.md`「Test凍結宣言」により2026-08-15以降の未使用データが蓄積された時点で初めて評価可能になる設計であり、現時点でK6mを判定することは原理的にできない。前回査読からの状況変化なし。 |

---

## 検証手順・再現ログ（要旨）

- `PYTHONPATH=src python3 -m pytest tests/ -q` を実行し、88件全passを確認（タスク依頼文の94件とは一致せず、軽微な差異として記録）。
- `research/method-notes/vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json`の全827トレード（Train524・Validation133・Test170）をPythonで読み込み、(a) 週またぎ0件、(b) exit_reason内訳、(c) 通貨別n・勝率・mean_r_net・leave-one-out比率、(d) payoff_ratio/PF/最大連続損失/spread_cost_multiplier/最大DD(ピーク比)を独立に再計算し、いずれも`vol_breakout_dow_theory_4pairs_v7_trailonly_kpi_evaluation.json`の値と完全一致することを確認した。
- `src/minmax_fx_dt/backtest/permutation.py`の`permutation_test_block()`を直接呼び出し（`seed=42`、クラスタキー=エントリー日）、Train p=0.035・Validation p=0.099・Test p=0.108（参考）が報告値と一致することを確認した。
- `scripts/backtest_vol_breakout_dow_theory.py`の`select_non_overlapping_breakout_events()`（102〜150行目）・`scripts/backtest_vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd.py`の`find_trades_for_period()`・コスト按分ロジック（146〜160行目）を読み、重複トレード修正・T-09コストモデルの実装を検証した。
- `scripts/evaluate_vol_breakout_dow_theory_4pairs_v7_trailonly_kpi.py`（`perm_p_field="perm_p_block"`・`apply_n_correlation_discount=False`）と`scripts/evaluate_vol_breakout_dow_theory_kpi.py`の`evaluate_period()`を読み、T-06・T-07の適用箇所を確認した。
- `research/method-notes/currency_homogeneity.json`・`margin_call_stress.json`の`generated_at`・トレード件数（619/143/204、540/138/173）を確認し、最終候補（524/133/170）とは異なるデータセットで計算されていることを特定した。
- `obs/minmax_fx_day_trading_lab/70対応待ち/OBS000008-EXP-FX000005-外部レビュー対応.md`・`obs/minmax_fx_day_trading_lab/85外部レビュー/2026-08-20_EXP-FX000005_External_Review/01_TODO_BREAKDOWN.md`を読み、T-08・T-10・T-11・T-14〜T-20の対応状況を確認した。

---

## 反論

- **T-06・T-07・T-09の対応、および重複トレード生成バグの自己発見・修正は、いずれも実装・テストとも独立検証に耐える質だった**。特にB実装チームが自らのセッション内で発見した重複バグを隠さず開示し、影響範囲（Test期間で総利益36.4%が重複計上由来）を定量化した上で回帰テストを追加した対応は、CLAUDE.mdの「良好な実践」として評価に値する。
- **T-13（トレール専業化）は正しい方向の変更だった**。ペイオフレシオがTrain 0.787→1.087・Validation 0.770→1.180まで改善し、Trainのpermutation p値が初めて有意水準を満たした（0.035）。これは「段階利確そのものがペイオフレシオの上限を構造的に抑えていた」という外部レビュー§1-1の仮説の一部を裏付けるものであり、B実装チームの検証プロセスは機能している。
- **ただし、これらの改善は「不採用の理由が減った」ことを意味しない**。むしろ本査読は、報告書に記載されている通貨別均質性・証拠金維持率ストレスの数値が、実は最終候補データではなく1〜2世代前のデータで計算されたものであることを新たに発見した。特に通貨別均質性については、最終データで独立に再計算した結果が報告値より一貫して悪い方向（集中度が高い、符号が反転する通貨がある）に出ており、これは「T-14（通貨拡大）で緩和できる見込みが高い」という`01-trade-scenario-definition.md`の楽観的な記述の説得力を弱める。
- **K3m（最大連続損失）が統計的に無意味な基準であることは外部レビューが2026-08-20時点で既に指摘済み（T-08）だが、1年以上前の当時の優先順位付けで見落とされたまま、現在の「6/10」「7/10」という達成数の中に依然として1票として算入され続けている**。これは意思決定の透明性を損なう軽視できないプロセス上の欠落であり、B実装チームまたはA設計チームで再度優先順位を検討すべきである。
- **総じて、本戦略はTrain・Validationで正のエッジを示す複数の頑健な兆候（Sharpe・PF・月次期待値の一貫した達成、Trainでのpermutation有意性到達）を持っており、「エッジが存在しない」と断じるほど弱くはない**。しかし、(a) K4m未達、(b) K6m判定不能、(c) レジーム別分析皆無、(d) 通貨別集中依存が最終データでは報告より悪化、(e) 証拠金維持率リスク（Train/Testで約25〜35%）が最終データで未再検証、という複数の独立した理由が積み重なっており、これらすべてが解消されない限り採用に足る水準には至らない。

## 差し戻し（該当あり）

- 差し戻し先: **B 実装チーム**（→再評価後、必要ならA設計チームへ設計見直し・司令塔へ最終判断）
- 内容:
  1. **通貨別均質性検証（`analyze_currency_homogeneity.py`）を最終候補データ（T-13後、Train524/Validation133/Test170件）で再実行する**。本査読の独立集計により、報告値より深刻な集中依存（Validation AUD/JPY除外で残存29.4%、Test USD/JPY除外で残存28.3%）とGBP/JPY（Validation）・EUR/JPY（Test）の符号反転（正→負）を確認しており、この再実行なしに通貨別の均質性を「概ね問題ない」と判断すべきではない。
  2. **証拠金維持率ストレステスト（T-12、`analyze_margin_call_stress.py`）を最終候補データで再実行する**。最大同時保有ポジション数（=4）は最終データでも本査読で独立確認できたが、ロスカット相当水準の割合（約25〜35%）自体は未検証のまま残っている。
  3. **T-08（K3mのスケール不変な再定義）に着手する**。外部レビュー(2026-08-20)で指摘されて以来一度も対応されておらず、優先順位トラッキング(`OBS000008-...md`)からも漏れている。この基準がi.i.d.想定下でも通過率6割前後というノイズに近い基準であることを踏まえ、KPI達成数の算出方法（項目6のT-11とあわせて）を見直すべきである。
  4. **T-11（hard gate/参考指標の明示的分離）に着手する**。「X/10」という単純な達成数の羅列は、統計的に無意味と判明したK3mや、閾値付近で不安定な実効n判定を、Sharpe・PFのような頑健な指標と同列に扱っており、採否判断の透明性を損なっている。
  5. **通貨ペア拡大（T-14）または現行4通貨での運用継続いずれを取るにせよ、フォワードテスト（K6m）とレジーム別分析（項目9）は採用検討の前提として不可欠**であり、これらが揃わない限り正式なGO判定を出すべきではないと本査読は考える。
  6. 上記対応後、**改めてC査読（本タスクの再実施）を経てから司令塔判断を仰ぐ**こと。

## 変更履歴
- 2026-08-21: 作成（独立エージェントによるC査読、T-16対応）
- 2026-08-21: 再判定（重複トレードバグ修正後・T-01〜T-13すべて反映後のデータで再査読。B実装チームとは別の独立エージェントインスタンスによる再検証。前回査読の指摘事項の解消確認に加え、通貨別均質性が最終データで未再検証かつ報告値より悪化していること・K3m（T-08）が優先順位トラッキングから漏れていることを新規発見。結論は前回同様「不採用」だが、独立した再検証に基づく判断であり前回の機械的な踏襲ではない）
