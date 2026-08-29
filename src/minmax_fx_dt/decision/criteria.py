"""撤退/採用判定基準 (SYS-FX007 レンジブレイク・プルバック戦略向け).

背景 (親プロジェクト minmax-trading-pilot の criteria.py を流用・拡張):
- 親PJは単一 Sharpe 閾値で判定 (044=0.5, 046=1.0, 048=0.5, 050=0.5)
- 本PJの SYS-FX007 は K1m〜K7m の 7 指標で多面的に評価
  * K1m: 月次 Sharpe ≥ 0.4, PF ≥ 1.2, 期待値 > 0
  * K2m: 月間 DD ≤ 10%, 年間 DD ≤ 20%
  * K3m: 最大連続損失 ≤ 5 トレード
  * K4m: ペイオフレシオ ≥ 1.5
  * K5m: 1トレード期待値 > スプレッド往復 × 3
  * K6m: バックテスト ↔ フォワード乖離率 ≤ 30%
  * K7m: 両建て証拠金消費率 ≤ 30%

ルール (本PJ 2026-08-13 司令塔 GO + spec 評価基準を数値固定):
1. Day 30 まで: 累積 Sharpe < -0.5 → 撤退検討 / 取引数 0 → 戦略ロジック再点検
2. Day 30: RANGE 2 シグナル以上 + Sharpe < 0 → 評価延長
3. Day 60: 累積 Sharpe < 0.0 + RANGE 2 シグナル以上 → Phase 2 延長 or 撤退
4. Day 90: 累積 Sharpe < 0.4 → 本採用見送り
5. Day 90: 取引数 < 3 → 06 ヶ月延長を自動オプション化

親PJ criteria.py からの差分:
- 単一 Sharpe → K1m〜K7m の 7 指標で評価
- STRATEGY_GO_THRESHOLD を 1 戦略に最適化 (SYS-FX007 = 0.4)
- 多通貨 (5 通貨) の KPI 合算判定
- 弱いブレイク排除率 (v2 拡張) を追加チェック
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TypedDict

import numpy as np

from ..backtest.permutation import effective_pair_count

ROOT = Path(__file__).resolve().parents[3]
REGIME_LATEST = ROOT / "research" / "regime_latest.json"


class Verdict(str, Enum):
    """撤退/採用判定."""

    GO = "GO"
    WATCH = "WATCH"
    REJECT = "REJECT"
    SAMPLE_DEFICIT = "SAMPLE-DEFICIT"
    REGIME_EXTEND = "REGIME-EXTEND"  # レンジ相場で評価延長


# 戦略別 GO 閾値
# 親PJ: {"EXP-OBS000044": 0.5, "EXP-OBS000046": 1.0, ...}
STRATEGY_GO_THRESHOLD: dict[str, float] = {
    "SYS-FX007": 0.4,  # レンジブレイク・プルバック (中期スイング + 両建て)
    "SYS-FX009": 0.4,  # 上位足トレンド+ダブルトップ/ボトム (1:1+トレーリング)
}

# min_n_trades の再導出 (2026-08-17、司令塔指摘対応):
# 旧値 60→66 は統計的な検出力計算に基づくものではなく、(1) 60 は初期 spec の
# 経験則的な値 (導出根拠の記録なし)、(2) 66 は SYS-FX007 ベースラインプリセットの
# Train 実測トレード数がたまたま 66 件だった、という2つのその場しのぎの数字が
# 「同じ基準を使う」という理由だけで SYS-FX008/009 へ踏襲されてきたに過ぎない
# (詳細: research/method-notes/min_n_trades_power_analysis.json)。
#
# 再導出: 実際に使っている backtest.permutation.permutation_test() そのものを使い、
# 実測トレード値幅分布 (SYS-FX009 Train pooled, n=179) をブートストラップして、
# 「勝率60%相当の弱いエッジ」を検出力80%・有意水準0.05で検出するために必要な
# 最小サンプル数をシミュレーションで算出した (n=300)。この水準を採用する。
# 勝率55%相当まで拾おうとすると n=1,165 まで跳ね上がり、本PJの実測シグナル頻度
# (5通貨・数年で数十〜数百件) では原理的に届かないため、60%を実務上の下限とした。
MIN_N_TRADES_STATISTICAL = 300  # 勝率60%相当のエッジを検出力80%で検出できる最小n

# min_n_trades の「期間考慮」(2026-08-28、司令塔判断 (b)「n値は、期間を考慮」)
#
# 背景: MIN_N_TRADES_STATISTICAL=300 は Train (2023-11〜2025-03、約17ヶ月) の
# 実測トレード値幅分布から導出した検出力シミュレーションの結果である。この値を
# Validation (8.0ヶ月) や Test (8.4ヶ月) といった**より短い窓へそのまま適用すると、
# 同じ絶対件数を約半分の期間で要求する = 約2倍のトレード頻度を要求する**ことになる。
# 本PJのスコープ (保有期間 数日〜数週間・4通貨) では、戦略の優劣と無関係に
# 短い窓では原理的に達成できない (EXP-FX000021 amendment-01 §7)。
#
# 恒久ルール: 閾値を「基準窓 (Train と同じ長さ) あたり300件」と読み替え、
# 評価窓の長さに比例させる。**300 という水準自体は緩めない。**
#
# 重要な但し書き (PJ000004 Q15 に明記): 検出力80%という保証は n の絶対値に依存する
# ため、期間比例化した閾値は「その窓で戦略が想定どおりの頻度でトレードしたか」を
# 見る指標であって、「弱いエッジを検出するのに十分な検出力があるか」の保証ではない。
# 短い窓で閾値を満たしても、統計的検出力そのものは不足したままである。
REFERENCE_WINDOW_MONTHS = 16.95  # Train: 2023-11-01〜2025-03-31


def min_n_trades_for_window(window_months: float,
                            base: int = MIN_N_TRADES_STATISTICAL,
                            reference_months: float = REFERENCE_WINDOW_MONTHS) -> int:
    """評価窓の長さを考慮した min_n_trades 閾値を返す (司令塔判断 (b)、2026-08-28).

    基準窓 (Train と同じ長さ) で `base` 件を要求し、それより短い/長い窓では
    窓長に比例させる。四捨五入して整数を返す。

    >>> min_n_trades_for_window(16.95)   # Train と同じ長さ
    300
    >>> min_n_trades_for_window(7.98)    # Validation (8.0ヶ月)
    141
    >>> min_n_trades_for_window(8.44)    # Test (8.4ヶ月)
    149

    注意: 本関数は「頻度が想定どおりか」を見るためのものであり、検出力80%の保証は
    n の絶対値に依存するため短い窓では失われる (上のコメント参照)。
    """
    if window_months <= 0:
        raise ValueError(f"window_months must be positive, got {window_months}")
    return round(base * window_months / reference_months)


# SYS-FX007 の K1m〜K7m 閾値 (research/EXP-FX000001/00-spec.md から転記)
KPI_THRESHOLDS: dict[str, dict[str, float]] = {
    "SYS-FX007": {
        "sharpe_monthly": 0.4,            # K1m
        "profit_factor_monthly": 1.2,     # K1m
        "max_dd_monthly_pct": 10.0,       # K2m (証拠金 %)
        "max_dd_yearly_pct": 20.0,        # K2m
        "max_consecutive_losses": 5,      # K3m
        "payoff_ratio": 1.5,              # K4m
        "spread_cost_multiple": 3.0,      # K5m
        "backtest_forward_divergence_pct": 30.0,  # K6m
        "max_margin_usage_pct": 30.0,     # K7m
        "weak_breakout_exclusion_pct": 30.0,  # v2 拡張
        "min_n_trades": MIN_N_TRADES_STATISTICAL,  # 統計的有意性 (2026-08-17 再導出、旧60)
        "permutation_p_value": 0.05,      # 統計的有意性
    },
    "SYS-FX008": {
        # EXP-FX000002/00-spec.md v1 から転記。SYS-FX007と同一のK1m〜K6m基準を使い
        # ポートフォリオ比較を可能にする。K7m は片側ポジションのみのため対象外
        # (hedging_enabled=False で evaluate_kpis() が自動的に判定対象外にする)。
        "sharpe_monthly": 0.4,            # K1m
        "profit_factor_monthly": 1.2,     # K1m
        "max_dd_monthly_pct": 10.0,       # K2m (証拠金 %)
        "max_dd_yearly_pct": 20.0,        # K2m
        "max_consecutive_losses": 5,      # K3m
        "payoff_ratio": 1.5,              # K4m
        "spread_cost_multiple": 3.0,      # K5m
        "backtest_forward_divergence_pct": 30.0,  # K6m
        "max_margin_usage_pct": 30.0,     # K7m (判定対象外だが閾値自体は形式上必要)
        "weak_breakout_exclusion_pct": 0.0,  # SYS-FX008にMT-3フォールバック概念はないためno-op (0以上は常に真)
        "min_n_trades": MIN_N_TRADES_STATISTICAL,  # 統計的有意性 (2026-08-17 再導出、旧66)
        "permutation_p_value": 0.05,      # 統計的有意性
    },
    "SYS-FX009": {
        # EXP-FX000003/00-spec.md から転記。SYS-FX007/008と同一のK1m〜K6m基準を使い
        # ポートフォリオ比較を可能にする。K7m は片側ポジションのみのため対象外
        # (hedging_enabled=False で evaluate_kpis() が自動的に判定対象外にする)。
        "sharpe_monthly": 0.4,            # K1m
        "profit_factor_monthly": 1.2,     # K1m
        "max_dd_monthly_pct": 10.0,       # K2m (証拠金 %)
        "max_dd_yearly_pct": 20.0,        # K2m
        "max_consecutive_losses": 5,      # K3m
        "payoff_ratio": 1.5,              # K4m
        "spread_cost_multiple": 3.0,      # K5m
        "backtest_forward_divergence_pct": 30.0,  # K6m
        "max_margin_usage_pct": 30.0,     # K7m (判定対象外だが閾値自体は形式上必要)
        "weak_breakout_exclusion_pct": 0.0,  # SYS-FX009にMT-3フォールバック概念はないためno-op
        "min_n_trades": MIN_N_TRADES_STATISTICAL,  # 統計的有意性 (2026-08-17 再導出、旧66)
        "permutation_p_value": 0.05,      # 統計的有意性
    },
}

# Day 30 短期撤退閾値 (親PJ踏襲)
DAY30_SHARPE_REJECT = -0.5

# Day 60 中期撤退閾値 (親PJ踏襲)
DAY60_SHARPE_REJECT = 0.0

# 取引数 SAMPLE-DEFICIT 閾値 (親PJ踏襲)
SAMPLE_DEFICIT_N_TRADES = 3


@dataclass
class KPIEvaluation:
    """K1m〜K7m 評価結果."""

    metric: str
    observed: float
    threshold: float
    pass_: bool
    note: str = ""
    # OBS000005 差し戻し1 対応: バックテストのみでは判定不能な指標 (K6m: フォワード
    # 未実施, K7m: 両建てロジック未統合, permutation: 未実行) を「不合格」ではなく
    # 「判定対象外」として扱うためのフラグ。evaluate() の failed_kpis 集計からは除外する。
    applicable: bool = True


class Stats(TypedDict, total=False):
    """撤退判定に必要な統計量 (SYS-FX007 用に拡張).

    K1m〜K7m 評価用のフィールド名は backtest.metrics.to_dict() の出力キーに
    合わせてある (OBS000005 差し戻し1: 判定エンジンをバックテストパイプラインに
    直結できるようにするため)。
    """

    strategy_id: str  # 例: "SYS-FX007"
    n_days: int
    n_trades: int
    sharpe: float  # Day30/60/90 判定用の総合シャープ (通常は sharpe_monthly と同値)
    max_dd: float  # Day30/60/90 判定表示用 (口座 %)
    regime: str  # "TREND" / "RANGE" / "RANGE_WARN" / "INSUFFICIENT_DATA"
    # K1m〜K7m + 統計的有意性 (backtest.metrics.to_dict() のフィールド名に合わせる)
    sharpe_monthly: float
    profit_factor_monthly: float
    expectancy_jpy: float
    max_dd_monthly_pct: float
    max_dd_yearly_pct: float
    payoff_ratio: float
    max_consecutive_losses: int
    edge_per_trade_jpy: float
    spread_round_trip_jpy: float
    max_margin_usage_pct: float
    weak_breakout_exclusion_pct: float
    backtest_forward_divergence_pct: float | None  # フォワード未実施なら None (K6m 判定対象外)
    permutation_p_value: float | None  # permutation test 未実行なら None (判定対象外)
    n_trades_per_currency: dict[str, int]  # 通貨別取引数
    hedging_enabled: bool  # 両建てロジックが実際に稼働したか (K7m の有効性判定用)
    n_trades_effective: float  # 実効トレード数 (2026-08-18 提案5対応、明示指定時はこちらを優先)
    win_rate: float  # 勝率 (2026-08-21 T-08対応、K3mのスケール不変判定に使用。未指定なら旧来の絶対件数判定にフォールバック)


def compute_n_trades_effective(
    n_trades_per_currency: dict[str, int] | None,
    n_trades_fallback: int,
    *,
    apply_correlation_discount: bool = True,
) -> float:
    """通貨間相関を考慮した実効トレード数を算出する (2026-08-18 提案5対応).

    `n_trades_per_currency` から実際にトレードが発生した通貨の組み合わせを特定し、
    `backtest.permutation.effective_pair_count()` (market_character.json の実測相関から
    N_eff = k/(1+(k-1)*rho_bar) で算出) を使って名目トレード数を実効値へ縮小する。
    通貨別内訳が無い場合 (単一通貨評価など) は名目値をそのまま返す (後方互換)。

    apply_correlation_discount (2026-08-21、外部レビューT-07対応): 既定Trueは
    従来通りの挙動 (SYS-FX007〜010等、既存の判定結果を再現する用途の後方互換)。

    外部レビュー(F2)・C査読(`research/EXP-FX000005/20-c-review.md`)が共通して
    指摘した通り、通貨間相関による罰則を「実効トレード数(この関数)」と
    「permutation検定(`permutation_test_clustered`)」の両方に別々に適用すると、
    同じ相関構造に対して二重にペナルティが掛かる(min_n_trades=300に対し
    4通貨では名目1,027件を要求する一方、検定側もp値の下限が0.3158に張り付く)。

    T-07の方針: 相関補正は「検定」と「n」のどちらか一方にのみ入れる。新規の
    判定で`permutation_test_block()`(クラスタ=エントリー日、外部レビューT-06
    対応)を使う場合は、依存構造が検定側(ブロック順列のクラスタリング)で
    直接捕捉されるため、`apply_correlation_discount=False`を指定してnを
    名目値のまま扱うこと(相関ハンデを二重計上しない)。旧`permutation_test_
    clustered()`(通貨ペア相関行列によるガウス・コピュラ)を使い続ける場合は、
    そちらは依存構造を「検定」側で明示的に扱わないため、引き続き
    `apply_correlation_discount=True`(既定)でnを縮小しておくこと。
    """
    if not n_trades_per_currency:
        return float(n_trades_fallback)
    active = {pair: n for pair, n in n_trades_per_currency.items() if n > 0}
    n_total = sum(active.values()) if active else float(n_trades_fallback)
    if not apply_correlation_discount:
        return float(n_total)
    if len(active) <= 1:
        return float(n_total)
    n_pairs = len(active)
    eff_count = effective_pair_count(list(active.keys()))
    return n_total * (eff_count / n_pairs)


def _max_consecutive_true(flags: list[bool] | np.ndarray) -> int:
    """bool 配列中の最長連続 True 区間の長さを返す (K3m 用の最大連敗長カウンタ)."""
    best = cur = 0
    for flag in flags:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best


def compute_k3m_scale_invariant(
    n_trades: int,
    win_rate: float,
    observed_max_consecutive_losses: int,
    *,
    reps: int = 3000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """K3m(最大連続損失)のスケール不変な再定義 (T-08対応、2026-08-21).

    外部レビュー(2026-08-20)が指摘: 「最大連続損失 ≤ 5」は絶対件数のため n に依存し、
    観測勝率の i.i.d. 帰無仮説の下でも通過率が約6割前後になる(=観測値がランダムと
    区別できない)。T-16再査読(2026-08-21)は、この指摘が外部レビュー以来一度も
    `decision/criteria.py` に反映されず優先順位トラッキングから漏れていたことを
    新規発見した。

    再定義: 同じ n・観測勝率を持つ i.i.d. 系列(独立なベルヌーイ試行)で最大連敗を
    Monte Carlo シミュレートし、観測された最大連敗がその帰無分布の何パーセンタイル
    に位置するかを算出する。絶対件数ではなく相対位置で評価するためスケール不変。

    事前登録(結果を見る前に固定): 片側検定として、観測された最大連敗が i.i.d.
    帰無分布の上位 `alpha`(既定5%、他の統計的有意性判定=permutation p値と同じ
    水準に揃える)に入る場合のみ不合格とする。すなわち「ランダムな連敗の分布より
    明確に悪い(長い連敗が続く)場合」のみ罰則を科し、良い/ランダムと区別が
    つかない場合はいずれも合格とする(旧来の絶対件数閾値のような機械的な足切りは
    行わない)。

    `scripts/verify_external_review_findings.py` の `check_k3m_scale_dependence()`
    が診断目的で使っていたロジックと同一の Monte Carlo 手法をここに一本化し、
    判定エンジン(`evaluate_kpis()`)からも呼び出せるようにした。
    """
    rng = np.random.default_rng(seed)
    runs = np.array([
        _max_consecutive_true(list(rng.random(n_trades) >= win_rate))
        for _ in range(reps)
    ]) if n_trades > 0 else np.array([0])
    percentile = float((runs < observed_max_consecutive_losses).mean())
    passed = percentile <= (1.0 - alpha)
    return {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "observed_max_consecutive_losses": observed_max_consecutive_losses,
        "iid_null_mean": round(float(runs.mean()), 2),
        "iid_null_median": int(np.median(runs)),
        "observed_percentile_in_null": round(percentile, 4),
        "alpha": alpha,
        "pass_": passed,
    }


def load_regime() -> str:
    """regime_latest.json からレジームを読み込み (親PJ踏襲)."""
    if not REGIME_LATEST.exists():
        return "INSUFFICIENT_DATA"
    try:
        data = json.loads(REGIME_LATEST.read_text(encoding="utf-8"))
        return str(data.get("regime", "INSUFFICIENT_DATA"))
    except (json.JSONDecodeError, OSError):
        return "INSUFFICIENT_DATA"


def is_range_regime(regime: str) -> bool:
    """レンジ相場判定 (RANGE 確定, 親PJ踏襲)."""
    return regime == "RANGE"


def evaluate_kpis(stats: Stats) -> list[KPIEvaluation]:
    """K1m〜K7m + 統計的有意性 評価 (本PJ固有).

    OBS000005 差し戻し1 (独立監査追記2 での明記事項): K5m・K6m・min_n_trades・
    permutation_p_value を含む spec 記載の全ゲートを評価する。バックテストのみ
    では原理的に判定できない指標 (K6m: フォワード比較, K7m: 両建て未統合時,
    permutation: 未実行時) は `applicable=False` として「判定対象外」を明示し、
    黙って pass 扱いにしない。
    """
    strategy_id = stats.get("strategy_id", "")
    thresholds = KPI_THRESHOLDS.get(strategy_id)
    if not thresholds:
        return []

    evals: list[KPIEvaluation] = []

    # K1m: 月次 Sharpe
    sharpe = float(stats.get("sharpe_monthly", stats.get("sharpe", 0.0)))
    evals.append(
        KPIEvaluation(
            "K1m_sharpe",
            sharpe,
            thresholds["sharpe_monthly"],
            sharpe >= thresholds["sharpe_monthly"],
        )
    )

    # K1m: 月次 PF
    pf = float(stats.get("profit_factor_monthly", 0.0))
    evals.append(
        KPIEvaluation(
            "K1m_pf",
            pf,
            thresholds["profit_factor_monthly"],
            pf >= thresholds["profit_factor_monthly"],
        )
    )

    # K1m: 月次期待値 > 0円
    expectancy = float(stats.get("expectancy_jpy", 0.0))
    evals.append(
        KPIEvaluation(
            "K1m_expectancy",
            expectancy,
            0.0,
            expectancy > 0.0,
            "1トレードあたり期待値(円)",
        )
    )

    # K2m: 月間 DD (2026-08-21 T-10対応: DD定義ルールは
    # `backtest.metrics.max_drawdown()`のdocstring・`PJ000004`「Q9」参照。
    # 固定ロットサイジングの戦略はこのまま`stats`に初期資金比の値を渡す。
    # 複利サイジングの戦略は呼び出し側で`peak_relative_max_dd_pct()`系の値を
    # 算出してから`max_dd_monthly_pct`にセットすること(この関数自体は
    # どちらの定義で計算された値が渡されても同じ閾値と比較するのみ)。
    max_dd_m = abs(float(stats.get("max_dd_monthly_pct", stats.get("max_dd", 0.0))))
    evals.append(
        KPIEvaluation(
            "K2m_dd_monthly",
            max_dd_m,
            thresholds["max_dd_monthly_pct"],
            max_dd_m <= thresholds["max_dd_monthly_pct"],
            "証拠金 % に対する比率",
        )
    )

    # K2m: 年間 DD
    max_dd_y = abs(float(stats.get("max_dd_yearly_pct", max_dd_m)))
    evals.append(
        KPIEvaluation(
            "K2m_dd_yearly",
            max_dd_y,
            thresholds["max_dd_yearly_pct"],
            max_dd_y <= thresholds["max_dd_yearly_pct"],
            "証拠金 % に対する比率",
        )
    )

    # K3m: 最大連続損失 (2026-08-21 T-08対応: win_rate が指定されていればスケール
    # 不変なi.i.d.パーセンタイル判定に切り替える。外部レビューが指摘した通り、絶対
    # 件数閾値(旧来の挙動)はnに依存し観測勝率のi.i.d.帰無仮説でも通過率が約6割
    # 前後になる=ほぼノイズと区別がつかない。win_rate 未指定の場合は既存呼び出し
    # (テストケース等)との後方互換のため旧来の絶対件数判定を維持する。)
    mcl = int(stats.get("max_consecutive_losses", 999))
    win_rate = stats.get("win_rate")
    n_trades_for_k3m = int(stats.get("n_trades", 0))
    if win_rate is not None and n_trades_for_k3m > 0:
        k3m = compute_k3m_scale_invariant(n_trades_for_k3m, float(win_rate), mcl)
        evals.append(
            KPIEvaluation(
                "K3m_max_consecutive_losses",
                float(mcl),
                float(thresholds["max_consecutive_losses"]),
                k3m["pass_"],
                f"スケール不変判定(T-08): i.i.d.帰無分布パーセンタイル={k3m['observed_percentile_in_null']:.3f}"
                f" (n={n_trades_for_k3m}, win_rate={win_rate:.3f}, 不合格は上位{k3m['alpha']*100:.0f}%のみ)",
            )
        )
    else:
        evals.append(
            KPIEvaluation(
                "K3m_max_consecutive_losses",
                float(mcl),
                thresholds["max_consecutive_losses"],
                float(mcl) <= thresholds["max_consecutive_losses"],
                "win_rate未指定のため旧来の絶対件数判定にフォールバック(T-08未適用)",
            )
        )

    # K4m: ペイオフレシオ
    pr = float(stats.get("payoff_ratio", 0.0))
    evals.append(
        KPIEvaluation(
            "K4m_payoff",
            pr,
            thresholds["payoff_ratio"],
            pr >= thresholds["payoff_ratio"],
        )
    )

    # K5m: 1トレード期待値 > スプレッド往復コスト × 3
    edge = float(stats.get("edge_per_trade_jpy", stats.get("expectancy_jpy", 0.0)))
    spread_rt = float(stats.get("spread_round_trip_jpy", 0.0))
    if spread_rt > 0:
        multiple = edge / spread_rt
        evals.append(
            KPIEvaluation(
                "K5m_spread_cost_multiple",
                multiple,
                thresholds["spread_cost_multiple"],
                multiple >= thresholds["spread_cost_multiple"],
                f"edge={edge:.1f}円 / spread往復={spread_rt:.1f}円",
            )
        )
    else:
        evals.append(
            KPIEvaluation(
                "K5m_spread_cost_multiple",
                0.0,
                thresholds["spread_cost_multiple"],
                False,
                "spread_round_trip_jpy 未提供のため判定対象外",
                applicable=False,
            )
        )

    # K6m: バックテスト ↔ フォワードテスト乖離率 (フォワード未実施なら判定対象外)
    bt_ft = stats.get("backtest_forward_divergence_pct")
    if bt_ft is None:
        evals.append(
            KPIEvaluation(
                "K6m_bt_ft_divergence",
                float("nan"),
                thresholds["backtest_forward_divergence_pct"],
                False,
                "フォワードテスト未実施のため判定対象外",
                applicable=False,
            )
        )
    else:
        bt_ft = float(bt_ft)
        evals.append(
            KPIEvaluation(
                "K6m_bt_ft_divergence",
                bt_ft,
                thresholds["backtest_forward_divergence_pct"],
                bt_ft <= thresholds["backtest_forward_divergence_pct"],
            )
        )

    # K7m: 両建て証拠金消費率 (両建てロジック未統合の間は「判定対象外」)
    margin = float(stats.get("max_margin_usage_pct", 0.0))
    if stats.get("hedging_enabled"):
        evals.append(
            KPIEvaluation(
                "K7m_margin_usage",
                margin,
                thresholds["max_margin_usage_pct"],
                margin <= thresholds["max_margin_usage_pct"],
            )
        )
    else:
        evals.append(
            KPIEvaluation(
                "K7m_margin_usage",
                margin,
                thresholds["max_margin_usage_pct"],
                margin <= thresholds["max_margin_usage_pct"],
                "両建てロジック未統合のため単一ポジションの証拠金比率のみ。K7m の意図（両建て時消費率）は未検証",
                applicable=False,
            )
        )

    # v2 拡張: 弱いブレイク排除率
    we = float(stats.get("weak_breakout_exclusion_pct", 0.0))
    evals.append(
        KPIEvaluation(
            "v2_weak_breakout_exclusion",
            we,
            thresholds["weak_breakout_exclusion_pct"],
            we >= thresholds["weak_breakout_exclusion_pct"],
        )
    )

    # 統計的有意性: 有効サンプル数 (2026-08-18 提案5対応: 名目→実効トレード数へ切替え)
    n_trades = int(stats.get("n_trades", 0))
    n_trades_effective = stats.get("n_trades_effective")
    if n_trades_effective is None:
        n_trades_effective = compute_n_trades_effective(stats.get("n_trades_per_currency"), n_trades)
    else:
        n_trades_effective = float(n_trades_effective)
    note = "" if n_trades_effective == n_trades else f"名目n_trades={n_trades}(通貨間相関を考慮し実効値へ縮小)"
    evals.append(
        KPIEvaluation(
            "min_n_trades",
            round(n_trades_effective, 2),
            thresholds["min_n_trades"],
            n_trades_effective >= thresholds["min_n_trades"],
            note,
        )
    )

    # 統計的有意性: permutation p 値 (未実行なら判定対象外)
    p_value = stats.get("permutation_p_value")
    if p_value is None:
        evals.append(
            KPIEvaluation(
                "permutation_p_value",
                float("nan"),
                thresholds["permutation_p_value"],
                False,
                "permutation test 未実行のため判定対象外",
                applicable=False,
            )
        )
    else:
        p_value = float(p_value)
        evals.append(
            KPIEvaluation(
                "permutation_p_value",
                p_value,
                thresholds["permutation_p_value"],
                p_value < thresholds["permutation_p_value"],
            )
        )

    return evals


def kpi_pass_summary(evals: list[KPIEvaluation]) -> dict:
    """KPI 評価一覧を件数集計に変換 (判定対象外は分母・分子から除外して明示)."""
    applicable = [e for e in evals if e.applicable]
    not_applicable = [e for e in evals if not e.applicable]
    passed = [e for e in applicable if e.pass_]
    return {
        "total": len(evals),
        "applicable": len(applicable),
        "not_applicable": len(not_applicable),
        "not_applicable_metrics": [e.metric for e in not_applicable],
        "pass": len(passed),
        "fail": len(applicable) - len(passed),
        "fail_metrics": [e.metric for e in applicable if not e.pass_],
        "all_applicable_pass": len(passed) == len(applicable) and len(applicable) > 0,
    }


def evaluate(stats: Stats) -> tuple[Verdict, str]:
    """撤退/採用判定 (SYS-FX007 レンジブレイク・プルバック戦略).

    K1m〜K7m すべて満たせば GO、Day ベースの特例判定あり。
    """
    strategy_id = stats.get("strategy_id", "")
    n_days = int(stats.get("n_days", 0))
    n_trades = int(stats.get("n_trades", 0))
    sharpe = float(stats.get("sharpe", 0.0))
    regime = stats.get("regime") or load_regime()

    go_threshold = STRATEGY_GO_THRESHOLD.get(strategy_id, 0.4)

    # 取引数 0 → SAMPLE-DEFICIT
    if n_trades == 0:
        return Verdict.SAMPLE_DEFICIT, f"取引数 0 (Day {n_days})"

    # 取引数 < 3 (Day 90 経過後) → 06 ヶ月延長オプション
    if n_days >= 90 and n_trades < SAMPLE_DEFICIT_N_TRADES:
        return (
            Verdict.SAMPLE_DEFICIT,
            f"取引数 {n_trades} < {SAMPLE_DEFICIT_N_TRADES} (Day {n_days} 経過) → 06 ヶ月延長検討",
        )

    # K1m〜K7m 評価
    kpi_evals = evaluate_kpis(stats)
    # 判定対象外 (applicable=False) の指標は不合格扱いにしない (OBS000005 差し戻し1)
    failed_kpis = [e.metric for e in kpi_evals if e.applicable and not e.pass_]

    # Day 30 まで: 短期撤退閾値
    if n_days < 30:
        if sharpe < DAY30_SHARPE_REJECT:
            return (
                Verdict.REJECT,
                f"短期: Sharpe {sharpe:.2f} < {DAY30_SHARPE_REJECT} (Day {n_days})",
            )
        if sharpe < 0 and is_range_regime(regime):
            return (
                Verdict.REGIME_EXTEND,
                f"レンジ相場例外: Sharpe {sharpe:.2f} < 0 だが regime={regime} で評価延長",
            )
        if sharpe >= go_threshold and not failed_kpis:
            return Verdict.GO, f"早期 GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
        if failed_kpis:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f}, KPI 失敗: {','.join(failed_kpis)}",
            )
        return Verdict.WATCH, f"Day {n_days}: Sharpe {sharpe:.2f}"

    # Day 30-60: 中期評価
    if n_days < 60:
        if sharpe >= go_threshold and not failed_kpis:
            return Verdict.GO, f"GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
        if sharpe < 0 and is_range_regime(regime):
            return (
                Verdict.REGIME_EXTEND,
                f"レンジ相場例外: Sharpe {sharpe:.2f} < 0 だが regime={regime} で評価延長",
            )
        if failed_kpis:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f}, KPI 失敗: {','.join(failed_kpis)}",
            )
        return Verdict.WATCH, f"Day {n_days}: Sharpe {sharpe:.2f}"

    # Day 60-90: 中期撤退閾値
    if n_days < 90:
        if sharpe >= go_threshold and not failed_kpis:
            return Verdict.GO, f"GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
        if sharpe < DAY60_SHARPE_REJECT and is_range_regime(regime):
            return (
                Verdict.REJECT,
                f"中期 REJECT: Sharpe {sharpe:.2f} < {DAY60_SHARPE_REJECT} + regime={regime}",
            )
        if sharpe < DAY60_SHARPE_REJECT:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f} < {DAY60_SHARPE_REJECT} 注視",
            )
        if failed_kpis:
            return (
                Verdict.WATCH,
                f"Day {n_days}: Sharpe {sharpe:.2f}, KPI 失敗: {','.join(failed_kpis)}",
            )
        return Verdict.WATCH, f"Day {n_days}: Sharpe {sharpe:.2f}"

    # Day 90+: 本採用判定 (KPI すべて pass 必須)
    if sharpe >= go_threshold and not failed_kpis:
        return Verdict.GO, f"GO: Sharpe {sharpe:.2f} >= {go_threshold} (Day {n_days})"
    if failed_kpis:
        return (
            Verdict.REJECT,
            f"本採用見送り: KPI 失敗 {','.join(failed_kpis)} (Day {n_days})",
        )
    return (
        Verdict.REJECT,
        f"本採用見送り: Sharpe {sharpe:.2f} < {go_threshold} (Day {n_days})",
    )


def main() -> int:
    """CLI エントリポイント: SYS-FX007 テストケース."""
    test_cases: list[Stats] = [
        {
            "strategy_id": "SYS-FX007",
            "n_days": 25,
            "n_trades": 5,
            "sharpe": -0.6,
            "sharpe_monthly": -0.6,
            "max_dd": 8.0,
            "max_dd_monthly_pct": 8.0,
            "max_dd_yearly_pct": 8.0,
            "regime": "TREND",
            "profit_factor_monthly": 1.3,
            "expectancy_jpy": 50.0,
            "payoff_ratio": 1.8,
            "max_consecutive_losses": 3,
            "edge_per_trade_jpy": 50.0,
            "spread_round_trip_jpy": 20.0,
            "max_margin_usage_pct": 15.0,
            "weak_breakout_exclusion_pct": 35.0,
        },
        {
            "strategy_id": "SYS-FX007",
            "n_days": 95,
            "n_trades": 80,
            "sharpe": 0.5,
            "sharpe_monthly": 0.5,
            "max_dd": 7.0,
            "max_dd_monthly_pct": 7.0,
            "max_dd_yearly_pct": 12.0,
            "regime": "TREND",
            "profit_factor_monthly": 1.4,
            "expectancy_jpy": 120.0,
            "payoff_ratio": 2.0,
            "max_consecutive_losses": 4,
            "edge_per_trade_jpy": 120.0,
            "spread_round_trip_jpy": 20.0,
            "max_margin_usage_pct": 22.0,
            "weak_breakout_exclusion_pct": 40.0,
            "permutation_p_value": 0.02,
        },
        {
            "strategy_id": "SYS-FX007",
            "n_days": 95,
            "n_trades": 2,
            "sharpe": 0.3,
            "sharpe_monthly": 0.3,
            "max_dd": 5.0,
            "max_dd_monthly_pct": 5.0,
            "max_dd_yearly_pct": 5.0,
            "regime": "TREND",
            "profit_factor_monthly": 1.1,
            "expectancy_jpy": 10.0,
            "payoff_ratio": 1.5,
            "max_consecutive_losses": 2,
            "edge_per_trade_jpy": 10.0,
            "spread_round_trip_jpy": 20.0,
            "max_margin_usage_pct": 10.0,
            "weak_breakout_exclusion_pct": 30.0,
        },
    ]
    for stats in test_cases:
        verdict, reason = evaluate(stats)
        print(
            f"{stats['strategy_id']} Day {stats['n_days']:>3} "
            f"Sharpe {stats['sharpe']:>5.2f} regime={stats['regime']:<6} "
            f"-> {verdict.value:<18} {reason}"
        )
    print(f"\n[decision_criteria] ts={datetime.now(UTC).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
