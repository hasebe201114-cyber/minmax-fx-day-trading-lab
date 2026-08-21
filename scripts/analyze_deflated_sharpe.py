"""EXP-FX000005 T-18: 試行回数の一覧化と多重検定補正(Deflated Sharpe Ratio).

外部レビュー(2026-08-20)が指摘: 「司令塔の設計変更指示がそのまま設計変更として
直入りしており、試行回数が数えられていない。少なくとも十数回」。改善ループの
全過程で「結果を見て次の一手を選ぶ」という逐次的な意思決定が繰り返されており、
選ばれた最良候補のSharpe比が、単に多数の候補を試した結果として偶然高く出た
可能性(=選択バイアス)を補正せずに評価してきた。

## 方法: Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)

観測されたSharpe比(SR_hat)が、N回の独立試行の中から最良のものを選んだ場合に
「偶然」で説明できる水準(期待最大Sharpe比 E[max SR_N])をどれだけ上回っているかを、
リターン分布の歪度・尖度を考慮した検定統計量として算出する。

    V[SR_hat] = (1 - γ3・SR_hat + (γ4-1)/4・SR_hat^2) / (T-1)
    E[max SR_N] = sqrt(V[SR_hat]) * [(1-e_m)Φ^-1(1-1/N) + e_m・Φ^-1(1-1/(N・e))]
    DSR = Φ( (SR_hat - E[max SR_N]) / sqrt(V[SR_hat]) )

    γ3=歪度, γ4=尖度(正規分布=3), T=リターン観測数, N=独立試行数,
    e_m≈0.5772(オイラー・マスケローニ定数), e≈2.71828

DSRは「真のSharpe比が0を上回っている確率」であり、Sharpe比そのものではない。

## 事前登録(結果を見る前に固定)

試行回数Nの数え方: `00-spec.md`・`research/ACTIVE.md`の決定履歴を通読し、
以下の基準で「独立試行」を判定する(結果を見てから数え方を変えない)。

- **含める**: Train(またはTrain+Validation)の観測結果を見てから「採用する/しない」
  を決めた自由な設計・パラメータの選択(グリッドサーチの各点を含む)
- **含めない**: 既知の実装バグの修正(観測結果からではなく、コードとspecの記述の
  乖離自体から発覚したもの)、統計的検定手法そのものの是正(検出力の議論に基づく
  ものであり、Sharpe比を上げるための探索ではない)、コスト計算の実務的な正確性
  是正

この基準は主観の余地が残るため、**保守的カウント(N_conservative)と網羅的カウント
(N_liberal)の両方を算出し、DSRが両方でどう変わるかを報告する**(単一の数値に
過度な精度を持たせない)。

出力: research/method-notes/deflated_sharpe_analysis.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

EULER_MASCHERONI = 0.5772156649015329

# --------------------------------------------------------------------------
# 試行回数の一覧化(事前登録、00-spec.md・research/ACTIVE.mdの決定履歴から集計)
# --------------------------------------------------------------------------
TRIAL_LOG = [
    # (名称, 種別, カウント対象か, 備考)
    ("Q39 単発版ベースライン", "design", True, "Train評価、mean_R=-0.242で棄却"),
    ("Q40 改善ループ第1試行(M5ダウ理論連続追跡への設計変更)", "design", True, "Train評価、mean_R符号がプラスに転換し採用"),
    ("Q49 改善ループ第2試行(4通貨への絞り込み、EUR_USD除外)", "design", True, "Train単独結果で除外通貨を選定"),
    ("Q50 改善ループ第3試行(BOJ/FOMCブラックアウト窓の追加)", "design", True, "Train/Validation/Testで評価し採用"),
    ("Q51 改善ループ第4試行 Part A(N閾値×ZigZag閾値グリッドサーチ)", "grid", 9, "9通り、n_eff最大の候補を選定(結局不採用)"),
    ("Q51 改善ループ第4試行 Part B(カレンダー窓のグリッドサーチ)", "grid", 6, "6通り(結局不採用)"),
    ("Q52 改善ループ第5試行(TP3=4R×初回エントリー除外の組合せ)", "grid", 4, "4通り、Testで悪化と判明し不採用"),
    ("Q52 改善ループ第6試行(TP3=4Rのみ)", "design", True, "3期間で第3試行と同数、僅差で採用"),
    ("Q53 改善ループ第7試行 v1(単一バー極値ショック抑制)", "design", True, "自己検証で検出漏れを発見し破棄(v2へ改訂)"),
    ("Q53 改善ループ第7試行 v2(複数通貨同時ブレイク検知)", "design", True, "Train/Validation/Testで評価し採用、外部レビュー時点の最良候補"),
    ("T-13 出口設計の方向性((a)トレール専業化)", "design", True, "司令塔選択、Train/Validationで評価し採用"),
    ("T-14 通貨ペア拡大(EUR_USD追加、5通貨化)", "design", True, "Train/Validationで評価し明確な負の結果、不採用"),
]

# T-01〜T-12(F1/F3修正・コストモデル確定・検定方式修正・K3m再定義・DD定義・
# 必須参考分離・重複バグ修正・証拠金維持率ストレス・記録規約)は、既知の実装
# バグ修正または統計的検定手法自体の是正であり、「結果を見て複数の候補から
# 良いものを選ぶ」試行には該当しないため、事前登録の基準によりカウントしない。

N_LIBERAL = sum(v if isinstance(v, int) else 1 for _, _, v, _ in TRIAL_LOG if v)
N_CONSERVATIVE = sum(v for _, kind, v, _ in TRIAL_LOG if kind == "grid")  # グリッドサーチのみ(Part A+B+第5試行)


def compute_dsr(sr_hat: float, t_obs: int, skew: float, kurt: float, n_trials: int) -> dict:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) を算出する.

    sr_hat: 観測された非年率化Sharpe比(月次リターンのmean/std)
    t_obs: リターン観測数(月次リターンの本数)
    skew: 観測リターンの歪度
    kurt: 観測リターンの尖度(正規分布=3、非超過)
    n_trials: 独立試行数
    """
    var_sr = (1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2) / (t_obs - 1)
    if var_sr <= 0:
        var_sr = 1e-12
    std_sr = float(np.sqrt(var_sr))

    if n_trials <= 1:
        sr_benchmark = 0.0
    else:
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_benchmark = std_sr * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)

    dsr = float(stats.norm.cdf((sr_hat - sr_benchmark) / std_sr))
    return {
        "n_trials": n_trials,
        "sr_hat_monthly": round(sr_hat, 4),
        "sr_hat_annualized": round(sr_hat * np.sqrt(12), 4),
        "t_observations": t_obs,
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "variance_sr_hat": round(var_sr, 6),
        "expected_max_sr_under_n_trials_monthly": round(sr_benchmark, 4),
        "expected_max_sr_under_n_trials_annualized": round(sr_benchmark * np.sqrt(12), 4),
        "deflated_sharpe_ratio": round(dsr, 4),
        "interpretation": (
            f"真のSharpe比が0を上回っている確率(選択バイアス補正後)は約{dsr*100:.1f}%"
        ),
    }


def main() -> int:
    print("=== EXP-FX000005 T-18: 試行回数の一覧化と多重検定補正 ===\n")

    print("--- 試行一覧 ---")
    for name, kind, count, note in TRIAL_LOG:
        c = count if isinstance(count, int) else 1
        print(f"  [{kind:6}] x{c:<3} {name}")
        print(f"           {note}")
    print(f"\n  N_liberal(網羅的カウント) = {N_LIBERAL}")
    print(f"  N_conservative(グリッドサーチのみ) = {N_CONSERVATIVE}\n")

    with (ROOT / "research" / "method-notes" / "vol_breakout_dow_theory_4pairs_v7_trailonly_1000usd_backtest.json").open(
        encoding="utf-8"
    ) as f:
        backtest = json.load(f)

    eq_curve = pd.DataFrame(backtest["periods"]["train"]["equity_curve"])
    eq_curve["timestamp"] = pd.to_datetime(eq_curve["time"], format="mixed", utc=True).dt.tz_localize(None)
    eq = eq_curve.set_index("timestamp")["balance"].resample("ME").last()
    monthly_returns = eq.pct_change().dropna()

    sr_hat = float(monthly_returns.mean() / monthly_returns.std())
    t_obs = len(monthly_returns)
    skew = float(stats.skew(monthly_returns, bias=False))
    kurt = float(stats.kurtosis(monthly_returns, fisher=False, bias=False))

    print(f"--- Train月次リターン統計 ---")
    print(f"  観測月数={t_obs}  月次SR(非年率)={sr_hat:.4f}  年率SR={sr_hat*np.sqrt(12):.4f}"
          f"  (`monthly_sharpe()`実測値2.940と比較)")
    print(f"  歪度={skew:.4f}  尖度={kurt:.4f}\n")

    result_liberal = compute_dsr(sr_hat, t_obs, skew, kurt, N_LIBERAL)
    result_conservative = compute_dsr(sr_hat, t_obs, skew, kurt, N_CONSERVATIVE)
    result_n1 = compute_dsr(sr_hat, t_obs, skew, kurt, 1)

    print("--- DSR結果 ---")
    for label, r in [("N=1(補正なし、参考)", result_n1),
                      ("N_conservative(グリッドサーチのみ)", result_conservative),
                      ("N_liberal(全試行)", result_liberal)]:
        print(f"  {label}: N={r['n_trials']}  期待最大SR(年率換算)={r['expected_max_sr_under_n_trials_annualized']}"
              f"  DSR={r['deflated_sharpe_ratio']}  ({r['interpretation']})")

    out = {
        "generated_at": datetime.now().isoformat(),
        "trial_log": [{"name": n, "kind": k, "count": (c if isinstance(c, int) else 1), "note": note}
                       for n, k, c, note in TRIAL_LOG],
        "n_liberal": N_LIBERAL,
        "n_conservative": N_CONSERVATIVE,
        "train_monthly_returns": {
            "t_observations": t_obs,
            "sr_hat_monthly": round(sr_hat, 4),
            "sr_hat_annualized": round(sr_hat * np.sqrt(12), 4),
            "skewness": round(skew, 4),
            "kurtosis": round(kurt, 4),
        },
        "dsr_no_correction_n1": result_n1,
        "dsr_conservative": result_conservative,
        "dsr_liberal": result_liberal,
        "_note": (
            "N_conservativeは改善ループ第4試行(Part A 9通り+Part B 6通り)と第5試行"
            "(4通り)の自由パラメータグリッドサーチのみをカウント(最も疑いの余地が"
            "ないもの)。N_liberalは、結果を見てから採用/不採用を決めた設計上の分岐点"
            "(Train単独評価も含む)をすべて含む。既知の実装バグ修正・統計的検定手法"
            "自体の是正(T-01〜T-12等)は、Sharpe比を上げるための自由探索ではないため"
            "いずれのカウントにも含めていない。"
        ),
    }
    out_path = ROOT / "research" / "method-notes" / "deflated_sharpe_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
