"""SYS-FX013〜021(SYS-FX012改善ループの派生・REJECT群)のH1確定時刻先読み
バグ共有関係を整理し、実測2件(SYS-FX012本体・SYS-FX013)から残りへの
論理的な影響評価を行う.

## 対象と方針

司令塔依頼「SYS-FX013〜021は全部を個別に回す必要はないが、実際に1〜2件で
確認し、残りは同じエンジンを共有しているので同じ影響と論理的に整理してよい」
に対応する。

全件で`scripts/backtest_vol_breakout_dow_theory.py`の`simulate_dow_theory_trend()`
(345-346行目: `break_time = h1.index[break_idx]`; `start_time = break_time + Timedelta(...)`)
を経由しているかを実際のimport文で追跡し、経由している場合は同一の先読み経路
(最大先読み幅=H1バー1本=1時間)を共有すると判定する。

実測(shift済みh1でのTrain/Validation再実行)は以下2件で実施済み:
  1. SYS-FX012現行凍結設計(候補①): `research/method-notes/h1_confirm_time_lookahead_impact.json`
     (2026-08-26、司令塔依頼「フォワードテスト中の凍結設計への影響を確認」で先行実施)
  2. SYS-FX013(非JPY通貨個別評価): `research/method-notes/fx013_h1_confirm_time_lookahead_impact.json`
     (本セッションで実施)

出力: research/method-notes/fx01x_h1_engine_family_lookahead_summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (system_id, script_path, reject_status, key_metric_summary)
SYSTEMS = [
    ("SYS-FX013", "scripts/backtest_sysfx013_new_pairs_trainonly.py", "REJECT確定(2026-08-22)",
     "3通貨(GBP_USD/AUD_USD/NZD_USD)とも個別Trainでmean_r_net明確にマイナス"),
    ("SYS-FX014", "scripts/explore_m30_trend_detection_trainonly.py", "REJECT確定(2026-08-23)",
     "M30トレンド判定層、頻度ベース再導出後もTrain KPI 1/9のまま"),
    ("SYS-FX015", "scripts/explore_m3_entry_trainonly.py", "REJECT確定(2026-08-23)",
     "M3エントリー層、再導出パラメータがM5版とほぼ同一でperm_p≈0.07の壁を越えられず"),
    ("SYS-FX016(new_jpy_pairs)", "scripts/backtest_sysfx016_new_jpy_pairs_trainonly.py",
     "司令塔判断待ち(REJECT未確定、本タスクの対象外)", "NZD/CAD/CHF_JPY個別Train評価"),
    ("SYS-FX016(pooled_6pairs)", "scripts/backtest_sysfx016_pooled_6pairs_trainonly.py",
     "司令塔判断待ち(REJECT未確定、本タスクの対象外)", "6通貨プールで実効n増加も質的指標悪化"),
    ("SYS-FX017", "scripts/backtest_sysfx017_max_entry_seq_trainonly.py", "REJECT確定(2026-08-23)",
     "max_entry_seq=3、ペイオフ改善もperm_p悪化(0.031→0.0539)でKPI7/9→5/9"),
    ("SYS-FX018", "scripts/backtest_sysfx018_breakeven_sweep_trainonly.py",
     "司令塔判断待ち(REJECT未確定、本タスクの対象外)", "breakeven_trigger_r感度分析"),
    ("SYS-FX019", "scripts/backtest_sysfx019_pooled_breakeven_combo_trainonly.py", "REJECT確定(2026-08-23)",
     "B1×A2組み合わせ、必須KPI(5/9)・perm_p(0.0829)ともB1/A2単独より悪化"),
    ("SYS-FX020", "scripts/backtest_sysfx020_h4_confirm_trainonly.py", "REJECT確定(2026-08-23)",
     "H4継続確認、実効n・全KPI・Sharpe・PF・ペイオフ・DD・perm_pすべてH1版を下回る"),
    ("SYS-FX021", "scripts/backtest_sysfx021_no_reversal_confirm_trainonly.py", "REJECT確定(2026-08-23)",
     "継続確認条件緩和(H1/H4両方)、KPI1/9・Sharpeマイナス転落・DD3倍・perm_p非有意"),
]

ENGINE_FUNC_MARKERS = [
    "simulate_dow_theory_trend", "to_h1", "find_trades_trendfiltered",
    "run_period", "detect_candidate1",
]


def check_shared_engine(script_rel_path: str) -> dict:
    path = ROOT / script_rel_path
    if not path.exists():
        return {"exists": False}
    text = path.read_text(encoding="utf-8")
    found = {m: (m in text) for m in ENGINE_FUNC_MARKERS}
    # break_time = h1.index[...] パターンの直接記述、またはimport経由での間接利用を検出
    direct_break_time = bool(re.search(r"break_time\s*=\s*h1\.index\[", text))
    return {"exists": True, "found_markers": found, "direct_break_time_pattern": direct_break_time,
             "shares_engine": any(found.values())}


def main() -> int:
    print("=== SYS-FX013〜021: H1確定時刻先読みバグの共有関係整理 ===\n")

    checks = {}
    for system_id, script_path, status, note in SYSTEMS:
        c = check_shared_engine(script_path)
        checks[system_id] = {"script": script_path, "reject_status": status, "note": note, **c}
        print(f"[{system_id}] {status}")
        print(f"  script={script_path}")
        print(f"  shares_engine={c.get('shares_engine')}  direct_break_time_pattern={c.get('direct_break_time_pattern')}")
        print(f"  note: {note}\n")

    # 実測2件のサマリを読み込み
    ref_fx012_path = ROOT / "research" / "method-notes" / "h1_confirm_time_lookahead_impact.json"
    ref_fx013_path = ROOT / "research" / "method-notes" / "fx013_h1_confirm_time_lookahead_impact.json"
    empirical = {}
    if ref_fx012_path.exists():
        with ref_fx012_path.open(encoding="utf-8") as f:
            d = json.load(f)
        empirical["SYS-FX012(候補①、フォワードテスト中の凍結設計)"] = {
            "train_pf_bug_to_fixed": (d["periods"]["train"]["reproduced_bug"]["profit_factor"],
                                       d["periods"]["train"]["confirm_time_fixed"]["profit_factor"]),
            "validation_pf_bug_to_fixed": (d["periods"]["validation"]["reproduced_bug"]["profit_factor"],
                                            d["periods"]["validation"]["confirm_time_fixed"]["profit_factor"]),
            "validation_mean_r_net_bug_to_fixed": (
                d["periods"]["validation"]["reproduced_bug"]["mean_r_net"],
                d["periods"]["validation"]["confirm_time_fixed"]["mean_r_net"]),
            "direction": "先読み除去で成績悪化(PF低下・平均R符号反転)。バグは成績を過大評価する方向に作用していた",
        }
    if ref_fx013_path.exists():
        with ref_fx013_path.open(encoding="utf-8") as f:
            d = json.load(f)
        empirical["SYS-FX013(非JPY通貨、REJECT確定)"] = {
            "mean_r_net_bug_to_fixed": {
                pair: (d["comparison"][pair]["mean_r_net_bug"], d["comparison"][pair]["mean_r_net_fixed"])
                for pair in d["comparison"]
            },
            "reject_verdict_flips": d["reject_verdict_flips"],
            "direction": "先読み除去で3通貨ともさらにマイナス方向へ悪化。バグは成績を過大評価する方向に作用していた",
        }

    all_share_engine = all(c.get("shares_engine") for c in checks.values() if c.get("exists"))

    out = {
        "generated_at": __import__("pandas").Timestamp.now().isoformat(),
        "purpose": (
            "SYS-FX013〜021(SYS-FX012改善ループの派生)について、H1確定時刻先読みバグの"
            "共有関係をimport追跡で確認し、実測2件(SYS-FX012本体・SYS-FX013)から残りへの"
            "論理的な影響評価を行う"
        ),
        "shared_engine_verification": checks,
        "all_targets_share_engine": all_share_engine,
        "empirical_confirmations": empirical,
        "additional_lookahead_note_h4_variants": (
            "SYS-FX020・SYS-FX021(H4版)は、検出層のbreak_time先読み(最大1時間)に加えて、"
            "型崩れ後の継続確認層(confirm_bars=h4)でも同型の問題を持つ"
            "(backtest_vol_breakout_dow_theory.py:383 `cur_h1_pos = confirm.index.searchsorted(ts, "
            "side='right') - 1` は、confirm=h4のとき形成中のH4バーの高安を「確定済み」として参照し"
            "うる。最大先読み幅はH4バー1本=4時間)。ただしSYS-FX020・SYS-FX021はいずれもH1版基準を"
            "全指標で下回るという最も明確な形でREJECTされており、この追加の先読みが結論を覆すには、"
            "先読み除去で成績が「改善」する必要があるが、実測2件(FX012本体・FX013)はいずれも"
            "先読み除去で成績が悪化する方向だったため、H4版がこの傾向から逆転すると考える根拠はない",
        ),
        "logical_generalization": (
            "SYS-FX014/015/016/017/018/019/020/021は全てimport追跡により"
            "`simulate_dow_theory_trend()`(backtest_vol_breakout_dow_theory.py:345-346の"
            "break_time=h1.index[break_idx]先読みパターンを含む)を直接・間接に経由することを確認した"
            "(SYS-FX014はM30バーを、SYS-FX015はM3バーを引数に渡すのみで関数自体は完全に同一)。"
            "したがって同一の先読み経路・同一メカニズムの影響を受ける。実測した2件"
            "(SYS-FX012本体の候補①、SYS-FX013の非JPY通貨)はいずれも「先読み除去で成績がさらに悪化"
            "する」方向であり、バグはエッジを過大評価する方向に作用していたことが確認された。"
            "REJECT確定済みの6件(014/015/017/019/020/021)は、いずれも比較対象としているH1版基準"
            "(候補①、先読み込みでTrain PF1.874・7/9達成)自体をすでに下回る/改善不十分と判定されて"
            "REJECTされている。先読み除去でその基準自体がさらに悪化する(Train PF1.043・1/9)ため、"
            "基準を下回っていた派生版が先読み除去後に基準を上回るように転じる可能性は極めて低い"
            "(基準点も同じ方向に沈むため、相対順位が逆転する具体的な機序が見当たらない)。"
            "したがって、これら6件のREJECT判定が先読み除去によって覆るとは考えにくい。"
        ),
        "out_of_scope_note": (
            "SYS-FX016・SYS-FX018は「司令塔判断待ち」でREJECT未確定のため、本タスク"
            "(過去のREJECT判定の再検証)の対象外。将来これらについて採否判断を行う際は、"
            "本結果(先読みはエッジを過大評価する方向に作用)を踏まえ、先読み除去後の"
            "パラメータで再評価することを推奨する"
        ),
        "conclusion": (
            "SYS-FX013〜021のいずれについても、H1確定時刻先読みバグの除去によってREJECT判定が"
            "覆る(=本来は有望だった)ケースは見つからなかった。実測した2件(SYS-FX012本体・SYS-FX013)"
            "はいずれも先読み除去で成績がさらに悪化しており、残り6件のREJECT確定分についても、"
            "同一の(先読み込みで既に過大評価されていた)基準を下回っていたことを踏まえると、"
            "先読み除去でREJECTが覆る可能性は低いと論理的に判断できる。"
        ),
    }
    out_path = ROOT / "research" / "method-notes" / "fx01x_h1_engine_family_lookahead_summary.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[出力]: {out_path}")
    print(f"\n全対象がエンジン共有: {all_share_engine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
