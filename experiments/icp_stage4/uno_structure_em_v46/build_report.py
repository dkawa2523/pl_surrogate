#!/usr/bin/env python
"""Build a durable Japanese technical report for the isolated v46 study."""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/icp_conference_materials/uno_structure_em_v46"
LABELS = {
    "formal_dimension": "正式 Dimension",
    "formal_sdf": "正式 SDF",
    "A_multires": "A: 多解像度UNO",
    "AB_separate_fusion": "AB: 入力群別融合",
    "ABC_adaptive_mix": "ABC: 局所・大域適応混合",
    "ABD_shared_coils": "ABD: 共有コイル符号化",
    "ABE_em_shape_amplitude": "ABE: 磁場形状・振幅分離",
    "ABF_structure_delta": "ABF: 構造差分監督",
    "ABG_sdf_em_aux": "ABG: SDF→磁場補助課題",
    "ABH_target_decoders": "ABH: 物理量別デコーダ",
    "ALL_combined": "ALL: 全統合",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: str | float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def main() -> int:
    rows = _read_csv(REPORT / "model_summary.csv")
    paired = _read_csv(REPORT / "paired_effects_vs_formal_sdf.csv")
    validation = json.loads((REPORT / "validation_report.json").read_text(encoding="utf-8"))
    decision = json.loads((REPORT / "decision_summary.json").read_text(encoding="utf-8"))
    values = {row["model"]: row for row in rows}
    candidates = [row for row in rows if row["role"] == "isolated_candidate"]
    best_ni = min(candidates, key=lambda row: float(row["unknown_ni_rel_l2_median_pct"]))
    best_bohm = min(candidates, key=lambda row: float(row["unknown_bohm_rel_l2_median_pct"]))
    formal_sdf = values["formal_sdf"]

    paired_ni = {
        row["model"]: row for row in paired if row["metric"] == "ni_rel_l2_pct"
    }
    best_pair = paired_ni[best_ni["model"]]
    decisive = best_pair["direction"] == "candidate_better"
    claim = (
        f"未知75構造のイオン密度では {LABELS[best_ni['model']]} が最良で、中央値 "
        f"{_fmt(best_ni['unknown_ni_rel_l2_median_pct'])}%（正式SDF "
        f"{_fmt(formal_sdf['unknown_ni_rel_l2_median_pct'])}%）だった。"
    )
    boundary = (
        "同一75ケースの差の95%区間も0未満で、ケース対応で改善が確認できた。"
        if decisive else
        "同一75ケースの差の95%区間は0をまたぐため、1シードで正式版を置換する根拠にはしない。"
    )

    ordered = sorted(rows, key=lambda row: float(row["unknown_ni_rel_l2_median_pct"]))
    table_lines = [
        "|nᵢ順位|モデル|未知 nᵢ L2 中央値|未知 Bohm L2 中央値|構造差分 nᵢ L2|標準テスト|学習時間*|",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ordered, start=1):
        table_lines.append(
            f"|{rank}|{LABELS[row['model']]}|{_fmt(row['unknown_ni_rel_l2_median_pct'])}%|"
            f"{_fmt(row['unknown_bohm_rel_l2_median_pct'])}%|"
            f"{_fmt(row['structural_delta_ni_rel_l2_median_pct'])}%|"
            f"{_fmt(row['standard_test_primary_metric'], 5)}|{_fmt(row['training_elapsed_min'], 1)}分|"
        )
    markdown = f"""# ICP UNO 構造・磁場表現 v46 評価報告

生成日時: {datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')}

## 結論

{claim} {boundary}

Bohmフラックスの最良候補は **{LABELS[best_bohm['model']]}**（中央値 {_fmt(best_bohm['unknown_bohm_rel_l2_median_pct'])}%）だが、正式SDFの {_fmt(formal_sdf['unknown_bohm_rel_l2_median_pct'])}% より悪い。全9候補で正式SDFよりBohmが悪化し、同一ケース差の95%区間も全候補で0より大きかった。したがって総合判断は **正式SDFを維持** である。ハッシュ監査結果は **{validation['status'].upper()}**、正式重み非変更は **{validation['formal_models_unchanged']}** だった。

## 比較条件

- データ: 957ケース、固定分割 685/136/136、追加COMSOLなし
- 未知評価: 75構造（間隔25、高さ・サイズ25、複合変化25）と対応アンカー25
- 候補間の共通条件: UNO、seed 1237、同じ正式SDF初期重み、同じ最適化器、15エポック固定追加学習
- 正式参照: 既存の凍結済み正式Dimension/SDF（歴史的50エポック学習）を再学習せず評価
- 主指標: プラズマ内の物理空間相対L2、Bohm 1-Dプロファイル相対L2
- 補助指標: 正則構造アンカーからのCOMSOL構造差分を再現する相対L2

## 全モデル結果

{chr(10).join(table_lines)}

\* 候補の学習時間は同じ15エポック追加学習で比較可能。正式2モデルの時間は過去の50エポック学習時間であり、候補との直接比較には使わない。

単純な75ケース中央値はケース順位を無視するため、Hのnᵢ中央値は正式SDFより低くても、同じケース同士の差では中央値 +{_fmt(best_pair['median_error_difference_pp_vs_formal_sdf'])}ポイント、勝率 {100.0 * float(best_pair['fraction_cases_better_than_formal_sdf']):.1f}% だった。このため図06の対応差を最終判断に用いる。

## 機能別評価

- **A 多解像度UNO**: nᵢ単純中央値は {_fmt(values['A_multires']['unknown_ni_rel_l2_median_pct'])}% まで下がったが、Bohmは {_fmt(values['A_multires']['unknown_bohm_rel_l2_median_pct'])}% に悪化。多解像度化だけでは両立しなかった。
- **AB 入力群別融合**: nᵢ単純中央値 {_fmt(values['AB_separate_fusion']['unknown_ni_rel_l2_median_pct'])}% だが、対応差では正式SDFより明確に悪い。process/geometry/EMの分離自体は十分条件でない。
- **ABC 局所・大域適応混合**: 候補中のBohm最良 {_fmt(values['ABC_adaptive_mix']['unknown_bohm_rel_l2_median_pct'])}% だが、正式SDF {_fmt(formal_sdf['unknown_bohm_rel_l2_median_pct'])}% には届かない。候補内では最も筋が良い演算改善。
- **ABD 共有コイル符号化**: nᵢ {_fmt(values['ABD_shared_coils']['unknown_ni_rel_l2_median_pct'])}%、構造差分 {_fmt(values['ABD_shared_coils']['structural_delta_ni_rel_l2_median_pct'])}% で正式SDFより悪化。コイル別SDFを持つだけでは利用されない。
- **ABE 磁場形状・振幅分離**: nᵢ {_fmt(values['ABE_em_shape_amplitude']['unknown_ni_rel_l2_median_pct'])}%、Bohm {_fmt(values['ABE_em_shape_amplitude']['unknown_bohm_rel_l2_median_pct'])}% で改善なし。単純な再パラメータ化は不十分。
- **ABF 構造差分監督**: 標準テストは {_fmt(values['ABF_structure_delta']['standard_test_primary_metric'], 5)} と良いが、未知構造差分 {_fmt(values['ABF_structure_delta']['structural_delta_ni_rel_l2_median_pct'])}% は正式SDF {_fmt(formal_sdf['structural_delta_ni_rel_l2_median_pct'])}% を超えない。補助課題の学習成功が外部構造汎化へ直結しなかった。
- **ABG SDF→磁場補助課題**: nᵢ単純中央値 {_fmt(values['ABG_sdf_em_aux']['unknown_ni_rel_l2_median_pct'])}% だが、Bohm {_fmt(values['ABG_sdf_em_aux']['unknown_bohm_rel_l2_median_pct'])}% に悪化。全磁場復元を同時に課す方式は主課題と競合した。
- **ABH 物理量別デコーダ**: 標準テスト {_fmt(values['ABH_target_decoders']['standard_test_primary_metric'], 5)} とnᵢ単純中央値 {_fmt(values['ABH_target_decoders']['unknown_ni_rel_l2_median_pct'])}% は候補中最良。ただしBohm {_fmt(values['ABH_target_decoders']['unknown_bohm_rel_l2_median_pct'])}% と対応勝率40%のため、nᵢ特化の改善に留まる。
- **ALL 全統合**: nᵢ {_fmt(values['ALL_combined']['unknown_ni_rel_l2_median_pct'])}%、Bohm {_fmt(values['ALL_combined']['unknown_bohm_rel_l2_median_pct'])}% で悪化。機能の足し算は有効でなく、目的の選別が必要。

## 読み方

1. 図00は代表3構造でコイル形状、COMSOL、正式Dimension、正式SDF、最良候補を同一色軸で直接比較する。
2. 図02は未知75構造のnᵢとBohmの中央値を比較する主ランキングである。
3. 図03は間隔、高さ・サイズ、複合変化のどこで効いたかを示す。
4. 図04は絶対場ではなく、構造を変えたことで生じたCOMSOL差分を再現できるかを直接測る。
5. 図05は固定予算内の精度と計算時間の関係を示す。
6. 図06は同一75ケースで正式SDFとの差と95%区間を示し、単純中央値の逆転が一貫した改善かを判定する。

## 判断境界

この結果は現在の生成設計範囲内にある凍結G4未知組合せへの汎化を示す。任意のコイル数・任意トポロジーへの無制限外挿は主張しない。学習seedは1つなので、ケース・ブートストラップはケース差を評価するが、再学習ばらつきは評価しない。正式版への昇格は追加seedでの再現確認後とする。

## ファイル

- `00_direct_field_comparison.png`: 代表場の直接比較
- `01_bohm_profiles.png`: Bohm 1-D比較
- `02_overall_accuracy_ranking.png`: 全体順位
- `03_family_accuracy_heatmap.png`: 構造ファミリー別
- `04_structural_response_error.png`: 構造差分再現
- `05_accuracy_cost_pareto.png`: 精度・学習時間
- `06_paired_effect_vs_formal_sdf.png`: 正式SDFとの同一ケース対応差
- `model_summary.csv`: 数値表
- `paired_effects_vs_formal_sdf.csv`: 同一ケース対応差と95%区間
- `validation_report.json`: データ・評価監査
- `formal_preservation_audit.json`: 正式重み非変更監査
"""
    (REPORT / "RESULTS_JA.md").write_text(markdown, encoding="utf-8")

    body_rows = "".join(
        "<tr>" + "".join(
            f"<td>{html.escape(value)}</td>" for value in (
                str(rank), LABELS[row["model"]], _fmt(row["unknown_ni_rel_l2_median_pct"]) + "%",
                _fmt(row["unknown_bohm_rel_l2_median_pct"]) + "%",
                _fmt(row["structural_delta_ni_rel_l2_median_pct"]) + "%",
                _fmt(row["training_elapsed_min"], 1) + " min",
            )
        ) + "</tr>"
        for rank, row in enumerate(ordered, start=1)
    )
    html_text = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ICP UNO v46 evaluation</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1180px;margin:auto;padding:28px;color:#172033;line-height:1.65}}h1,h2{{line-height:1.25}}.lead{{font-size:1.15rem;background:#eef7f3;border-left:6px solid #009e73;padding:18px}}.warn{{background:#fff6e5;border-left:6px solid #e69f00;padding:14px}}img{{width:100%;height:auto;border:1px solid #d8dde6;margin:12px 0 28px}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{border-bottom:1px solid #d8dde6;padding:8px;text-align:right}}th:nth-child(2),td:nth-child(2){{text-align:left}}code{{background:#f2f4f7;padding:2px 4px}}@media(max-width:700px){{body{{padding:14px}}table{{font-size:.75rem}}}}</style></head><body>
<h1>ICP UNO 構造・磁場表現 v46 評価報告</h1><p class="lead">{html.escape(claim)} {html.escape(boundary)}</p>
<p>Bohm最良候補: <strong>{html.escape(LABELS[best_bohm['model']])}</strong> {_fmt(best_bohm['unknown_bohm_rel_l2_median_pct'])}%（正式SDF {_fmt(formal_sdf['unknown_bohm_rel_l2_median_pct'])}%）。全候補が正式SDFより悪いため、総合判断は正式SDF維持。監査: <strong>{html.escape(validation['status'].upper())}</strong></p>
<p class="warn">正式学会モデルは保持した。これは評価用1-seed比較であり、自動昇格は行わない。</p>
<h2>全体順位（nᵢ基準）</h2><img src="02_overall_accuracy_ranking.png" alt="overall ranking"><table><thead><tr><th>nᵢ順位</th><th>モデル</th><th>未知 nᵢ</th><th>未知 Bohm</th><th>構造差分 nᵢ</th><th>学習時間*</th></tr></thead><tbody>{body_rows}</tbody></table><p>* 候補は同じ15エポック追加学習。正式2モデルは過去50エポックの時間で直接比較しない。</p>
<h2>代表3構造の直接比較</h2><img src="00_direct_field_comparison.png" alt="direct field comparison">
<h2>Bohmプロファイル</h2><img src="01_bohm_profiles.png" alt="Bohm profiles">
<h2>構造ファミリー別</h2><img src="03_family_accuracy_heatmap.png" alt="family heatmap">
<h2>構造変化の再現</h2><img src="04_structural_response_error.png" alt="structural response">
<h2>精度と学習時間</h2><img src="05_accuracy_cost_pareto.png" alt="accuracy cost">
<h2>正式SDFとの同一ケース対応差</h2><img src="06_paired_effect_vs_formal_sdf.png" alt="paired effect versus formal SDF">
<h2>機能別の結論</h2><ul><li>A/ABはnᵢ単純中央値を下げたが、同一ケース対応差とBohmで正式SDFを上回らない。</li><li>ABCは候補中のBohm最良だが正式SDFより悪い。</li><li>ABD/ABEのコイル別・磁場再表現は改善なし。</li><li>Fは標準テスト、Hはnᵢ単純中央値で強いが、未知構造のBohmと対応差へ移らない。</li><li>Gと全統合は補助課題の競合が大きく、追加複雑性に見合わない。</li></ul>
<h2>評価境界</h2><p>凍結G4の設計範囲内における未知組合せの結果であり、任意トポロジーへの無制限外挿は主張しない。学習seedは1つ。詳細数値と対応差は <code>model_summary.csv</code> と <code>paired_effects_vs_formal_sdf.csv</code> に保存した。</p>
</body></html>"""
    (REPORT / "report.html").write_text(html_text, encoding="utf-8")

    artifact = {
        "surface": "technical-report",
        "title": "ICP UNO structure and electromagnetic representation v46 evaluation",
        "status": validation["status"],
        "formal_model_status": decision["formal_model_status"],
        "best_unknown_ni_model": best_ni["model"],
        "best_unknown_bohm_model": best_bohm["model"],
        "sources": [
            "model_summary.csv", "paired_effects_vs_formal_sdf.csv", "validation_report.json",
            "formal_preservation_audit.json", "evaluation_contract.json",
        ],
        "figures": [f"{index:02d}_{name}" for index, name in enumerate((
            "direct_field_comparison", "bohm_profiles", "overall_accuracy_ranking",
            "family_accuracy_heatmap", "structural_response_error", "accuracy_cost_pareto",
            "paired_effect_vs_formal_sdf",
        ))],
    }
    (REPORT / "artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
