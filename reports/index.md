# Plasma Surrogate レポート一覧

[プロジェクト入口](../README.md)から参照する、生成済みレポートと発表素材の目次です。

| 対象 | 内容 | 入口 |
|---|---|---|
| GEC-CCP最終採用図 | RMSE散布図、電子密度空間分布、評価条件、学習loss数式 | [採用結果README](gec_conference_materials/README.md) |
| GEC-CCP学会画像 | 空間分布、真値・予測・誤差、モデル比較、最適化結果、品質図 | [GEC-CCP専用目次](gec_conference_materials/gec_ccp_index.md) |
| GEC-CCP 3モデル詳細 | FFNO・DeepONet・Global MLPの共通testケース空間図と対数学習曲線 | [モデル詳細図](gec_conference_materials/model_detail_assets/index.md) |
| GEC-CCP学習条件数 | 全11モデル、本番val/test固定でtrain 54・48・36・24件を比較 | [全モデル学習条件数アブレーション](../runs/gec_ccp_training_size_ablation_v1/summary_all_models/index.md) |
| GEC-CCP最適化 | 3計測×5 seedsの収束・達成率、費用、TPE profile、最適化後電子密度場 | [最適化グラフ集](gec_conference_materials/optimization_assets/index.md) |
| GEC-CCP / GEC-ICP学会素材 | 両装置の簡略図と独立パーツ | [学会発表用図版](gec_conference_materials/index.md) |
| ICP学会画像 | FFNOコイル構造、最適化履歴、密度場、QoI比較 | [ICP学会発表用グラフ](icp_conference_materials/index.md) |
| GEC-CCPジオメトリー | 計算領域、境界、メッシュ | [ジオメトリーレポート](gec_ccp_geometry/index.md) |
| GEC-CCPモデル候補 | モデル候補の整理 | [モデル候補レポート](gec_ccp_model_candidates/index.md) |
| GEC-ICPジオメトリー | 計算領域、コイル、構造格子 | [ジオメトリーレポート](gec_icp_geometry/index.md) |

学会スライドへ直接配置する場合はPNG、投稿・配布にはPDF、編集にはSVGを使用します。
数値根拠と再生成条件は各素材のmetadata JSONおよび各目次に記載しています。
