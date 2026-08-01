# Plasma Surrogate project portal

学会発表に採用したGEC-CCPのRMSE散布図、電子密度空間分布、評価条件、loss数式は、
[GEC-CCP学会発表用・採用結果README](reports/gec_conference_materials/README.md)から直接参照できます。
補助的な学習曲線とモデル別詳細図は[GEC-CCPモデル詳細図](reports/gec_conference_materials/model_detail_assets/index.md)にあります。

全11モデルのtrain 54→48／36／24件アブレーションは、[GEC-CCP全モデル学習条件数評価](runs/gec_ccp_training_size_ablation_v1/summary_all_models/index.md)にあります。元の3モデルだけの要約は[こちら](runs/gec_ccp_training_size_ablation_v1/summary/index.md)です。

実装・ベンチマーク・発表資料へプロジェクト直下から移動するための入口です。
製品コードの概要と利用方法は[メインREADME](README_mainline.md)、GEC-CCPの再学習手順は
[Benchmark runbook](Benchmarkrun.md)を参照してください。

## GEC-CCP学会用画像

[GEC-CCP学会用画像の専用目次](reports/gec_conference_materials/gec_ccp_index.md)に、
次の素材をまとめています。

- ジオメトリーとCOMSOLメッシュ
- 電子密度、イオン密度、電子温度、電位の2D空間分布と平面斜視図
- モデル別の真値・予測値・空間誤差分布
- モデル精度、物性別R²、データセット品質、前処理の図
- センサー同化・入力最適化の収束、達成率、費用、TPE profile、最適化後電子密度場
- PNG、PDF、SVG、metadataと再生成スクリプト

全レポートの入口は[reports/index.md](reports/index.md)、GEC-CCP / GEC-ICPを含む
学会素材全体は[学会発表用図版](reports/gec_conference_materials/index.md)です。

## ICP学会用画像

[ICP学会用画像の専用目次](reports/icp_conference_materials/index.md)に、FFNOサロゲートを
用いたコイル構造最適化の結果をまとめています。コイル配置の最適化前後、240 trialの
最適化履歴、電子・イオン密度場、主要QoIをPNG・PDF・SVGで参照できます。

この結果は学習済みサロゲート内の探索結果です。最適構造の物理的な採用判断には、別途
高忠実度シミュレーションによる再検証が必要です。

## 開発ドキュメント

- [アーキテクチャ](docs/01_architecture.md)
- [データ契約](docs/02_data_contract.md)
- [前処理と特徴量](docs/03_preprocess_and_features.md)
- [学習モデル](docs/04_training_models.md)
- [推論と評価](docs/05_inference_and_evaluation.md)
- [拡張ガイド](docs/07_extension_guide.md)
