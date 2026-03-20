# 02 Data Contract

## 1. 基本方針

データ契約の正本は `src/plasma_surrogate/core/dataset_io.py` です。  
mainline では、target 名、列名、値変換を暗黙に推定しません。すべて YAML で明示します。

この章でいう「データ契約」は、学習前処理より前の段階を指します。  
つまり「生データをどの logical target として読むか」の仕様です。

## 2. mainline の前提 dataset.type

現在の first-class mainline は `dataset.type=csv_npz` です。

- CSV: ケースごとのメタ情報
- NPZ: 各ケースの field 配列
- geometry: fixed grid を共有する空間情報

`synthetic` もコード上にはありますが、第三者運用の基盤として読むなら `csv_npz` を前提に理解するのが適切です。

## 3. `csv_npz` の想定構造

### index CSV

index CSV はケース一覧とメタ情報を持ちます。  
少なくとも以下の種類の情報を持つ想定です。

- 条件変数
- field NPZ の場所
- case ID
- split 用グループ
- 軸識別子

### per-case NPZ

各ケースの分布データを持つファイルです。  
field 名は dataset ごとに異なってよく、`dataset.targets[].source_key` で吸収します。

### geometry

全ケースで共有される fixed grid の geometry 情報です。  
mainline は fixed grid 前提のため、ケース間で field shape が一致している必要があります。

## 4. `dataset` の主要キー

```yaml
dataset:
  type: csv_npz
  root: data/my_dataset
  index_csv: index.csv
  cond_columns: [p0, td, gamma]
  axis_column: axis
  fields_npz_column: fields_npz
  case_id_column: case_id
  base_case_id_column: base_case_id
  split_group_column: split_group
  geometry_root: geometry
  targets:
    - id: density
      source_key: density_raw
      value_transform: identity
      units: "m^-3"
      dtype: float32
```

### 必須

- `root`
- `cond_columns`
- `targets`

### よく使う補助キー

- `index_csv`
- `axis_column`
- `fields_npz_column`
- `case_id_column`
- `base_case_id_column`
- `split_group_column`
- `geometry_root`

## 5. `dataset.targets[]` の意味

`dataset.targets[]` が mainline の target 定義の唯一の入口です。

### `id`

- 学習、推論、評価、compare で使う logical target 名
- `output_layout.vars` の候補になる
- compare 列名や metrics 列名の suffix にも使われる

#### どこで使われるか

- `dataset_io`
- preprocess schema
- `output_layout.vars`
- `model_dispatch`
- metrics / benchmark / compare

#### よくあるミス

- source 側の列名と `id` を混同する
- 既存スクリプトの変数名に引きずられて dataset 非依存でない名前を選ぶ

### `source_key`

- 元データでの列名または field 名
- logical target 名と元データ名を分離するためのキー
- 省略時は `id` を使う

#### よくあるミス

- CSV 側列名と NPZ 側 field 名のどちらを指すか曖昧にする
- 元データの実名が変わったのに YAML を更新しない

### `value_transform`

- dataset 読み込み時に source を logical target に変換する
- 現在の `dataset_io` では `identity | pow10 | exp10`

#### 役割

これは「生データの定義差」を吸収するための変換です。  
学習前の統計変換ではありません。

#### 例

- dataset 側が `log10(density)` を持っている
- 学習 target は linear density として扱いたい
- このとき `dataset.targets[].value_transform=pow10` を使う

### `units`

- 単位メタデータ
- 現状は主に説明と監査用途

### `dtype`

- 読み込み dtype
- 数値の安定性とメモリの両面で揃えておく

## 6. `dataset.targets[].value_transform` と `target_transforms` の違い

ここは特に誤解されやすい箇所です。

### dataset 側の `value_transform`

- 目的: source データの意味を logical target に写像する
- 例: `log_ne` を `ne` に戻す
- 層: `dataset_io`

### preprocess 側の `target_transforms.<var>.value_transform`

- 目的: logical target を学習しやすい統計表現へ変換する
- 例: linear density を `log10` にして学習する
- 層: `preprocessing/scalers.py`

この 2 つを分けることで、dataset 依存の表現差と学習都合の変換を混同しません。

## 7. geometry 契約

geometry は fixed grid 前提です。  
全ケースで同じ shape を持たない場合、mainline の grid 系モデルは成立しません。

### `geometry_root`

`geometry_root` は次のいずれかを指します。

1. geometry provider root そのもの
2. `geometry/` を内包する上位ディレクトリ

下流で典型的に使う geometry チャネル:

- `mask_plasma`
- `distance_signed`
- `distance_any`
- `x`
- `y`

## 8. fail-fast 条件

別データセット導入時に止まるべき条件は、意図的に厳しめです。

- `dataset.targets[]` が空
- `source_key` が存在しない
- field shape がケース間で一致しない
- geometry が解決できない
- 条件列が不足している

この設計により、「とりあえず動いたが意味が違った」という事故を減らします。

## 9. 新しいデータセットを載せる最小手順

1. `dataset.targets[]` を定義する
2. 各 target の `source_key` を元データへ対応付ける
3. source が log や指数表現なら dataset 側 `value_transform` で logical target に直す
4. `cond_columns` を決める
5. `geometry_root` を固定する
6. preprocess 側で `target_transforms` を決める
7. `train.<model>.target_vars` を `dataset.targets[].id` に合わせる
8. 物理項を使うなら `physics.symbols` / `inference.ood.physics.symbols` を定義する

## 10. target 名の設計指針

- 物理的意味が分かる名前を使う
- source 側の列名をそのまま使う必要はない
- ただし、比較・可視化で読みにくくならないよう、短く一貫した名前にする
- target 名を変えたら、physics symbol mapping と `--vars` 指定も一緒に見直す
