# ICP_stage4 物理関係を考慮した目的変数グルーピングメモ

作成日: 2026-05-29

## 背景

ICP_stage4 では、コイル構造・配置を raw 寸法パラメータとして surrogate に直接入れるのではなく、coil mask、SDF、boundary、part summary などの構造特徴量として学習する方針を取る。目的変数は空間2次元場であり、現状の主要候補は `ne`, `ni`, `Te`, `phi`, `Br`, `Bz`, `Jelr`, `Jelz` である。

これらを単純に all-8 joint model として同じ出力 head / 同じ loss / 同じ有効領域で学習すると、物理的に意味を持つ領域やスケールが異なる場が混在し、モデル容量と勾配が競合しやすい。

## 物理的な関係

低圧 ICP の大まかな因果関係は次のように整理できる。

```text
coil geometry / coil current
  -> Br, Bz
  -> induced E, power deposition, electron current
  -> Te
  -> ionization / transport
  -> ne, ni
  -> phi / ambipolar field / sheath boundary
```

この関係を考えると、すべての場を同列に扱うより、物理グループごとに head、loss、mask、評価指標を分けるほうが学習効率と解釈性の両方で有利である。

## 推奨する目的変数グループ

### Plasma state model

```text
ne, ni, Te, phi
```

この4場は plasma 領域で意味を持ち、密度、電子温度、電位が強く結合している。`ne` と `ni` は準中性に近く、`Te` は電離・輸送に効き、`phi` は ambipolar field や境界近傍構造と関係する。

今回の 4-field FFNO では、plasma-only 評価で all-8 joint より良い精度と分布連続性が得られた。

### Magnetic field model

```text
Br, Bz
```

磁場は plasma 領域だけでなく、真空領域やコイル近傍にも広がるため、`all_domain` で扱うのが自然である。`ne/Te/phi` と同じ plasma-only mask で学習・評価すると重要な空間構造を捨てる。

`Br/Bz` は符号付きベクトル場としてペアで扱う。将来的には弱い `div B ~= 0` consistency や gradient consistency が有効な可能性がある。

### Current density model

```text
Jelr, Jelz
```

電子電流密度は符号付きベクトル場であり、局所ピークや符号反転を持つ。データ上は plasma-only として扱われているが、誘導電場・磁場とも強く結合するため、単独ペアと EM-current joint の両方を比較する価値がある。

候補:

```text
Jelr, Jelz
Br, Bz, Jelr, Jelz
```

## 学習効率化の方向性

最終的に有望なのは、完全に別々のモデルだけではなく、次のような構成である。

```text
shared structure encoder
  + magnetic head: Br, Bz
  + current head: Jelr, Jelz
  + plasma head: ne, ni, Te, phi
```

構造 encoder は共有しつつ、出力 head、loss、mask、metric は物理グループごとに分ける。これにより、コイル構造から得られる共通情報を共有しながら、有効領域や空間スケールの違いを吸収できる。

製品基盤としては ICP 固有の field 名を本体に直書きしない。汎用機能としては、target group ごとの head / mask / loss / metric を扱える形が望ましい。ICP 固有の grouping や QoI は外部 config / script / docs に置く。

## 弱い物理 consistency の候補

強い物理拘束を雑に入れると外挿やノイズで悪化しやすいため、まずは弱い補助 loss または評価指標として扱う。

- `ne - ni` consistency: plasma bulk での準中性を弱く見る。
- `Br/Bz` gradient または `div B` consistency: 磁場の滑らかさと空間整合を見る。
- `Jelr/Jelz` gradient / shape consistency: 電流ピークと符号反転の位置ずれを見る。
- `Te` と `ne/ni` の結合: 手書き拘束より multi-task head 内で学習させるほうが安全。
- `phi` と密度勾配: plasma state model に含め、境界近傍評価を重視する。

## 評価すべきモデル構成

優先順:

```text
1. plasma model: ne, ni, Te, phi
2. magnetic model: Br, Bz
3. current model: Jelr, Jelz
4. EM-current model: Br, Bz, Jelr, Jelz
5. all-8 joint model: ne, ni, Te, phi, Br, Bz, Jelr, Jelz
```

all-8 joint model は baseline として重要だが、主力候補としては target group model と比較する。最終的な設計最適化では、各 group model の出力を統合して QoI を評価する。

## 判断基準

R2 だけでなく、場としての品質を評価する。

- 有効領域別 R2 / RMSE
- shape correlation
- peak location error
- center-of-mass error
- radial / axial profile error
- gradient RMSE
- density negative rate
- plasma-only と all-domain の扱いが物理的に整合していること

## 当面の結論

ICP_stage4 では、目的変数は個数で区切るのではなく、物理領域と結合関係で分ける。

最も自然な主構成:

```text
Plasma state: ne, ni, Te, phi
Magnetic field: Br, Bz
Electron current: Jelr, Jelz
```

将来的な本命構成:

```text
shared structure encoder + physics-group heads
```

ただし、これは本体に ICP 固有処理を埋め込むという意味ではない。本体は target group を扱える汎用基盤に留め、ICP 固有の grouping、単位、QoI、図、最適化目的は外部処理として管理する。
