# ICP学会発表用グラフ

ICPコイル構造サロゲートと構造最適化に関する発表用図をまとめる。

| テーマ | 内容 | 入口 |
|---|---|---|
| FFNOコイル構造最適化 | コイル配置、最適化履歴、密度場、QoI比較 | [FFNO構造最適化グラフ](ffno_structure_optimization/index.md) |

スライドにはPNG、投稿・印刷にはPDF、編集にはSVGを使用する。各図は同一スクリプトから3形式を生成している。

## 再生成

```powershell
.venv-torch\Scripts\python.exe experiments/conference/scripts/plot_icp_ffno_structure_optimization.py
```

生成元：[plot_icp_ffno_structure_optimization.py](../../experiments/conference/scripts/plot_icp_ffno_structure_optimization.py)
