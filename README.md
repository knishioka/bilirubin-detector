# ビリルビン検出システム & ダークサークル検出 プロトタイプ

<div align="center">
  <img src="docs/images/system_overview.png" alt="System Overview" width="600" style="max-width: 100%;">
  
  **AIを活用した非侵襲的眼球画像解析システム**
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
  [![License](https://img.shields.io/badge/License-Research%20Use-orange.svg)](LICENSE)
</div>

## 📋 概要

本システムは、スマートフォンやWebカメラで撮影した眼球画像から、以下の健康指標を非侵襲的に推定するAIベースのプロトタイプです：

1. **ビリルビン検出**: 黄疸の指標となるビリルビン値を結膜画像から推定
2. **ダークサークル検出**: 目の下のクマ（眼窩周囲色素沈着）の検出と重症度評価

最新の研究成果に基づき、RGB/HSV/CIELABカラー空間解析と機械学習を組み合わせた手法を採用しています。

### 🎯 主な特徴

- **非侵襲的測定**: 採血不要で痛みのない検査
- **迅速な結果**: 数秒で推定値を算出
- **高精度**: 研究論文で実証された85-96%の精度を目指す設計
- **色較正対応**: 様々な照明条件での安定した測定
- **拡張可能**: 将来的な深層学習モデルの統合を考慮した設計

## 🚀 クイックスタート

```bash
# 1. リポジトリのクローン
git clone https://github.com/knishioka/bilirubin-detector.git
cd bilirubin-detector

# 2. 依存関係のインストール（uvを推奨）
uv pip install -r requirements.txt

# 3. ビリルビン検出
python bilirubin_detector.py sample_images/eye_mild_jaundice.jpg

# 4. ダークサークル検出
python dark_circle_detector.py path/to/face_image.jpg
```

詳細は[インストールガイド](docs/installation.md)を参照してください。

## 📚 ドキュメント

### セットアップ・使い方
- [📥 インストールガイド](docs/installation.md) - 環境構築の詳細手順
- [💡 使用例とチュートリアル](docs/examples.md) - 実践的なコード例
- [🔧 トラブルシューティング](docs/troubleshooting.md) - 問題解決ガイド

### 技術ドキュメント
- [🔬 ビリルビン検出の技術手法](docs/methodology.md) - アルゴリズムの科学的背景
- [👁️ ダークサークル検出の技術手法](docs/dark_circle_methodology.md) - CIELAB解析の詳細
- [📐 システムアーキテクチャ](docs/architecture.md) - 設計思想と構造
- [📊 アルゴリズムフロー図](docs/algorithm_flowchart.md) - 処理の流れ

### 開発者向け
- [🔍 APIリファレンス](docs/api_reference.md) - クラス・メソッドの詳細
- [🧪 テストガイド](docs/testing.md) - テスト方法と結果の解釈
- [💻 開発ガイド](docs/development.md) - コントリビューション方法

## 🚀 使用方法

### 基本的な使用

```bash
# 画像からビリルビン値を推定
python bilirubin_detector.py path/to/eye_image.jpg

# 結果を可視化して保存
python bilirubin_detector.py path/to/eye_image.jpg --output result.png

# JSON形式で結果を出力（他システムとの連携用）
python bilirubin_detector.py path/to/eye_image.jpg --json

# ダークサークル（目の下のクマ）検出
python dark_circle_detector.py path/to/face_image.jpg

# ダークサークル検出結果を可視化
python dark_circle_detector.py path/to/face_image.jpg --output result.png
```

### 高度な使用方法

#### 色較正を使用した高精度測定

```bash
# 色較正カードの生成（印刷して使用）
python test_detector.py --create-calibration-card

# 色較正モードで実行
python bilirubin_detector.py path/to/eye_image.jpg --calibrate
```

#### バッチ処理

```bash
# 複数画像の一括処理
for img in images/*.jpg; do
    python bilirubin_detector.py "$img" --json >> results.jsonl
done
```

### 📸 撮影のコツ

最適な結果を得るために：

1. **照明**: 自然光または白色LEDを使用
2. **距離**: 眼球が画面の1/3程度を占める距離
3. **角度**: 正面からやや上向きで撮影
4. **表情**: 目を大きく開き、上を見る
5. **背景**: 無地の明るい背景を推奨

## 📁 プロジェクト構造

```
health-tech/
├── bilirubin_detector.py      # ビリルビン検出メインスクリプト
├── dark_circle_detector.py    # ダークサークル検出メインスクリプト
├── requirements.txt           # Python依存パッケージ
├── README.md                  # プロジェクト概要
│
├── utils/                     # コアモジュール
│   ├── image_processing.py    # 画像処理・眼球検出
│   ├── color_analysis.py      # 色空間解析
│   ├── calibration.py         # 色較正機能
│   ├── periorbital_detection.py      # 眼窩周囲領域検出
│   ├── dark_circle_analysis.py       # ダークサークル色解析
│   └── dark_circle_segmentation.py   # 領域分割
│
├── scripts/                   # ユーティリティスクリプト
│   ├── data_generation/       # テストデータ生成
│   │   ├── create_jaundice_samples.py
│   │   └── generate_dark_circle_samples.py
│   ├── evaluation/            # 評価・テスト
│   │   ├── test_detector.py
│   │   └── test_dark_circle_detection.py
│   └── visualization/         # 可視化
│       └── create_algorithm_diagrams.py
│
├── docs/                      # ドキュメント
│   ├── installation.md        # インストールガイド
│   ├── troubleshooting.md     # トラブルシューティング
│   ├── methodology.md         # ビリルビン検出技術
│   ├── dark_circle_methodology.md  # ダークサークル検出技術
│   ├── architecture.md        # システムアーキテクチャ
│   ├── api_reference.md       # APIリファレンス
│   ├── testing.md             # テストガイド
│   ├── examples.md            # 使用例
│   └── development.md         # 開発ガイド
│
└── sample_images/             # サンプル画像
    └── samples/
        └── dark_circles/      # ダークサークルサンプル
```

## 📊 出力結果

### 結果の解釈

#### ビリルビン検出結果

```json
{
  "success": true,
  "bilirubin_level_mg_dl": 5.8,
  "risk_level": "moderate",
  "confidence": 0.82,
  "color_features": {
    "hsv_yellow_ratio": 0.23,
    "rgb_red_blue_ratio": 1.45,
    "yellowness_index": 0.156
  },
  "calibrated": false
}
```

#### ダークサークル検出結果

```json
{
  "success": true,
  "average_delta_e": 6.2,
  "severity": "moderate",
  "symmetry_score": 0.85,
  "left_eye": {
    "delta_e": 6.5,
    "darkness_ratio": 0.12,
    "ita_infraorbital": 32.5,
    "redness_index": 0.15,
    "blueness_index": 0.08
  },
  "right_eye": {
    "delta_e": 5.9,
    "darkness_ratio": 0.10,
    "ita_infraorbital": 33.2,
    "redness_index": 0.13,
    "blueness_index": 0.07
  }
}
```

### リスクレベルの基準

| レベル | ビリルビン値 (mg/dL) | 状態 | 推奨アクション |
|--------|---------------------|------|----------------|
| 🟢 **Low** | < 3 | 正常範囲 | 経過観察 |
| 🟡 **Moderate** | 3-12 | 軽度の黄疸 | 医療機関での確認を推奨 |
| 🟠 **High** | 12-20 | 中等度の黄疸 | 速やかに医療機関を受診 |
| 🔴 **Critical** | > 20 | 重度の黄疸 | 緊急に医療機関を受診 |

### ダークサークル重症度の基準

| レベル | ΔE値 | 状態 | 視覚的特徴 |
|--------|------|------|-----------|
| 🟢 **None** | < 3 | なし | ほとんど目立たない |
| 🟡 **Mild** | 3-5 | 軽度 | わずかに目立つ |
| 🟠 **Moderate** | 5-8 | 中等度 | 明らかに目立つ |
| 🔴 **Severe** | > 8 | 重度 | 非常に目立つ |

### 色特徴の意味

#### ビリルビン検出
- **HSV Yellow Ratio**: 黄色成分の割合（0-1）
- **RGB R/B Ratio**: 赤/青比率（黄疸で上昇）
- **Yellowness Index**: 総合的な黄色度指標

#### ダークサークル検出
- **Delta E (ΔE)**: 頬と眼窩下領域の色差（CIE2000）
- **Darkness Ratio**: 明度差の比率
- **ITA**: Individual Typology Angle（肌色分類）
- **Redness Index**: 血管成分の指標
- **Blueness Index**: 静脈うっ血の指標

## 🔬 技術概要

### ビリルビン検出
- **原理**: 結膜の黄色度からビリルビン値を推定
- **色空間**: RGB/HSV/LAB多重解析
- **精度目標**: 研究論文の85-96%を目指す

### ダークサークル検出
- **原理**: CIELAB色空間でのΔE（色差）解析
- **評価基準**: CIE2000色差式による客観的評価
- **対応タイプ**: 色素沈着型、血管型、構造型、混合型

詳細は技術ドキュメントを参照してください。

## ⚠️ 注意事項

### 医療上の免責事項

- 本システムは**研究・教育目的**のプロトタイプです
- **医療診断には使用できません**
- 実際の黄疸診断には必ず医療機関での血液検査を受けてください
- 推定値は参考値であり、診断の代替にはなりません

### 技術的制限

- 実際の臨床データでの学習は未実施
- 照明条件に依存（標準化された環境を推奨）
- 人種・年齢・性別による個人差は未考慮

## 🚧 今後の開発計画

### Phase 1: 基礎機能強化（現在）
- [x] 基本的な色解析機能
- [x] 眼球検出アルゴリズム
- [x] ダークサークル検出機能
- [x] CIELAB色空間解析とΔE計算
- [ ] Webカメラリアルタイム入力
- [ ] 複数画像の統計処理

### Phase 2: 機械学習統合
- [ ] 転移学習モデル（ResNet/EfficientNet）の実装
- [ ] 臨床データセットでの学習
- [ ] 信頼区間の推定

### Phase 3: 実用化
- [ ] モバイルアプリ開発（iOS/Android）
- [ ] 医療機器認証取得の検討
- [ ] 多施設臨床試験

## 🤝 貢献方法

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/AmazingFeature`)
3. 変更をコミット (`git commit -m 'Add some AmazingFeature'`)
4. ブランチにプッシュ (`git push origin feature/AmazingFeature`)
5. プルリクエストを作成

詳細は[開発ガイド](docs/development.md)を参照してください。

## 📚 参考文献

1. [Nature Scientific Reports: AI-based jaundice detection (2025)](https://www.nature.com/articles/s41598-025-96100-9)
2. [PMC: Smartphone-based neonatal jaundice assessment (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8588081/)
3. [詳細な参考文献リスト](docs/research/references.md)
