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

## 🔬 技術的背景

### 科学的根拠

黄疸は血中ビリルビン濃度の上昇により皮膚や眼球結膜が黄色く変色する症状です。本システムは以下の科学的知見に基づいています：

1. **ビリルビンの光学的特性**: ビリルビンは450-460nmの青色光を吸収し、黄色を呈する
2. **結膜での視認性**: 眼球結膜は血管が豊富で、ビリルビンの蓄積が視覚的に確認しやすい
3. **色空間での定量化**: HSV色空間における黄色成分（Hue: 20-40°）の比率がビリルビン濃度と相関

### 採用技術

- **RGB/HSVカラー空間解析**: 研究で実証された高精度な色特徴抽出
- **自動眼球検出**: OpenCVのカスケード分類器による結膜領域の自動検出
- **色較正機能**: 照明条件の違いを補正する色較正カードサポート
- **機械学習対応設計**: 将来的な深層学習モデルの統合を考慮したアーキテクチャ

## 📥 セットアップ

### 必要環境

- Python 3.9以上
- OpenCV対応のシステム（Windows/macOS/Linux）
- Webカメラまたは画像ファイル

### クイックスタート

```bash
# リポジトリのクローン
git clone https://github.com/your-repo/bilirubin-detector.git
cd bilirubin-detector

# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # macOS/Linux
# または
venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -r requirements.txt

# 動作確認
python test_detector.py --test-all
```

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
├── bilirubin_detector.py    # ビリルビン検出スクリプト
├── dark_circle_detector.py  # ダークサークル検出スクリプト
├── test_detector.py         # ビリルビン検出テスト
├── test_dark_circle_detection.py # ダークサークル検出テスト
├── generate_dark_circle_samples.py # サンプル画像生成
├── requirements.txt         # Python依存パッケージ
├── README.md               # プロジェクト概要
├── utils/                  # ユーティリティモジュール
│   ├── __init__.py
│   ├── image_processing.py  # 画像処理（眼球検出）
│   ├── color_analysis.py    # 色空間解析
│   ├── calibration.py       # 色較正機能
│   ├── periorbital_detection.py # 眼窩周囲領域検出
│   ├── dark_circle_analysis.py  # ダークサークル色解析
│   └── dark_circle_segmentation.py # ダークサークル領域分割
├── docs/                   # ドキュメント
│   ├── methodology.md      # 技術手法の詳細
│   ├── api_reference.md    # API リファレンス
│   ├── examples.md         # 使用例とチュートリアル
│   ├── development.md      # 開発ガイド
│   ├── algorithm_flowchart.md # アルゴリズムフロー図
│   ├── images/            # ドキュメント用画像
│   └── research/          # 参考研究論文
└── sample_images/         # テスト用画像
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

## 🔧 技術的詳細

### 📚 技術ドキュメント

- [技術手法の詳細](docs/methodology.md) - アルゴリズムの科学的背景と実装詳細
- [APIリファレンス](docs/api_reference.md) - クラス・メソッドの詳細仕様
- [使用例とチュートリアル](docs/examples.md) - 実践的なコード例
- [アルゴリズムフロー図](docs/algorithm_flowchart.md) - 処理の流れを視覚的に説明
- [開発ガイド](docs/development.md) - 開発環境構築とコントリビューション方法

### アルゴリズムの概要

1. **前処理**: CLAHE（適応的ヒストグラム均等化）によるコントラスト強調
2. **領域検出**: Haar Cascadeによる顔・眼検出、HSVマスキングによる結膜抽出
3. **特徴抽出**: 複数色空間（RGB, HSV, LAB）での統計量算出
4. **推定**: 線形回帰モデルによるビリルビン値推定（将来的にDNNへ移行予定）

### パフォーマンス

- 処理時間: 約1-3秒/画像（CPU）
- メモリ使用量: < 500MB
- 対応画像サイズ: 最大4K（自動リサイズ）

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
