# ビリルビン検出システム プロトタイプ

<div align="center">
  <img src="docs/images/system_overview.png" alt="System Overview" width="600" style="max-width: 100%;">
  
  **AIを活用した非侵襲的ビリルビン検出システム**
  
  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green.svg)](https://opencv.org/)
  [![License](https://img.shields.io/badge/License-Research%20Use-orange.svg)](LICENSE)
</div>

## 📋 概要

本システムは、スマートフォンやWebカメラで撮影した眼球画像から、黄疸の指標となるビリルビン値を非侵襲的に推定するAIベースのプロトタイプです。最新の研究成果に基づき、RGB/HSVカラー空間解析と機械学習を組み合わせた手法を採用しています。

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
├── bilirubin_detector.py    # メイン検出スクリプト
├── test_detector.py         # テスト用スクリプト
├── requirements.txt         # Python依存パッケージ
├── README.md               # プロジェクト概要
├── utils/                  # ユーティリティモジュール
│   ├── __init__.py
│   ├── image_processing.py  # 画像処理（眼球検出）
│   ├── color_analysis.py    # 色空間解析
│   └── calibration.py       # 色較正機能
├── docs/                   # ドキュメント
│   ├── methodology.md      # 技術手法の詳細
│   ├── api_reference.md    # API リファレンス
│   ├── development.md      # 開発ガイド
│   └── research/          # 参考研究論文
└── sample_images/         # テスト用画像
```

## 📊 出力結果

### 結果の解釈

システムは以下の情報を提供します：

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

### リスクレベルの基準

| レベル | ビリルビン値 (mg/dL) | 状態 | 推奨アクション |
|--------|---------------------|------|----------------|
| 🟢 **Low** | < 3 | 正常範囲 | 経過観察 |
| 🟡 **Moderate** | 3-12 | 軽度の黄疸 | 医療機関での確認を推奨 |
| 🟠 **High** | 12-20 | 中等度の黄疸 | 速やかに医療機関を受診 |
| 🔴 **Critical** | > 20 | 重度の黄疸 | 緊急に医療機関を受診 |

### 色特徴の意味

- **HSV Yellow Ratio**: 黄色成分の割合（0-1）
- **RGB R/B Ratio**: 赤/青比率（黄疸で上昇）
- **Yellowness Index**: 総合的な黄色度指標

## 🔧 技術的詳細

詳細な技術情報は[技術ドキュメント](docs/methodology.md)を参照してください。

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
