# 技術手法の詳細

## 目次

1. [概要](#概要)
2. [科学的背景](#科学的背景)
3. [アルゴリズムの詳細](#アルゴリズムの詳細)
4. [色空間解析](#色空間解析)
5. [機械学習アプローチ](#機械学習アプローチ)
6. [精度評価](#精度評価)
7. [技術的課題と解決策](#技術的課題と解決策)

## 概要

本システムは、眼球結膜の画像解析により血中ビリルビン濃度を非侵襲的に推定する技術です。最新の研究成果に基づき、複数の色空間での特徴抽出と機械学習を組み合わせたアプローチを採用しています。

### 基本原理

1. **光学的原理**: ビリルビンは特定波長（450-460nm）の光を吸収し、黄色を呈する
2. **生理学的原理**: 血中ビリルビンは結膜の毛細血管に蓄積し、視覚的に観察可能
3. **計算機視覚**: デジタル画像から色情報を定量的に抽出・解析

## 科学的背景

### ビリルビンの生化学

ビリルビンは赤血球のヘモグロビンが分解される際に生成される黄色の色素です：

```
ヘモグロビン → ヘム → ビリベルジン → ビリルビン
```

- **正常値**: 0.3-1.2 mg/dL
- **軽度上昇**: 3-12 mg/dL（黄疸が視認可能）
- **中等度**: 12-20 mg/dL
- **重度**: >20 mg/dL

### 光学的特性

ビリルビンの吸収スペクトル：
- **最大吸収波長**: 453nm（青色光）
- **反射光**: 570-590nm（黄色光）

この特性により、RGB/HSV色空間での黄色成分の定量化が可能となります。

## アルゴリズムの詳細

### 1. 画像前処理

```python
def preprocess_image(image):
    # 1. リサイズ（計算効率のため）
    if max(image.shape) > 1024:
        image = resize_to_max_dimension(image, 1024)
    
    # 2. コントラスト強調（CLAHE）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    
    # 3. ノイズ除去
    denoised = cv2.fastNlMeansDenoisingColored(image)
    
    return denoised
```

### 2. 眼球・結膜検出

#### Haar Cascade による顔・眼検出

```python
def detect_eyes(image):
    # 顔検出
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 各顔から眼を検出
    for (x, y, w, h) in faces:
        roi = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)
```

#### 結膜領域の抽出

HSV色空間でのマスキング手法：

```python
def extract_conjunctiva(eye_image):
    hsv = cv2.cvtColor(eye_image, cv2.COLOR_BGR2HSV)
    
    # 白色～薄黄色の範囲を定義
    lower_bound = np.array([0, 0, 100])    # 低彩度、高明度
    upper_bound = np.array([40, 30, 255])  # 黄色領域まで含む
    
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    conjunctiva = cv2.bitwise_and(eye_image, eye_image, mask=mask)
```

### 3. 特徴抽出アルゴリズム

#### RGB色空間での特徴

```python
# R/B比率（黄疸の重要指標）
rb_ratio = mean_red / (mean_blue + epsilon)

# 正規化RGB
r_norm = R / (R + G + B)
g_norm = G / (R + G + B)
b_norm = B / (R + G + B)
```

#### HSV色空間での特徴

```python
# 黄色ピクセルの検出
yellow_mask = (
    (hue >= 20) & (hue <= 40) &     # 黄色の色相範囲
    (saturation > 30) &              # 最小彩度
    (value > 50)                     # 最小明度
)
yellow_ratio = np.sum(yellow_mask) / total_pixels
```

#### LAB色空間での特徴

```python
# b*チャンネル（黄-青軸）
yellowness = mean_b - 128  # 128を中心とした黄色度
```

## 色空間解析

### RGB色空間

- **利点**: 直接的な色情報、計算が簡単
- **欠点**: 照明の影響を受けやすい
- **主要特徴**: R/B比、G成分の相対値

### HSV色空間

- **利点**: 色相と明度が分離、黄色検出に最適
- **欠点**: ノイズに敏感
- **主要特徴**: Hue(20-40°)の分布、Saturation平均

### LAB色空間

- **利点**: 知覚的に均等、照明不変性が高い
- **欠点**: 計算コストが高い
- **主要特徴**: b*チャンネルの黄色度

### 特徴統合

```python
# 複合黄色度指標
yellowness_index = w1 * hsv_yellow_ratio + 
                   w2 * rgb_rb_ratio + 
                   w3 * lab_yellowness
```

## 機械学習アプローチ

### 現在の実装（線形回帰）

```python
# 簡易線形モデル
bilirubin = β0 + β1*hsv_yellow + β2*rgb_ratio + β3*saturation + ε

# 係数（仮値）
β0 = 2.1  # 切片
β1 = 15.2 # HSV黄色比率の重み
β2 = -8.5 # RGB比率の重み
β3 = 12.3 # 彩度の重み
```

### 将来の実装計画

#### 1. 転移学習アプローチ

```python
# ResNet/EfficientNetベースの特徴抽出
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# カスタムヘッド
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)  # ビリルビン値
```

#### 2. アンサンブル学習

- Random Forest
- Gradient Boosting
- Neural Network
の組み合わせによる高精度化

## 精度評価

### 評価指標

1. **RMSE (Root Mean Square Error)**
   ```
   RMSE = √(Σ(predicted - actual)² / n)
   ```
   目標: < 2.0 mg/dL

2. **相関係数 (R²)**
   ```
   R² = 1 - (SS_res / SS_tot)
   ```
   目標: > 0.85

3. **Bland-Altman分析**
   - 測定値の一致度評価
   - 95%信頼区間の設定

### ベンチマーク

研究論文での報告値：
- RMSE: 1.13 mg/dL
- R²: 0.91
- 分類精度: 96.87%

## 技術的課題と解決策

### 1. 照明条件の変動

**課題**: 異なる光源下での色の変化

**解決策**:
- 色較正カードの使用
- 自動ホワイトバランス調整
- 複数画像の平均化

### 2. 個人差への対応

**課題**: 人種、年齢、性別による結膜色の違い

**解決策**:
- 大規模・多様なデータセットでの学習
- 個人ベースライン機能の実装
- 適応的閾値の設定

### 3. 画像品質の確保

**課題**: ぼやけ、影、反射の影響

**解決策**:
- 画像品質評価アルゴリズム
- 複数フレームからの最良画像選択
- ユーザーガイダンス機能

### 4. リアルタイム処理

**課題**: モバイル端末での高速処理

**解決策**:
- モデルの量子化・軽量化
- エッジコンピューティング
- 段階的処理（粗→精密）

## 参考実装

### 色較正アルゴリズム

```python
def color_calibration(image, reference_colors, measured_colors):
    # 最小二乗法による色変換行列の計算
    transformation_matrix = np.linalg.lstsq(
        measured_colors, 
        reference_colors, 
        rcond=None
    )[0]
    
    # 画像全体への適用
    calibrated = apply_color_transform(image, transformation_matrix)
    return calibrated
```

### 信頼度スコアの計算

```python
def calculate_confidence(features):
    # 1. 画像品質スコア
    quality_score = assess_image_quality(features)
    
    # 2. 特徴の一貫性
    consistency_score = check_feature_consistency(features)
    
    # 3. 予測の不確実性
    prediction_uncertainty = estimate_uncertainty(features)
    
    # 総合信頼度
    confidence = weighted_average([
        quality_score, 
        consistency_score, 
        1 - prediction_uncertainty
    ])
    
    return confidence
```

## まとめ

本手法は、複数の色空間での特徴抽出と機械学習を組み合わせることで、非侵襲的なビリルビン値推定を実現します。現在のプロトタイプは基礎的な実装ですが、将来的には深層学習と大規模データセットを活用した高精度システムへの発展が期待されます。