# API リファレンス

[← README](../README.md) | [← 技術手法](methodology.md) | [使用例 →](examples.md) | [開発ガイド →](development.md)

---

## 目次

### ビリルビン検出
- [BilirubinDetector](#bilirubindetector)
- [EyeDetector](#eyedetector)
- [ColorAnalyzer](#coloranalyzer)
- [ColorCalibrator](#colorcalibrator)

### ダークサークル検出
- [DarkCircleDetector](#darkcircledetector)
- [PerioribitalDetector](#perioribitaldetector)
- [DarkCircleAnalyzer](#darkcircleanalyzer)
- [DarkCircleSegmenter](#darkcirclesegmenter)

### ユーティリティ
- [ユーティリティ関数](#ユーティリティ関数)

---

## BilirubinDetector

メインのビリルビン検出クラス。画像からビリルビン値を推定します。

### クラス定義

```python
class BilirubinDetector:
    def __init__(self, calibration_mode: bool = False)
```

#### パラメータ

- `calibration_mode` (bool, optional): 色較正モードを有効にするかどうか。デフォルトは`False`。

### メソッド

#### detect_bilirubin

```python
def detect_bilirubin(self, image_path: str) -> Dict
```

画像からビリルビン値を検出します。

**パラメータ:**
- `image_path` (str): 入力画像のファイルパス

**戻り値:**
- `Dict`: 検出結果を含む辞書

**戻り値の構造:**
```python
{
    'success': bool,              # 検出成功フラグ
    'bilirubin_level_mg_dl': float,  # ビリルビン値 (mg/dL)
    'risk_level': str,            # リスクレベル (low/moderate/high/critical)
    'confidence': float,          # 信頼度スコア (0-1)
    'color_features': dict,       # 色特徴の詳細
    'calibrated': bool,           # 色較正適用フラグ
    'error': str                  # エラーメッセージ（失敗時のみ）
}
```

**使用例:**
```python
detector = BilirubinDetector()
results = detector.detect_bilirubin('eye_image.jpg')
print(f"ビリルビン値: {results['bilirubin_level_mg_dl']} mg/dL")
```

#### visualize_results

```python
def visualize_results(self, image_path: str, results: Dict, output_path: str)
```

検出結果の可視化画像を生成します。

**パラメータ:**
- `image_path` (str): 元画像のパス
- `results` (Dict): `detect_bilirubin`の戻り値
- `output_path` (str): 出力画像の保存パス

---

## EyeDetector

眼球および結膜領域を検出するクラス。

### クラス定義

```python
class EyeDetector:
    def __init__(self)
```

### メソッド

#### detect_conjunctiva

```python
def detect_conjunctiva(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]
```

画像から結膜（白目）領域を検出します。

**パラメータ:**
- `image` (np.ndarray): 入力画像（BGR形式）

**戻り値:**
- `Tuple[Optional[np.ndarray], float]`: 
  - 結膜領域の画像（検出失敗時は`None`）
  - 信頼度スコア（0-1）

**アルゴリズム:**
1. Haar Cascadeによる顔検出
2. 顔領域内での眼検出
3. HSVマスキングによる結膜抽出

---

## ColorAnalyzer

色特徴を抽出・解析するクラス。

### クラス定義

```python
class ColorAnalyzer:
    def __init__(self)
```

### メソッド

#### analyze

```python
def analyze(self, image: np.ndarray) -> Dict
```

画像から色特徴を抽出します。

**パラメータ:**
- `image` (np.ndarray): 入力画像（BGR形式）

**戻り値:**
- `Dict`: 色特徴を含む辞書

**抽出される特徴:**
```python
{
    # RGB特徴
    'rgb_mean_r': float,         # R成分の平均値
    'rgb_mean_g': float,         # G成分の平均値
    'rgb_mean_b': float,         # B成分の平均値
    'rgb_std_r': float,          # R成分の標準偏差
    'rgb_std_g': float,          # G成分の標準偏差
    'rgb_std_b': float,          # B成分の標準偏差
    'rgb_red_blue_ratio': float, # R/B比率
    
    # HSV特徴
    'hsv_mean_h': float,         # 色相の平均値
    'hsv_mean_s': float,         # 彩度の平均値
    'hsv_mean_v': float,         # 明度の平均値
    'hsv_yellow_ratio': float,   # 黄色ピクセルの割合
    'saturation_mean': float,    # 正規化彩度平均
    
    # LAB特徴
    'lab_mean_l': float,         # L*の平均値
    'lab_mean_a': float,         # a*の平均値
    'lab_mean_b': float,         # b*の平均値
    'lab_yellowness': float,     # 黄色度指標
    
    # 複合特徴
    'yellowness_index': float    # 総合黄色度指標
}
```

---

## ColorCalibrator

色較正機能を提供するクラス。

### クラス定義

```python
class ColorCalibrator:
    def __init__(self)
```

### プロパティ

- `is_calibrated` (bool): 較正済みかどうか

### メソッド

#### calibrate_from_card

```python
def calibrate_from_card(self, image: np.ndarray, 
                       card_coords: Optional[Tuple[int, int, int, int]] = None) -> bool
```

較正カードを使用して色較正を実行します。

**パラメータ:**
- `image` (np.ndarray): 較正カードを含む画像
- `card_coords` (Tuple, optional): カード領域の座標 (x, y, width, height)

**戻り値:**
- `bool`: 較正成功フラグ

#### correct_colors

```python
def correct_colors(self, image: np.ndarray) -> np.ndarray
```

較正済みの変換行列を使用して画像の色を補正します。

**パラメータ:**
- `image` (np.ndarray): 入力画像

**戻り値:**
- `np.ndarray`: 色補正済み画像

#### save_calibration / load_calibration

```python
def save_calibration(self, filepath: str)
def load_calibration(self, filepath: str) -> bool
```

較正データの保存・読み込みを行います。

---

## ユーティリティ関数

### image_processing.py

#### preprocess_image

```python
def preprocess_image(image: np.ndarray) -> np.ndarray
```

画像の前処理を実行します。

**処理内容:**
1. サイズ調整（最大1024px）
2. CLAHE によるコントラスト強調
3. ノイズ除去

**パラメータ:**
- `image` (np.ndarray): 入力画像

**戻り値:**
- `np.ndarray`: 前処理済み画像

### color_analysis.py

#### extract_color_features

```python
def extract_color_features(image: np.ndarray) -> Dict
```

`ColorAnalyzer.analyze()`の便利関数版。

#### visualize_color_analysis

```python
def visualize_color_analysis(image: np.ndarray, features: Dict) -> np.ndarray
```

色解析結果の可視化画像を生成します。

**パラメータ:**
- `image` (np.ndarray): 元画像
- `features` (Dict): 色特徴辞書

**戻り値:**
- `np.ndarray`: 可視化画像

### calibration.py

#### create_calibration_card_reference

```python
def create_calibration_card_reference() -> np.ndarray
```

印刷用の色較正カード画像を生成します。

**戻り値:**
- `np.ndarray`: 較正カード画像（6色パッチ）

---

## エラーハンドリング

### 例外

システムは以下の例外を発生させる可能性があります：

- `ValueError`: 無効な入力パラメータ
- `FileNotFoundError`: 画像ファイルが見つからない
- `cv2.error`: OpenCV関連のエラー

### エラー処理の例

```python
from bilirubin_detector import BilirubinDetector

detector = BilirubinDetector()

try:
    results = detector.detect_bilirubin('image.jpg')
    if results['success']:
        print(f"ビリルビン値: {results['bilirubin_level_mg_dl']}")
    else:
        print(f"検出失敗: {results['error']}")
except Exception as e:
    print(f"エラーが発生しました: {str(e)}")
```

---

## パフォーマンス考慮事項

### メモリ使用量

- 画像は自動的に1024px以下にリサイズされます
- 大量の画像を処理する場合は、バッチ処理を推奨

### 処理時間

典型的な処理時間（Intel Core i5）:
- 画像読み込み: ~50ms
- 前処理: ~100ms
- 眼検出: ~200ms
- 色解析: ~50ms
- 合計: ~400ms/画像

### 最適化のヒント

1. **バッチ処理**: 複数画像を連続処理する場合、検出器を再利用
2. **並列処理**: 複数コアでの並列実行が可能
3. **GPU活用**: 将来的な深層学習モデルではGPUを推奨

---

## DarkCircleDetector

ダークサークル（目の下のクマ）を検出・評価するメインクラス。

### クラス定義

```python
class DarkCircleDetector:
    def __init__(self)
```

### メソッド

#### detect_dark_circles

```python
def detect_dark_circles(self, image_path: str) -> Dict
```

画像からダークサークルを検出し、重症度を評価します。

**パラメータ:**
- `image_path` (str): 入力画像のファイルパス

**戻り値:**
- `Dict`: 検出結果を含む辞書

**戻り値の構造:**
```python
{
    'success': bool,              # 検出成功フラグ
    'average_delta_e': float,     # 平均ΔE値（CIE2000）
    'severity': str,              # 重症度 (none/mild/moderate/severe)
    'symmetry_score': float,      # 左右対称性スコア (0-1)
    'left_eye': {                 # 左眼の詳細
        'delta_e': float,         # ΔE値
        'darkness_ratio': float,  # 暗さの比率
        'ita_infraorbital': float,# 眼窩下ITA値
        'ita_cheek': float,       # 頬のITA値
        'lab_infraorbital': dict, # LAB色値
        'lab_cheek': dict,        # LAB色値
        'redness_index': float,   # 赤み指標
        'blueness_index': float   # 青み指標
    },
    'right_eye': dict,            # 右眼の詳細（同構造）
    'face_bbox': tuple,           # 顔領域の座標
    'masks': dict,                # セグメンテーションマスク
    'error': str                  # エラーメッセージ（失敗時のみ）
}
```

**使用例:**
```python
detector = DarkCircleDetector()
results = detector.detect_dark_circles('face_image.jpg')
print(f"ダークサークル重症度: {results['severity']}")
print(f"平均ΔE: {results['average_delta_e']}")
```

#### visualize_results

```python
def visualize_results(self, image_path: str, results: Dict, output_path: str)
```

検出結果の可視化画像を生成します。

**パラメータ:**
- `image_path` (str): 元画像のパス
- `results` (Dict): `detect_dark_circles`の戻り値
- `output_path` (str): 出力画像の保存パス

---

## PerioribitalDetector

眼窩周囲領域（目の周り）を検出するクラス。

### クラス定義

```python
class PerioribitalDetector:
    def __init__(self)
```

### プロパティ

- `infraorbital_ratio` (float): 眼窩下領域の高さ比率（デフォルト: 0.3）
- `cheek_ratio` (float): 頬参照領域の高さ比率（デフォルト: 0.2）
- `lateral_extension` (float): 横方向の拡張比率（デフォルト: 0.2）

### メソッド

#### detect_periorbital_regions

```python
def detect_periorbital_regions(self, image: np.ndarray) -> Dict
```

画像から眼窩周囲領域を検出します。

**パラメータ:**
- `image` (np.ndarray): 入力画像（BGR形式）

**戻り値:**
- `Dict`: 検出結果を含む辞書

**戻り値の構造:**
```python
{
    'success': bool,              # 検出成功フラグ
    'face_bbox': tuple,           # 顔領域 (x, y, w, h)
    'left_eye': np.ndarray,       # 左眼画像
    'left_infraorbital': np.ndarray,  # 左眼窩下領域
    'left_cheek': np.ndarray,     # 左頬参照領域
    'left_eye_bbox': tuple,       # 左眼座標
    'right_eye': np.ndarray,      # 右眼画像
    'right_infraorbital': np.ndarray, # 右眼窩下領域
    'right_cheek': np.ndarray,    # 右頬参照領域
    'right_eye_bbox': tuple,      # 右眼座標
    'error': str                  # エラーメッセージ（失敗時のみ）
}
```

#### draw_regions

```python
def draw_regions(self, image: np.ndarray, detection_result: Dict) -> np.ndarray
```

検出した領域を画像上に描画します（デバッグ用）。

---

## DarkCircleAnalyzer

ダークサークルの色解析を行うクラス。CIELAB色空間での解析を実行。

### クラス定義

```python
class DarkCircleAnalyzer:
    def __init__(self)
```

### メソッド

#### calculate_delta_e

```python
def calculate_delta_e(self, region1: np.ndarray, region2: np.ndarray) -> float
```

2つの領域間のCIE2000 ΔE（色差）を計算します。

**パラメータ:**
- `region1` (np.ndarray): 第1領域（例：眼窩下）
- `region2` (np.ndarray): 第2領域（例：頬）

**戻り値:**
- `float`: ΔE値（色差）

**ΔE値の解釈:**
- < 1.0: 人間の目では区別できない
- 1.0-3.0: わずかに認識可能
- 3.0-5.0: 明確に認識可能
- > 5.0: 大きな色差

#### get_mean_lab

```python
def get_mean_lab(self, region: np.ndarray) -> np.ndarray
```

領域の平均LAB色値を取得します。

**パラメータ:**
- `region` (np.ndarray): 入力領域（BGR形式）

**戻り値:**
- `np.ndarray`: 平均LAB値 [L, a, b]

#### calculate_ita

```python
def calculate_ita(self, lab_values: np.ndarray) -> float
```

Individual Typology Angle (ITA)を計算します。肌色分類に使用。

**パラメータ:**
- `lab_values` (np.ndarray): LAB色値 [L, a, b]

**戻り値:**
- `float`: ITA値（度）

**ITA値による肌色分類:**
- > 55°: Very light
- 41-55°: Light
- 28-41°: Intermediate
- 10-28°: Tan
- -30-10°: Brown
- < -30°: Dark

#### calculate_redness_index / calculate_blueness_index

```python
def calculate_redness_index(self, region: np.ndarray) -> float
def calculate_blueness_index(self, region: np.ndarray) -> float
```

血管性・静脈性ダークサークルの検出用指標を計算します。

**戻り値:**
- `float`: 指標値（0-1、高いほど赤み/青みが強い）

---

## DarkCircleSegmenter

ダークサークル領域をセグメンテーションするクラス。

### クラス定義

```python
class DarkCircleSegmenter:
    def __init__(self)
```

### プロパティ

- `delta_e_threshold` (float): セグメンテーション閾値（デフォルト: 3.0）
- `min_area_ratio` (float): 最小領域比率（デフォルト: 0.05）
- `max_area_ratio` (float): 最大領域比率（デフォルト: 0.7）

### メソッド

#### segment_dark_circle

```python
def segment_dark_circle(self, eye_region: np.ndarray,
                       infraorbital_region: np.ndarray,
                       delta_e: float) -> np.ndarray
```

ダークサークル領域をセグメンテーションします。

**パラメータ:**
- `eye_region` (np.ndarray): 眼領域画像
- `infraorbital_region` (np.ndarray): 眼窩下領域画像
- `delta_e` (float): 全体のΔE値

**戻り値:**
- `np.ndarray`: バイナリマスク（ダークサークル領域が255）

#### create_severity_map

```python
def create_severity_map(self, delta_e_map: np.ndarray) -> np.ndarray
```

ピクセル単位のΔE値から重症度マップを作成します。

**パラメータ:**
- `delta_e_map` (np.ndarray): ピクセル単位のΔE値

**戻り値:**
- `np.ndarray`: 重症度マップ（0-3: なし/軽度/中等度/重度）

---

## バージョン互換性

### ビリルビン検出
- Python: 3.9+
- OpenCV: 4.5+
- NumPy: 1.19+
- scipy: 1.10+

### ダークサークル検出（追加要件）
- scikit-image: 0.21+
- colormath: 3.0+

旧バージョンでの動作は保証されません。