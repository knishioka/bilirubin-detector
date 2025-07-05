# API リファレンス

[← README](../README.md) | [← 技術手法](methodology.md) | [使用例 →](examples.md) | [開発ガイド →](development.md)

---

## 目次

- [BilirubinDetector](#bilirubindetector)
- [EyeDetector](#eyedetector)
- [ColorAnalyzer](#coloranalyzer)
- [ColorCalibrator](#colorcalibrator)
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

## バージョン互換性

- Python: 3.9+
- OpenCV: 4.5+
- NumPy: 1.19+
- scikit-learn: 0.24+

旧バージョンでの動作は保証されません。