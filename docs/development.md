# 開発ガイド

[← README](../README.md) | [← 使用例](examples.md) | [技術手法 →](methodology.md) | [フロー図 →](algorithm_flowchart.md)

---

## 目次

1. [開発環境のセットアップ](#開発環境のセットアップ)
2. [アーキテクチャ概要](#アーキテクチャ概要)
3. [コーディング規約](#コーディング規約)
4. [テストの実行](#テストの実行)
5. [新機能の追加](#新機能の追加)
6. [デバッグとログ](#デバッグとログ)
7. [パフォーマンス最適化](#パフォーマンス最適化)
8. [コントリビューション](#コントリビューション)

---

## 開発環境のセットアップ

### 必要なツール

- Python 3.9以上
- Git
- VSCode または PyCharm（推奨）
- Docker（オプション）

### 環境構築手順

1. **リポジトリのクローン**
```bash
git clone https://github.com/your-repo/bilirubin-detector.git
cd bilirubin-detector
```

2. **仮想環境の作成**
```bash
# venvを使用
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# またはcondaを使用
conda create -n bilirubin python=3.9
conda activate bilirubin
```

3. **開発用依存関係のインストール**
```bash
# 本番用 + 開発用パッケージ
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **pre-commitフックの設定**
```bash
pre-commit install
```

### requirements-dev.txt

```txt
pytest==7.4.3
pytest-cov==4.1.0
black==23.11.0
flake8==6.1.0
mypy==1.7.1
pre-commit==3.5.0
ipython==8.18.1
jupyter==1.0.0
```

### VSCode推奨設定

`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

---

## アーキテクチャ概要

### システム構成

```
┌─────────────────┐
│   User Input    │
│  (Image File)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BilirubinDetector│
│   (Main Class)   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────────┐ ┌──────────┐
│EyeDetector│ │ColorAnalyzer│
└──────────┘ └──────────┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │
│ (Future: DNN)   │
└─────────────────┘
```

### モジュール構成

- **bilirubin_detector.py**: メインのコントローラー
- **utils/image_processing.py**: 画像処理とコンピュータビジョン
- **utils/color_analysis.py**: 色特徴抽出アルゴリズム
- **utils/calibration.py**: 色較正システム

### データフロー

1. **入力**: 画像ファイル（JPEG/PNG）
2. **前処理**: リサイズ、コントラスト強調、ノイズ除去
3. **検出**: 顔→眼→結膜の階層的検出
4. **特徴抽出**: RGB/HSV/LAB色空間での解析
5. **推定**: 機械学習モデルによるビリルビン値推定
6. **出力**: JSON形式の結果

---

## コーディング規約

### Pythonスタイルガイド

PEP 8に準拠し、以下の追加規則を適用：

```python
# クラス名: PascalCase
class BilirubinDetector:
    pass

# 関数名・変数名: snake_case
def calculate_bilirubin_level(image_path: str) -> float:
    pass

# 定数: UPPER_SNAKE_CASE
MAX_IMAGE_SIZE = 1024
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# プライベートメソッド: アンダースコアプレフィックス
def _internal_method(self):
    pass
```

### 型アノテーション

すべての関数に型アノテーションを使用：

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def process_image(
    image: np.ndarray,
    size: Tuple[int, int] = (224, 224),
    normalize: bool = True
) -> np.ndarray:
    """
    画像を処理する
    
    Args:
        image: 入力画像
        size: リサイズ後のサイズ
        normalize: 正規化するかどうか
        
    Returns:
        処理済み画像
    """
    pass
```

### ドキュメント文字列

Google スタイルのdocstringを使用：

```python
def detect_eye_region(image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """
    画像から眼領域を検出する
    
    Args:
        image: BGR形式の入力画像
        
    Returns:
        Tuple containing:
            - 眼領域の画像（検出失敗時はNone）
            - 信頼度スコア（0-1）
            
    Raises:
        ValueError: 画像が無効な場合
        
    Example:
        >>> detector = EyeDetector()
        >>> eye, confidence = detector.detect_eye_region(image)
        >>> if eye is not None:
        ...     print(f"Eye detected with {confidence:.0%} confidence")
    """
    pass
```

---

## テストの実行

### ユニットテスト

`tests/`ディレクトリにテストを配置：

```python
# tests/test_color_analysis.py
import pytest
import numpy as np
from utils.color_analysis import ColorAnalyzer

class TestColorAnalyzer:
    def setup_method(self):
        self.analyzer = ColorAnalyzer()
    
    def test_yellow_detection(self):
        # 黄色の画像を作成
        yellow_image = np.full((100, 100, 3), [0, 255, 255], dtype=np.uint8)
        features = self.analyzer.analyze(yellow_image)
        
        assert features['hsv_yellow_ratio'] > 0.9
        assert features['yellowness_index'] > 0.5
    
    def test_empty_image(self):
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        features = self.analyzer.analyze(empty_image)
        
        assert features['hsv_yellow_ratio'] == 0
    
    @pytest.mark.parametrize("color,expected_ratio", [
        ([255, 0, 0], 2.0),    # 青
        ([0, 0, 255], 0.5),    # 赤
        ([255, 255, 255], 1.0) # 白
    ])
    def test_rgb_ratio(self, color, expected_ratio):
        image = np.full((50, 50, 3), color, dtype=np.uint8)
        features = self.analyzer.analyze(image)
        
        # 許容誤差を考慮
        assert abs(features['rgb_red_blue_ratio'] - expected_ratio) < 0.1
```

### テストの実行方法

```bash
# すべてのテストを実行
pytest

# カバレッジレポート付き
pytest --cov=utils --cov-report=html

# 特定のテストのみ実行
pytest tests/test_color_analysis.py::TestColorAnalyzer::test_yellow_detection

# 詳細な出力
pytest -v
```

### 統合テスト

```python
# tests/test_integration.py
import os
import tempfile
from bilirubin_detector import BilirubinDetector

def test_end_to_end_detection():
    """エンドツーエンドのテスト"""
    detector = BilirubinDetector()
    
    # テスト画像を作成
    test_image = create_test_eye_image()
    
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
        cv2.imwrite(tmp.name, test_image)
        
        # 検出実行
        results = detector.detect_bilirubin(tmp.name)
        
        # 結果の検証
        assert results['success'] == True
        assert 0 <= results['bilirubin_level_mg_dl'] <= 30
        assert results['risk_level'] in ['low', 'moderate', 'high', 'critical']
        assert 0 <= results['confidence'] <= 1
```

---

## 新機能の追加

### 機能追加のワークフロー

1. **Issue作成**: 機能の目的と要件を明確化
2. **ブランチ作成**: `feature/機能名`形式
3. **実装**: TDD（テスト駆動開発）を推奨
4. **テスト**: ユニットテストと統合テストを作成
5. **ドキュメント**: APIドキュメントと使用例を更新
6. **プルリクエスト**: レビューとマージ

### 例: 新しい色特徴の追加

```python
# 1. utils/color_analysis.py に新機能を追加
def _extract_ycbcr_features(self, ycbcr: np.ndarray) -> Dict:
    """YCbCr色空間での特徴抽出"""
    # Y: 輝度、Cb: 青色差、Cr: 赤色差
    y_mean = np.mean(ycbcr[:, :, 0])
    cb_mean = np.mean(ycbcr[:, :, 1])
    cr_mean = np.mean(ycbcr[:, :, 2])
    
    # 黄色は高いCr値を持つ
    yellowness_ycbcr = cr_mean - cb_mean
    
    return {
        'ycbcr_y_mean': float(y_mean),
        'ycbcr_cb_mean': float(cb_mean),
        'ycbcr_cr_mean': float(cr_mean),
        'ycbcr_yellowness': float(yellowness_ycbcr)
    }

# 2. analyzeメソッドに統合
def analyze(self, image: np.ndarray) -> Dict:
    # 既存のコード...
    
    # YCbCr特徴を追加
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    features.update(self._extract_ycbcr_features(ycbcr))
    
    return features

# 3. テストを作成
def test_ycbcr_features():
    analyzer = ColorAnalyzer()
    yellow_image = create_yellow_test_image()
    features = analyzer.analyze(yellow_image)
    
    assert 'ycbcr_yellowness' in features
    assert features['ycbcr_yellowness'] > 0
```

### プラグインアーキテクチャ

将来的な拡張性のための設計：

```python
# utils/plugins.py
from abc import ABC, abstractmethod

class FeatureExtractor(ABC):
    """特徴抽出プラグインの基底クラス"""
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> Dict:
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

# カスタムプラグインの例
class TextureFeatureExtractor(FeatureExtractor):
    def extract(self, image: np.ndarray) -> Dict:
        # テクスチャ特徴の抽出
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # LBP (Local Binary Patterns)
        lbp = local_binary_pattern(gray, P=8, R=1)
        
        return {
            'texture_mean': np.mean(lbp),
            'texture_std': np.std(lbp)
        }
    
    @property
    def name(self) -> str:
        return "texture"
```

---

## デバッグとログ

### ログ設定

```python
# utils/logger.py
import logging
import sys

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """ロガーのセットアップ"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger

# 使用例
logger = setup_logger(__name__)

def detect_bilirubin(self, image_path: str) -> Dict:
    logger.info(f"Starting detection for: {image_path}")
    
    try:
        # 処理...
        logger.debug(f"Eye region detected with confidence: {confidence}")
        
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        raise
```

### デバッグモード

```python
# 環境変数でデバッグモードを制御
import os

DEBUG_MODE = os.environ.get('BILIRUBIN_DEBUG', 'false').lower() == 'true'

if DEBUG_MODE:
    # 中間結果を保存
    cv2.imwrite('debug_eye_region.jpg', eye_region)
    
    # 詳細情報を出力
    print(f"Color features: {json.dumps(features, indent=2)}")
```

### プロファイリング

```python
# utils/profiling.py
import functools
import time

def profile_time(func):
    """実行時間を計測するデコレーター"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        
        logger.debug(f"{func.__name__} took {end - start:.3f} seconds")
        return result
    return wrapper

# 使用例
@profile_time
def detect_conjunctiva(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    # 処理...
```

---

## パフォーマンス最適化

### 画像処理の最適化

```python
# 1. NumPy ベクトル化
# 悪い例
for i in range(height):
    for j in range(width):
        if image[i, j, 0] > threshold:
            mask[i, j] = 255

# 良い例
mask = (image[:, :, 0] > threshold).astype(np.uint8) * 255

# 2. OpenCV の最適化フラグ
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# 3. 画像のメモリレイアウト
# 連続メモリを確保
if not image.flags['C_CONTIGUOUS']:
    image = np.ascontiguousarray(image)
```

### 並列処理

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def batch_process_parallel(image_paths: List[str]) -> List[Dict]:
    """複数画像を並列処理"""
    detector = BilirubinDetector()
    
    # CPUコア数に基づいてワーカー数を決定
    num_workers = min(len(image_paths), multiprocessing.cpu_count())
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(detector.detect_bilirubin, image_paths))
    
    return results
```

### キャッシング

```python
from functools import lru_cache

class ColorAnalyzer:
    @lru_cache(maxsize=128)
    def _get_yellow_mask(self, image_hash: int) -> np.ndarray:
        """黄色マスクをキャッシュ"""
        # 重い計算...
        return yellow_mask
```

### メモリ最適化

```python
# 大きな画像の段階的処理
def process_large_image(image_path: str, chunk_size: int = 512):
    """大きな画像を分割して処理"""
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    results = []
    
    for y in range(0, height, chunk_size):
        for x in range(0, width, chunk_size):
            chunk = image[y:y+chunk_size, x:x+chunk_size]
            
            # チャンクごとに処理
            chunk_result = process_chunk(chunk)
            results.append(chunk_result)
    
    # 結果を統合
    return merge_results(results)
```

---

## コントリビューション

### 貢献の方法

1. **Issueの報告**
   - バグ報告テンプレートを使用
   - 再現手順を明確に記載
   - 環境情報を含める

2. **機能提案**
   - ユースケースを説明
   - 実装案を提示
   - 既存機能への影響を検討

3. **プルリクエスト**
   - 1つのPRで1つの機能/修正
   - テストを含める
   - ドキュメントを更新

### 開発フロー

```bash
# 1. フォークとクローン
git clone https://github.com/your-username/bilirubin-detector.git
cd bilirubin-detector

# 2. 開発ブランチ作成
git checkout -b feature/your-feature-name

# 3. 変更を実装
# ... コーディング ...

# 4. テスト実行
pytest
black .
flake8 .

# 5. コミット
git add .
git commit -m "feat: add new color feature extraction"

# 6. プッシュとPR作成
git push origin feature/your-feature-name
```

### コードレビューのチェックリスト

- [ ] すべてのテストがパスしている
- [ ] コードカバレッジが低下していない
- [ ] ドキュメントが更新されている
- [ ] 型アノテーションが追加されている
- [ ] パフォーマンスへの影響を検討した
- [ ] 後方互換性が保たれている

### リリースプロセス

1. バージョン番号の更新（セマンティックバージョニング）
2. CHANGELOGの更新
3. テストの完全実行
4. タグの作成とプッシュ
5. リリースノートの作成

---

## トラブルシューティング

### よくある開発上の問題

#### OpenCVのインストールエラー

```bash
# macOS
brew install opencv
pip install opencv-python-headless

# Ubuntu
sudo apt-get update
sudo apt-get install python3-opencv

# Windows
pip install opencv-python
```

#### 型チェックエラー

```bash
# mypyの設定
[mypy]
python_version = 3.9
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
```

### デバッグツール

```python
# IPythonでのインタラクティブデバッグ
from IPython import embed

def debug_point():
    # ここでデバッグ
    embed()
```

このガイドに従って開発を進めることで、高品質で保守性の高いコードを維持できます。