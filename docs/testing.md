# テストガイド / Testing Guide

[← README](../README.md) | [開発ガイド →](development.md)

---

## 目次

1. [テスト概要](#テスト概要)
2. [テスト環境のセットアップ](#テスト環境のセットアップ)
3. [テストの実行](#テストの実行)
4. [テストデータの準備](#テストデータの準備)
5. [テスト結果の解釈](#テスト結果の解釈)
6. [テストケース一覧](#テストケース一覧)
7. [パフォーマンステスト](#パフォーマンステスト)

---

## テスト概要

### テスト戦略

本システムでは以下のレベルのテストを実施：

1. **統合テスト**: 実際の画像を使用した検出精度の確認
2. **機能テスト**: 各モジュールの基本動作確認
3. **エラーテスト**: 異常系の動作確認
4. **パフォーマンステスト**: 処理速度とメモリ使用量の測定

### テストツール

- **pytest**: ユニットテストフレームワーク（将来実装）
- **統合テストスクリプト**: `test_detector.py`, `test_dark_circle_detection.py`
- **サンプル生成スクリプト**: テスト画像の自動生成

---

## テスト環境のセットアップ

### 必要なパッケージ

```bash
# テスト用パッケージのインストール
pip install pytest pytest-cov pytest-benchmark

# またはuvを使用
uv pip install pytest pytest-cov pytest-benchmark
```

### ディレクトリ構造

```
health-tech/
├── scripts/
│   ├── evaluation/
│   │   ├── test_detector.py
│   │   └── test_dark_circle_detection.py
│   └── data_generation/
│       ├── create_jaundice_samples.py
│       └── generate_dark_circle_samples.py
├── sample_images/         # ビリルビン検出用
├── samples/
│   └── dark_circles/     # ダークサークル検出用
└── outputs/              # テスト結果出力
```

### 環境変数の設定

```bash
# デバッグモードの有効化
export HEALTH_TECH_DEBUG=1

# テスト結果の出力先
export TEST_OUTPUT_DIR=./test_results
```

---

## テストの実行

### ビリルビン検出のテスト

#### 基本的なテスト実行

```bash
# インポートテスト
python scripts/evaluation/test_detector.py --test-import

# サンプル画像の生成
python scripts/data_generation/create_jaundice_samples.py

# 全テストの実行
python scripts/evaluation/test_detector.py --test-all
```

#### 個別テストの実行

```bash
# 特定の重症度レベルのテスト
python scripts/evaluation/test_detector.py --test-severity mild

# カスタム画像でのテスト
python bilirubin_detector.py path/to/image.jpg --json
```

### ダークサークル検出のテスト

#### サンプル生成とテスト

```bash
# サンプル画像の生成
python scripts/data_generation/generate_dark_circle_samples.py

# テストの実行
python scripts/evaluation/test_dark_circle_detection.py
```

#### テスト内容

1. **眼窩周囲検出テスト**
   - 顔検出の成功率
   - 眼の位置特定精度
   - 領域抽出の正確性

2. **色解析テスト**
   - ΔE計算の妥当性
   - ITA値の正確性
   - 色指標の範囲確認

3. **重症度分類テスト**
   - 各レベルの正しい分類
   - 閾値の妥当性確認

### バッチテストの実行

```bash
# 複数画像の一括テスト
for img in sample_images/*.jpg; do
    python bilirubin_detector.py "$img" --json >> test_results.jsonl
done

# 結果の集計
python -c "
import json
with open('test_results.jsonl') as f:
    results = [json.loads(line) for line in f]
    success_rate = sum(r['success'] for r in results) / len(results)
    print(f'Success rate: {success_rate:.2%}')
"
```

---

## テストデータの準備

### 合成テストデータ

#### ビリルビン検出用

```python
# create_jaundice_samples.py の使用
severity_levels = ['none', 'mild', 'moderate', 'severe']
for level in severity_levels:
    create_sample_eye_image(f'eye_{level}_jaundice.jpg', level)
```

生成される画像の特徴：
- 結膜の黄色度が段階的に変化
- 一定の照明条件
- ノイズとテクスチャを含む

#### ダークサークル検出用

```python
# generate_dark_circle_samples.py の設定
skin_tones = ['light', 'medium', 'tan', 'dark']
dark_circle_types = ['pigmentation', 'vascular', 'structural', 'mixed']
severities = ['none', 'mild', 'moderate', 'severe']
```

生成パラメータ：
- 4種類の肌色
- 4種類のダークサークルタイプ
- 4段階の重症度
- 合計64パターン

### 実画像でのテスト

#### 推奨される撮影条件

1. **照明**
   - 自然光（窓際）
   - 白色LED（5000K以上）
   - 直接光を避ける

2. **カメラ設定**
   - 解像度: 1280x720以上
   - フォーカス: オート
   - フラッシュ: オフ

3. **被写体**
   - カメラから30-50cm
   - 目を大きく開く
   - 正面を向く

### テストデータの検証

```python
# 画像品質の確認
def validate_test_image(image_path):
    img = cv2.imread(image_path)
    
    # サイズチェック
    assert img.shape[0] >= 480, "Height too small"
    assert img.shape[1] >= 640, "Width too small"
    
    # 明るさチェック
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    assert 50 < mean_brightness < 200, "Brightness out of range"
    
    # コントラストチェック
    std_brightness = np.std(gray)
    assert std_brightness > 20, "Contrast too low"
```

---

## テスト結果の解釈

### 成功基準

#### ビリルビン検出

| 指標 | 基準値 | 説明 |
|------|--------|------|
| 検出成功率 | > 90% | 眼球・結膜が正しく検出される割合 |
| 相関係数 | > 0.7 | 実測値との相関（臨床データ必要） |
| 処理時間 | < 1秒 | 1画像あたりの処理時間 |

#### ダークサークル検出

| 指標 | 基準値 | 説明 |
|------|--------|------|
| 顔検出率 | > 95% | 正面顔での検出成功率 |
| 重症度一致率 | > 80% | 期待される重症度との一致 |
| 左右対称性 | > 0.8 | 左右の検出結果の一致度 |

### エラー分析

#### 一般的なエラーパターン

1. **検出失敗**
   ```json
   {
     "success": false,
     "error": "No face detected"
   }
   ```
   - 原因: 顔の向き、照明、遮蔽物
   - 対策: 撮影条件の改善

2. **異常値**
   ```json
   {
     "success": true,
     "bilirubin_level_mg_dl": 0.0,
     "confidence": 0.1
   }
   ```
   - 原因: 結膜領域の誤検出
   - 対策: 信頼度閾値の調整

### 結果の可視化

```python
# テスト結果の可視化
import matplotlib.pyplot as plt

def plot_test_results(results):
    # 成功率のグラフ
    plt.figure(figsize=(10, 6))
    
    # 重症度別の成功率
    severities = ['none', 'mild', 'moderate', 'severe']
    success_rates = [
        calculate_success_rate(results, severity) 
        for severity in severities
    ]
    
    plt.bar(severities, success_rates)
    plt.ylabel('Success Rate')
    plt.title('Detection Success Rate by Severity')
    plt.ylim(0, 1)
    plt.show()
```

---

## テストケース一覧

### ビリルビン検出テストケース

| ID | テストケース | 期待結果 | 優先度 |
|----|-------------|---------|--------|
| B01 | 正常な眼球画像 | ビリルビン値 < 3 mg/dL | 高 |
| B02 | 軽度黄疸画像 | ビリルビン値 3-12 mg/dL | 高 |
| B03 | 中等度黄疸画像 | ビリルビン値 12-20 mg/dL | 高 |
| B04 | 重度黄疸画像 | ビリルビン値 > 20 mg/dL | 高 |
| B05 | 眼鏡着用画像 | エラーまたは低信頼度 | 中 |
| B06 | 横向き顔画像 | 検出失敗 | 中 |
| B07 | 暗い画像 | エラーまたは警告 | 低 |
| B08 | 過露出画像 | エラーまたは警告 | 低 |

### ダークサークル検出テストケース

| ID | テストケース | 期待結果 | 優先度 |
|----|-------------|---------|--------|
| D01 | ダークサークルなし | ΔE < 3.0 | 高 |
| D02 | 軽度ダークサークル | ΔE 3.0-5.0 | 高 |
| D03 | 中等度ダークサークル | ΔE 5.0-8.0 | 高 |
| D04 | 重度ダークサークル | ΔE > 8.0 | 高 |
| D05 | 異なる肌色（4種） | 各肌色で正しい検出 | 高 |
| D06 | 片側のみダークサークル | 低対称性スコア | 中 |
| D07 | メイクあり | 検出可能（精度低下） | 低 |
| D08 | 複数人の顔 | 最大の顔を選択 | 低 |

---

## パフォーマンステスト

### 処理速度の測定

```python
import time

def benchmark_detection(detector, image_paths, iterations=10):
    times = []
    
    for _ in range(iterations):
        start = time.time()
        for path in image_paths:
            detector.detect_bilirubin(path)
        end = time.time()
        times.append(end - start)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'fps': len(image_paths) / np.mean(times)
    }
```

### メモリ使用量の測定

```python
import psutil
import os

def measure_memory_usage():
    process = psutil.Process(os.getpid())
    
    # 初期メモリ
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 処理実行
    detector = BilirubinDetector()
    for i in range(100):
        detector.detect_bilirubin(f'sample_{i}.jpg')
    
    # 最終メモリ
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    return {
        'initial_mb': initial_memory,
        'final_mb': final_memory,
        'increase_mb': final_memory - initial_memory
    }
```

### 最適化の効果測定

```bash
# プロファイリングの実行
python -m cProfile -s cumulative bilirubin_detector.py sample.jpg

# 結果の分析
# - 最も時間がかかる関数の特定
# - ボトルネックの発見
# - 最適化対象の決定
```

---

## 継続的テスト

### GitHub Actions設定例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        python scripts/evaluation/test_detector.py --test-all
        python scripts/evaluation/test_dark_circle_detection.py
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: test-results
        path: outputs/
```

### ローカルでの自動テスト

```bash
# pre-commitフックの設定
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
python scripts/evaluation/test_detector.py --test-import
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```