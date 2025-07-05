# トラブルシューティングガイド / Troubleshooting Guide

[← インストール](installation.md) | [← README](../README.md)

---

## 目次

1. [インストール関連の問題](#インストール関連の問題)
2. [実行時エラー](#実行時エラー)
3. [検出精度の問題](#検出精度の問題)
4. [パフォーマンスの問題](#パフォーマンスの問題)
5. [環境固有の問題](#環境固有の問題)
6. [よくある質問（FAQ）](#よくある質問faq)

---

## インストール関連の問題

### OpenCVがインストールできない

#### 症状
```
ERROR: Could not find a version that satisfies the requirement opencv-python
```

#### 解決方法

1. **Pythonバージョンの確認**
```bash
python --version
# Python 3.9以上であることを確認
```

2. **pipのアップグレード**
```bash
pip install --upgrade pip
```

3. **システム依存関係のインストール（Linux）**
```bash
sudo apt-get update
sudo apt-get install -y python3-opencv libopencv-dev
```

4. **代替パッケージの使用**
```bash
# headless版を試す
pip install opencv-python-headless
```

### ModuleNotFoundError: No module named 'cv2'

#### 症状
OpenCVインストール後もインポートエラーが発生

#### 解決方法

1. **仮想環境の確認**
```bash
# 仮想環境が有効化されているか確認
which python
# venv内のpythonが表示されることを確認
```

2. **インストール先の確認**
```bash
pip show opencv-python
# Location: が仮想環境内を指していることを確認
```

3. **Pythonパスの確認**
```python
import sys
print(sys.path)
# site-packagesが含まれていることを確認
```

### colormathのインストールエラー

#### 症状
```
Building wheel for colormath (setup.py) ... error
```

#### 解決方法
```bash
# 開発ツールのインストール（Linux）
sudo apt-get install python3-dev build-essential

# または wheel版を直接インストール
pip install --only-binary :all: colormath
```

---

## 実行時エラー

### error: (-215:Assertion failed) !empty() in function 'detectMultiScale'

#### 症状
顔検出時にOpenCVエラーが発生

#### 原因
Haar Cascadeファイルが正しくロードされていない

#### 解決方法

1. **OpenCVデータパスの確認**
```python
import cv2
print(cv2.data.haarcascades)
# パスが存在することを確認
```

2. **ファイルの存在確認**
```python
import os
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
print(os.path.exists(cascade_path))
# Trueが表示されることを確認
```

3. **手動でファイルをダウンロード**
```bash
# GitHubから直接取得
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye.xml
```

### ValueError: cannot convert float NaN to integer

#### 症状
色解析時に数値エラーが発生

#### 原因
空の画像領域や無効なピクセル値

#### 解決方法

1. **入力画像の確認**
```python
# 画像が正しく読み込まれているか確認
image = cv2.imread('path/to/image.jpg')
if image is None:
    print("画像の読み込みに失敗")
else:
    print(f"画像サイズ: {image.shape}")
```

2. **検出領域の検証**
```python
# 検出された領域が空でないか確認
if conjunctiva_region.size == 0:
    print("結膜領域が検出されませんでした")
```

### Permission denied エラー

#### 症状
ファイルの読み書き時にアクセス拒否

#### 解決方法

1. **ファイル権限の確認**
```bash
ls -la image.jpg
# 読み取り権限があることを確認
```

2. **出力ディレクトリの作成**
```bash
mkdir -p outputs
chmod 755 outputs
```

---

## 検出精度の問題

### ビリルビン値が常に0または異常に高い

#### 原因
- 照明条件が不適切
- 眼球検出の失敗
- 色較正の必要性

#### 解決方法

1. **撮影条件の改善**
- 自然光または白色LED照明を使用
- 影を避ける
- カメラと眼の距離を調整（顔が画面の1/3程度）

2. **画像品質の確認**
```python
# テストスクリプトで品質確認
python test_detector.py --check-image path/to/image.jpg
```

3. **色較正の実施**
```bash
# 較正カードを使用
python bilirubin_detector.py image.jpg --calibrate
```

### ダークサークルが検出されない

#### 原因
- 顔が正面を向いていない
- 眼鏡やマスクによる遮蔽
- 極端な照明条件

#### 解決方法

1. **顔の向きを調整**
- カメラに対して正面を向く
- 顎を少し上げる
- 両目が均等に見えるようにする

2. **障害物の除去**
- 眼鏡を外す
- 前髪が目にかからないようにする
- 十分な明るさを確保

3. **検出パラメータの調整**
```python
# カスタムパラメータで実行
detector = DarkCircleDetector()
detector.severity_thresholds['mild'] = 4.0  # 閾値を調整
```

---

## パフォーマンスの問題

### 処理が非常に遅い

#### 原因
- 画像サイズが大きすぎる
- CPUの性能不足
- メモリ不足

#### 解決方法

1. **画像サイズの最適化**
```python
# 画像を自動リサイズ
# bilirubin_detector.py 内で自動的に1024px以下にリサイズされます
```

2. **バッチ処理の最適化**
```python
# 検出器を再利用
detector = BilirubinDetector()
for image_path in image_list:
    results = detector.detect_bilirubin(image_path)
```

3. **並列処理の活用**
```python
from multiprocessing import Pool

def process_image(path):
    detector = BilirubinDetector()
    return detector.detect_bilirubin(path)

with Pool(4) as p:
    results = p.map(process_image, image_paths)
```

### メモリ不足エラー

#### 症状
```
MemoryError: Unable to allocate array
```

#### 解決方法

1. **メモリ使用量の削減**
```python
# 処理後に明示的にメモリを解放
import gc
gc.collect()
```

2. **画像の逐次処理**
```python
# 一度に全画像を読み込まない
for image_path in image_paths:
    img = cv2.imread(image_path)
    # 処理
    del img  # 明示的に削除
```

---

## 環境固有の問題

### macOSでのOpenCVエラー

#### 症状
```
ImportError: dlopen failed: Library not loaded
```

#### 解決方法
```bash
# Homebrewで依存関係をインストール
brew install opencv
brew link --force opencv

# 環境変数の設定
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

### WSL (Windows Subsystem for Linux) での表示問題

#### 症状
画像が表示されない

#### 解決方法

1. **X11転送の設定**
```bash
# WSL側
export DISPLAY=:0

# Windows側でX11サーバー（VcXsrv等）を起動
```

2. **ヘッドレスモードで実行**
```python
# 画像表示の代わりにファイル保存
# --output オプションを使用
python bilirubin_detector.py image.jpg --output result.png
```

### Dockerでのエラー

#### 症状
```
libGL error: No matching fbConfigs or visuals found
```

#### 解決方法
```dockerfile
# Dockerfileに追加
ENV QT_X11_NO_MITSHM=1
RUN apt-get update && apt-get install -y libgl1-mesa-glx
```

---

## よくある質問（FAQ）

### Q: 精度はどの程度ですか？

A: 研究論文では85-96%の精度が報告されていますが、本プロトタイプは：
- 実際の臨床データでの学習は未実施
- 照明条件に大きく依存
- 医療診断には使用できません

### Q: スマートフォンで撮影した画像は使えますか？

A: はい、使用できます。以下の点に注意してください：
- フラッシュは使用しない
- 手ぶれに注意
- 十分な解像度（1280x720以上推奨）

### Q: リアルタイム検出は可能ですか？

A: 現在は静止画のみ対応していますが、将来的には：
```python
# Webカメラからのリアルタイム入力（開発中）
python bilirubin_detector.py --camera
```

### Q: 他の言語から使用できますか？

A: Python APIとして設計されていますが、以下の方法で連携可能：
- REST API化（Flask/FastAPIでラップ）
- コマンドライン経由でのJSON出力
- Python bindingsの作成

### Q: GPUは必要ですか？

A: 現在のバージョンではGPUは不要です。将来の深層学習モデルでは：
- CUDA対応GPUで高速化可能
- CPU版でも動作可能（低速）

### Q: 商用利用は可能ですか？

A: 本システムは研究・教育目的のプロトタイプです：
- 医療機器としての認証は受けていません
- 診断目的での使用は禁止されています
- ライセンスについてはLICENSEファイルを確認してください

---

## サポート

### 問題が解決しない場合

1. **GitHubイシューの確認**
   - [既存のイシュー](https://github.com/knishioka/bilirubin-detector/issues)を検索
   - 同様の問題が報告されていないか確認

2. **新規イシューの作成**
   - エラーメッセージの全文
   - 実行環境の詳細（OS、Pythonバージョン等）
   - 再現手順
   - 試した解決方法

3. **コミュニティサポート**
   - ディスカッションフォーラムで質問
   - 他のユーザーの経験を参考に

### デバッグモード

詳細なログ出力を有効にする：
```bash
# 環境変数でデバッグモードを有効化
export HEALTH_TECH_DEBUG=1
python bilirubin_detector.py image.jpg

# またはPythonコード内で
import logging
logging.basicConfig(level=logging.DEBUG)
```