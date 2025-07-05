# インストールガイド / Installation Guide

[← README](../README.md) | [トラブルシューティング →](troubleshooting.md)

---

## 目次

1. [システム要件](#システム要件)
2. [クイックインストール](#クイックインストール)
3. [詳細インストール手順](#詳細インストール手順)
4. [依存関係の説明](#依存関係の説明)
5. [環境別セットアップ](#環境別セットアップ)
6. [動作確認](#動作確認)
7. [アップデート方法](#アップデート方法)

---

## システム要件

### 必須要件

- **Python**: 3.9以上（3.11推奨）
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 20.04+
- **メモリ**: 4GB以上（8GB推奨）
- **ストレージ**: 1GB以上の空き容量

### ハードウェア要件

- **CPU**: Intel Core i3以上または同等のAMDプロセッサ
- **カメラ**: Webカメラまたはスマートフォンカメラ（リアルタイム検出用）
- **GPU**: 不要（将来の深層学習モデルではCUDA対応GPU推奨）

---

## クイックインストール

### uvを使用した推奨インストール方法

```bash
# 1. リポジトリのクローン
git clone https://github.com/knishioka/bilirubin-detector.git
cd bilirubin-detector

# 2. uvのインストール（未インストールの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Python環境の作成
uv venv

# 4. 環境の有効化
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 5. 依存関係のインストール
uv pip install -r requirements.txt

# 6. 動作確認
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### pipを使用した従来のインストール方法

```bash
# 1. リポジトリのクローン
git clone https://github.com/knishioka/bilirubin-detector.git
cd bilirubin-detector

# 2. 仮想環境の作成
python -m venv venv

# 3. 環境の有効化
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# 4. 依存関係のインストール
pip install -r requirements.txt
```

---

## 詳細インストール手順

### 1. Pythonのインストール

#### Windows
1. [Python公式サイト](https://www.python.org/downloads/)から最新の3.11.xをダウンロード
2. インストーラを実行（"Add Python to PATH"にチェック）
3. コマンドプロンプトで確認：
   ```cmd
   python --version
   ```

#### macOS
```bash
# Homebrewを使用
brew install python@3.11

# または pyenvを使用
pyenv install 3.11.8
pyenv global 3.11.8
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

### 2. Git のインストール

#### Windows
[Git for Windows](https://gitforwindows.org/)をダウンロードしてインストール

#### macOS
```bash
brew install git
```

#### Ubuntu/Debian
```bash
sudo apt install git
```

### 3. プロジェクトのセットアップ

```bash
# リポジトリのクローン
git clone https://github.com/knishioka/bilirubin-detector.git
cd bilirubin-detector

# ブランチの確認
git branch -a

# 最新版の取得
git pull origin master
```

---

## 依存関係の説明

### 主要パッケージ

| パッケージ | バージョン | 用途 |
|----------|----------|------|
| opencv-python | 4.9.0+ | 画像処理、顔・眼検出 |
| numpy | 1.24+ | 数値計算、配列操作 |
| scipy | 1.10+ | 科学技術計算、統計処理 |
| scikit-image | 0.21+ | 高度な画像処理 |
| matplotlib | 3.7+ | 可視化、グラフ作成 |
| Pillow | 10.0+ | 画像ファイルI/O |
| colormath | 3.0+ | 色空間変換、ΔE計算 |

### OpenCVの追加設定

OpenCVが正しくインストールされない場合：

```bash
# システムレベルの依存関係をインストール（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0

# macOSでの追加設定
brew install opencv
```

---

## 環境別セットアップ

### Docker環境

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# システム依存関係
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "bilirubin_detector.py"]
```

使用方法：
```bash
# イメージのビルド
docker build -t health-tech .

# コンテナの実行
docker run -v $(pwd)/images:/app/images health-tech python bilirubin_detector.py /app/images/sample.jpg
```

### Google Colab環境

```python
# Colabでの実行
!git clone https://github.com/knishioka/bilirubin-detector.git
%cd bilirubin-detector
!pip install -r requirements.txt

# ファイルのアップロード
from google.colab import files
uploaded = files.upload()

# 実行
!python bilirubin_detector.py uploaded_image.jpg
```

### Jupyter Notebook環境

```bash
# Jupyter環境の準備
pip install jupyter notebook ipykernel

# カーネルの登録
python -m ipykernel install --user --name health-tech --display-name "Health Tech"

# Notebookの起動
jupyter notebook
```

---

## 動作確認

### 1. 基本的な動作確認

```bash
# Pythonパッケージの確認
python -c "import cv2, numpy, scipy, matplotlib; print('All packages imported successfully')"

# OpenCVの詳細確認
python -c "import cv2; print(cv2.getBuildInformation())"

# サンプル画像での実行テスト
python test_detector.py --test-import
```

### 2. 機能別動作確認

```bash
# ビリルビン検出のテスト
python bilirubin_detector.py sample_images/eye_mild_jaundice.jpg

# ダークサークル検出のテスト
python generate_dark_circle_samples.py
python dark_circle_detector.py samples/dark_circles/sample_000_light_pigmentation_none.jpg

# 全テストの実行
python test_detector.py --test-all
python test_dark_circle_detection.py
```

### 3. よくある問題と解決方法

#### ImportError: No module named 'cv2'
```bash
# OpenCVの再インストール
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

#### NumPy バージョンの競合
```bash
# NumPyの再インストール
pip install --upgrade numpy
```

詳細なトラブルシューティングは[トラブルシューティングガイド](troubleshooting.md)を参照してください。

---

## アップデート方法

### 最新版への更新

```bash
# リポジトリの更新
git pull origin master

# 依存関係の更新
pip install --upgrade -r requirements.txt

# または uvを使用
uv pip install --upgrade -r requirements.txt
```

### 特定バージョンへの切り替え

```bash
# タグ一覧の確認
git tag -l

# 特定バージョンへ切り替え
git checkout v1.0.0

# 依存関係の再インストール
pip install -r requirements.txt
```

---

## 次のステップ

インストールが完了したら：

1. [使用例とチュートリアル](examples.md)で基本的な使い方を学ぶ
2. [APIリファレンス](api_reference.md)で詳細な機能を確認
3. [開発ガイド](development.md)でカスタマイズ方法を学ぶ

問題が発生した場合は[トラブルシューティングガイド](troubleshooting.md)を参照してください。