# 使用例とチュートリアル

[← README](../README.md) | [← APIリファレンス](api_reference.md) | [開発ガイド →](development.md) | [フロー図 →](algorithm_flowchart.md)

---

## 目次

1. [基本的な使用例](#基本的な使用例)
2. [高度な使用例](#高度な使用例)
3. [統合例](#統合例)
4. [トラブルシューティング](#トラブルシューティング)

---

## 基本的な使用例

### 1. 単一画像の解析

最も基本的な使用方法です。

```python
from bilirubin_detector import BilirubinDetector

# 検出器の初期化
detector = BilirubinDetector()

# 画像からビリルビン値を検出
results = detector.detect_bilirubin('eye_photo.jpg')

# 結果の表示
if results['success']:
    print(f"ビリルビン値: {results['bilirubin_level_mg_dl']} mg/dL")
    print(f"リスクレベル: {results['risk_level']}")
    print(f"信頼度: {results['confidence']:.2%}")
else:
    print(f"検出失敗: {results['error']}")
```

### 2. 結果の可視化

検出結果を視覚的に確認する方法です。

```python
from bilirubin_detector import BilirubinDetector
import matplotlib.pyplot as plt
import cv2

detector = BilirubinDetector()

# 検出実行
image_path = 'eye_photo.jpg'
results = detector.detect_bilirubin(image_path)

if results['success']:
    # 可視化画像を生成
    detector.visualize_results(image_path, results, 'result_visualization.png')
    
    # 結果を表示
    viz_image = cv2.imread('result_visualization.png')
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('ビリルビン検出結果')
    plt.show()
```

### 3. JSON出力の利用

他のシステムとの連携に便利なJSON形式での出力例です。

```python
import json
from bilirubin_detector import BilirubinDetector

detector = BilirubinDetector()
results = detector.detect_bilirubin('eye_photo.jpg')

# JSON形式で保存
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 後で読み込み
with open('results.json', 'r', encoding='utf-8') as f:
    loaded_results = json.load(f)
    print(f"保存されたビリルビン値: {loaded_results['bilirubin_level_mg_dl']}")
```

---

## 高度な使用例

### 1. 色較正を使用した高精度測定

```python
from bilirubin_detector import BilirubinDetector
from utils.calibration import create_calibration_card_reference
import cv2

# Step 1: 較正カードの生成（初回のみ）
calibration_card = create_calibration_card_reference()
cv2.imwrite('calibration_card.png', calibration_card)
print("較正カードを印刷してください: calibration_card.png")

# Step 2: 較正カードを含む画像で較正
detector = BilirubinDetector(calibration_mode=True)

# 較正カードと一緒に撮影した画像を使用
calibration_image = cv2.imread('image_with_calibration_card.jpg')
success = detector.calibrator.calibrate_from_card(calibration_image)

if success:
    print("色較正が完了しました")
    
    # Step 3: 較正済み検出器で測定
    results = detector.detect_bilirubin('eye_photo.jpg')
    print(f"較正済みビリルビン値: {results['bilirubin_level_mg_dl']} mg/dL")
    
    # 較正データの保存
    detector.calibrator.save_calibration('calibration_data.npz')
else:
    print("較正に失敗しました")
```

### 2. バッチ処理

複数の画像を効率的に処理する例です。

```python
import os
import pandas as pd
from bilirubin_detector import BilirubinDetector
from tqdm import tqdm

def batch_process_images(image_folder, output_csv):
    """フォルダ内の全画像を処理してCSVに保存"""
    detector = BilirubinDetector()
    results_list = []
    
    # 画像ファイルのリスト取得
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # プログレスバーを表示しながら処理
    for filename in tqdm(image_files, desc="画像処理中"):
        image_path = os.path.join(image_folder, filename)
        
        try:
            results = detector.detect_bilirubin(image_path)
            
            # 結果を記録
            results_list.append({
                'filename': filename,
                'success': results['success'],
                'bilirubin_mg_dl': results.get('bilirubin_level_mg_dl', None),
                'risk_level': results.get('risk_level', None),
                'confidence': results.get('confidence', None),
                'hsv_yellow_ratio': results.get('color_features', {}).get('hsv_yellow_ratio', None)
            })
        except Exception as e:
            results_list.append({
                'filename': filename,
                'success': False,
                'error': str(e)
            })
    
    # DataFrameに変換して保存
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    
    # 統計情報の表示
    successful = df[df['success'] == True]
    print(f"\n処理完了: {len(successful)}/{len(image_files)} 成功")
    print(f"平均ビリルビン値: {successful['bilirubin_mg_dl'].mean():.2f} mg/dL")
    print(f"リスクレベル分布:")
    print(successful['risk_level'].value_counts())
    
    return df

# 使用例
results_df = batch_process_images('patient_images/', 'batch_results.csv')
```

### 3. リアルタイムWebカメラ検出

```python
import cv2
from bilirubin_detector import BilirubinDetector
import time

def realtime_detection():
    """Webカメラからリアルタイムで検出"""
    detector = BilirubinDetector()
    cap = cv2.VideoCapture(0)
    
    print("スペースキーで検出、ESCで終了")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # フレームを表示
        cv2.imshow('ビリルビン検出システム', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # スペースキーで検出実行
        if key == ord(' '):
            # 一時ファイルに保存
            temp_path = 'temp_capture.jpg'
            cv2.imwrite(temp_path, frame)
            
            # 検出実行
            print("\n検出中...")
            start_time = time.time()
            results = detector.detect_bilirubin(temp_path)
            elapsed_time = time.time() - start_time
            
            if results['success']:
                print(f"ビリルビン値: {results['bilirubin_level_mg_dl']} mg/dL")
                print(f"リスクレベル: {results['risk_level']}")
                print(f"処理時間: {elapsed_time:.2f}秒")
                
                # 結果を画面に表示
                text = f"Bilirubin: {results['bilirubin_level_mg_dl']:.1f} mg/dL"
                cv2.putText(frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('結果', frame)
                cv2.waitKey(3000)  # 3秒間表示
            else:
                print(f"検出失敗: {results['error']}")
        
        # ESCで終了
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# 実行
realtime_detection()
```

### 4. カスタム色特徴の利用

```python
from utils.color_analysis import ColorAnalyzer
import cv2
import numpy as np

def analyze_color_trends(image_paths):
    """複数画像の色特徴の傾向を分析"""
    analyzer = ColorAnalyzer()
    all_features = []
    
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            features = analyzer.analyze(image)
            all_features.append(features)
    
    # 特徴の統計を計算
    feature_names = all_features[0].keys()
    stats = {}
    
    for feature in feature_names:
        values = [f[feature] for f in all_features]
        stats[feature] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # 黄疸の指標となる特徴を表示
    print("=== 色特徴の統計 ===")
    key_features = ['hsv_yellow_ratio', 'rgb_red_blue_ratio', 'yellowness_index']
    
    for feature in key_features:
        s = stats[feature]
        print(f"\n{feature}:")
        print(f"  平均: {s['mean']:.3f} (±{s['std']:.3f})")
        print(f"  範囲: {s['min']:.3f} - {s['max']:.3f}")
    
    return stats
```

---

## 統合例

### 1. Webアプリケーションとの統合

Flask を使用した簡単なWeb APIの例です。

```python
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from bilirubin_detector import BilirubinDetector
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

detector = BilirubinDetector()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/detect', methods=['POST'])
def detect_bilirubin():
    """画像をアップロードしてビリルビン値を検出"""
    if 'image' not in request.files:
        return jsonify({'error': '画像がアップロードされていません'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'ファイルが選択されていません'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # 検出実行
            results = detector.detect_bilirubin(filepath)
            
            # 一時ファイルを削除
            os.remove(filepath)
            
            return jsonify(results)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': '無効なファイル形式です'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, port=5000)
```

### 2. データベースとの連携

検出結果をデータベースに保存する例です。

```python
import sqlite3
from datetime import datetime
from bilirubin_detector import BilirubinDetector

class BilirubinDatabase:
    def __init__(self, db_path='bilirubin_records.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """テーブルの作成"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                bilirubin_level REAL,
                risk_level TEXT,
                confidence REAL,
                hsv_yellow_ratio REAL,
                rgb_ratio REAL,
                image_path TEXT
            )
        ''')
        self.conn.commit()
    
    def save_measurement(self, patient_id, results, image_path=None):
        """測定結果を保存"""
        cursor = self.conn.cursor()
        
        if results['success']:
            cursor.execute('''
                INSERT INTO measurements 
                (patient_id, bilirubin_level, risk_level, confidence, 
                 hsv_yellow_ratio, rgb_ratio, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                results['bilirubin_level_mg_dl'],
                results['risk_level'],
                results['confidence'],
                results['color_features'].get('hsv_yellow_ratio'),
                results['color_features'].get('rgb_red_blue_ratio'),
                image_path
            ))
            self.conn.commit()
            return cursor.lastrowid
        return None
    
    def get_patient_history(self, patient_id, limit=10):
        """患者の測定履歴を取得"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT timestamp, bilirubin_level, risk_level, confidence
            FROM measurements
            WHERE patient_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (patient_id, limit))
        
        return cursor.fetchall()
    
    def get_statistics(self, patient_id):
        """患者の統計情報を取得"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                COUNT(*) as total_measurements,
                AVG(bilirubin_level) as avg_bilirubin,
                MIN(bilirubin_level) as min_bilirubin,
                MAX(bilirubin_level) as max_bilirubin
            FROM measurements
            WHERE patient_id = ?
        ''', (patient_id,))
        
        return cursor.fetchone()

# 使用例
def track_patient_bilirubin(patient_id, image_path):
    detector = BilirubinDetector()
    db = BilirubinDatabase()
    
    # 検出実行
    results = detector.detect_bilirubin(image_path)
    
    # データベースに保存
    measurement_id = db.save_measurement(patient_id, results, image_path)
    
    if measurement_id:
        print(f"測定結果を保存しました (ID: {measurement_id})")
        
        # 履歴を表示
        history = db.get_patient_history(patient_id)
        print(f"\n患者 {patient_id} の測定履歴:")
        for timestamp, level, risk, confidence in history:
            print(f"  {timestamp}: {level:.1f} mg/dL ({risk}) - 信頼度: {confidence:.2%}")
        
        # 統計情報
        stats = db.get_statistics(patient_id)
        print(f"\n統計情報:")
        print(f"  測定回数: {stats[0]}")
        print(f"  平均値: {stats[1]:.2f} mg/dL")
        print(f"  範囲: {stats[2]:.2f} - {stats[3]:.2f} mg/dL")
```

---

## トラブルシューティング

### よくある問題と解決方法

#### 1. 眼球が検出されない

```python
from utils.image_processing import EyeDetector
import cv2

def debug_eye_detection(image_path):
    """眼球検出のデバッグ"""
    detector = EyeDetector()
    image = cv2.imread(image_path)
    
    # グレースケールで顔検出を試行
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.face_cascade.detectMultiScale(gray, 1.1, 3)
    
    print(f"検出された顔の数: {len(faces)}")
    
    # 検出された領域を描画
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, f"Face {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    cv2.imwrite('debug_faces.jpg', image)
    print("デバッグ画像を保存しました: debug_faces.jpg")
    
    # 推奨事項
    print("\n推奨事項:")
    print("- 顔が正面を向いているか確認")
    print("- 明るい照明で撮影")
    print("- 画像サイズが小さすぎないか確認")
```

#### 2. 信頼度が低い

```python
def improve_confidence(image_path):
    """信頼度を改善するためのヒント"""
    from utils.image_processing import preprocess_image
    import cv2
    
    image = cv2.imread(image_path)
    
    # 画像品質をチェック
    def check_image_quality(img):
        # 明るさチェック
        brightness = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
        
        # コントラストチェック
        contrast = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).std()
        
        # ブラーチェック
        laplacian_var = cv2.Laplacian(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F
        ).var()
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': laplacian_var
        }
    
    quality = check_image_quality(image)
    
    print("画像品質分析:")
    print(f"  明るさ: {quality['brightness']:.1f} (推奨: 100-200)")
    print(f"  コントラスト: {quality['contrast']:.1f} (推奨: >30)")
    print(f"  鮮明度: {quality['sharpness']:.1f} (推奨: >100)")
    
    # 改善提案
    suggestions = []
    if quality['brightness'] < 100:
        suggestions.append("画像が暗すぎます。明るい場所で撮影してください。")
    elif quality['brightness'] > 200:
        suggestions.append("画像が明るすぎます。露出を下げてください。")
    
    if quality['contrast'] < 30:
        suggestions.append("コントラストが低いです。背景を単色にしてください。")
    
    if quality['sharpness'] < 100:
        suggestions.append("画像がぼやけています。カメラを固定して撮影してください。")
    
    if suggestions:
        print("\n改善提案:")
        for s in suggestions:
            print(f"  - {s}")
    else:
        print("\n画像品質は良好です。")
```

#### 3. 色較正のトラブル

```python
def debug_calibration(image_path):
    """色較正のデバッグ"""
    from utils.calibration import ColorCalibrator
    import cv2
    
    calibrator = ColorCalibrator()
    image = cv2.imread(image_path)
    
    # 較正カードの検出を試行
    card_region, coords = calibrator._detect_calibration_card(image)
    
    if card_region is not None:
        print("較正カードが検出されました")
        x, y, w, h = coords
        
        # 検出領域を描画
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('debug_calibration.jpg', debug_img)
        
        # 色パッチを抽出
        colors = calibrator._extract_color_patches(card_region)
        if colors is not None:
            print(f"抽出された色数: {len(colors)}")
            for i, color in enumerate(colors):
                print(f"  パッチ {i+1}: BGR({color[0]:.0f}, {color[1]:.0f}, {color[2]:.0f})")
        else:
            print("色パッチの抽出に失敗しました")
    else:
        print("較正カードが検出されませんでした")
        print("確認事項:")
        print("  - 較正カード全体が画像に含まれているか")
        print("  - カードが平らに置かれているか")
        print("  - 照明が均一か")
```

### パフォーマンスの最適化

```python
import time
import cProfile
import pstats

def profile_detection(image_path):
    """処理時間のプロファイリング"""
    from bilirubin_detector import BilirubinDetector
    
    detector = BilirubinDetector()
    
    # プロファイリング実行
    profiler = cProfile.Profile()
    profiler.enable()
    
    start_time = time.time()
    results = detector.detect_bilirubin(image_path)
    end_time = time.time()
    
    profiler.disable()
    
    # 結果を表示
    print(f"総処理時間: {end_time - start_time:.3f}秒")
    
    # 詳細な統計
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 上位10個の関数を表示
    
    # ボトルネックの特定
    print("\nボトルネック分析:")
    stats.print_callers(5)  # 最も時間がかかる関数の呼び出し元を表示
```

これらの例を参考に、様々なユースケースでビリルビン検出システムを活用してください。