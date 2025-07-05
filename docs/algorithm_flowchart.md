# アルゴリズムフロー図

[← README](../README.md) | [← 技術手法](methodology.md) | [APIリファレンス →](api_reference.md) | [開発ガイド →](development.md)

---

## 1. システム全体のフロー

```mermaid
graph TB
    Start([開始]) --> Input[画像入力]
    Input --> Preprocess[前処理]
    
    Preprocess --> FaceDetect{顔検出}
    FaceDetect -->|成功| EyeDetect[眼検出]
    FaceDetect -->|失敗| DirectEye[直接眼検出]
    
    EyeDetect --> ScleraExtract[結膜抽出]
    DirectEye --> ScleraExtract
    
    ScleraExtract --> ColorAnalysis[色解析]
    
    ColorAnalysis --> RGB[RGB特徴]
    ColorAnalysis --> HSV[HSV特徴]
    ColorAnalysis --> LAB[LAB特徴]
    
    RGB --> FeatureCombine[特徴統合]
    HSV --> FeatureCombine
    LAB --> FeatureCombine
    
    FeatureCombine --> Estimation[ビリルビン推定]
    Estimation --> RiskAssess[リスク評価]
    
    RiskAssess --> Output[結果出力]
    Output --> End([終了])
    
    %% エラーパス
    ScleraExtract -->|失敗| Error[エラー処理]
    Error --> End
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Error fill:#FF6B6B
```

## 2. 画像前処理の詳細フロー

```mermaid
graph LR
    Input[入力画像] --> SizeCheck{サイズチェック}
    
    SizeCheck -->|>1024px| Resize[リサイズ]
    SizeCheck -->|≤1024px| ColorConvert
    
    Resize --> ColorConvert[LAB変換]
    ColorConvert --> SplitChannels[チャンネル分離]
    
    SplitChannels --> L[L channel]
    SplitChannels --> A[a channel]
    SplitChannels --> B[b channel]
    
    L --> CLAHE[CLAHE適用]
    CLAHE --> MergeChannels[チャンネル統合]
    A --> MergeChannels
    B --> MergeChannels
    
    MergeChannels --> BGR[BGR変換]
    BGR --> Denoise[ノイズ除去]
    Denoise --> Output[前処理済み画像]
    
    style Input fill:#E6F3FF
    style Output fill:#FFE6E6
```

## 3. 眼球・結膜検出の詳細フロー

```mermaid
graph TB
    Image[前処理済み画像] --> Gray[グレースケール変換]
    
    Gray --> FaceCascade[Haar Cascade<br/>顔検出]
    FaceCascade --> Faces{顔検出?}
    
    Faces -->|あり| FaceLoop[各顔で処理]
    Faces -->|なし| DirectEyeDetect[画像全体で<br/>眼検出]
    
    FaceLoop --> ROI[ROI抽出]
    ROI --> EyeCascade[Haar Cascade<br/>眼検出]
    
    EyeCascade --> EyeLoop[各眼で処理]
    DirectEyeDetect --> EyeLoop
    
    EyeLoop --> Quality[品質評価]
    Quality --> Score[信頼度スコア計算]
    
    Score --> Best{最良の眼?}
    Best -->|更新| UpdateBest[最良眼を更新]
    Best -->|そのまま| Continue[次の眼へ]
    
    UpdateBest --> ScleraExtract[結膜抽出]
    Continue --> EyeLoop
    
    %% 結膜抽出の詳細
    ScleraExtract --> HSVConvert[HSV変換]
    HSVConvert --> ColorRange[色範囲定義<br/>H: 0-180<br/>S: 0-30<br/>V: 100-255]
    
    ColorRange --> Mask[マスク生成]
    Mask --> Morphology[モルフォロジー処理]
    Morphology --> Apply[マスク適用]
    Apply --> Result[結膜領域]
    
    style Image fill:#E6F3FF
    style Result fill:#90EE90
```

## 4. 色特徴抽出の詳細フロー

```mermaid
graph TB
    Sclera[結膜領域] --> Multi[マルチ色空間変換]
    
    Multi --> RGB[RGB色空間]
    Multi --> HSV[HSV色空間]
    Multi --> LAB[LAB色空間]
    
    %% RGB処理
    RGB --> RGBStats[統計量計算]
    RGBStats --> RGBMean[平均値 R,G,B]
    RGBStats --> RGBStd[標準偏差 R,G,B]
    RGBMean --> RBRatio[R/B比率計算]
    
    %% HSV処理
    HSV --> HSVMask[黄色マスク生成<br/>H: 20-40°]
    HSVMask --> YellowRatio[黄色ピクセル比率]
    HSV --> SatMean[彩度平均値]
    
    %% LAB処理
    LAB --> LABStats[統計量計算]
    LABStats --> LABMean[平均値 L,a,b]
    LABMean --> Yellowness[黄色度計算<br/>b* - 128]
    
    %% 統合
    RBRatio --> Combine[特徴統合]
    YellowRatio --> Combine
    Yellowness --> Combine
    SatMean --> Combine
    
    Combine --> YellowIndex[複合黄色度指標]
    YellowIndex --> Features[特徴ベクトル]
    
    style Sclera fill:#E6F3FF
    style Features fill:#FFE6E6
```

## 5. ビリルビン推定アルゴリズム

```mermaid
graph LR
    Features[特徴ベクトル] --> Linear[線形結合]
    
    Linear --> Formula["推定式<br/>B = sum(βi * fi) + β0"]
    
    Formula --> Components[構成要素]
    
    Components --> F1["f1: HSV黄色比率<br/>β1 = 50.0"]
    Components --> F2["f2: RGB R/B比率<br/>β2 = 25.0"]
    Components --> F3["f3: LAB黄色度<br/>β3 = 0.8"]
    Components --> F4["f4: 複合指標<br/>β4 = 30.0"]
    Components --> F5["f5: 彩度平均<br/>β5 = 10.0"]
    Components --> F0["β0: 切片 = -5.0"]
    
    F1 --> Sum[加重和計算]
    F2 --> Sum
    F3 --> Sum
    F4 --> Sum
    F5 --> Sum
    F0 --> Sum
    
    Sum --> Clamp[値域制限<br/>0-30 mg/dL]
    Clamp --> Result[ビリルビン値]
    
    Result --> Risk{リスク評価}
    Risk -->|< 3| Low[低リスク]
    Risk -->|3-12| Moderate[中リスク]
    Risk -->|12-20| High[高リスク]
    Risk -->|> 20| Critical[重大リスク]
    
    style Features fill:#E6F3FF
    style Result fill:#90EE90
    style Critical fill:#FF6B6B
```

## 6. 色較正アルゴリズム

```mermaid
graph TB
    Image[較正カード画像] --> EdgeDetect[エッジ検出<br/>Canny]
    
    EdgeDetect --> Contours[輪郭検出]
    Contours --> RectCheck{矩形チェック}
    
    RectCheck -->|4頂点| AspectRatio{アスペクト比<br/>1.2-2.5?}
    RectCheck -->|その他| NextContour[次の輪郭]
    
    AspectRatio -->|OK| Extract[カード領域抽出]
    AspectRatio -->|NG| NextContour
    
    Extract --> Grid["グリッド分割<br/>2x3 or 3x2"]
    Grid --> Patches[色パッチ抽出]
    
    Patches --> Loop{各パッチ}
    Loop --> Center[中心領域抽出]
    Center --> AvgColor[平均色計算]
    AvgColor --> Store[色値保存]
    Store --> Loop
    
    Store --> Matrix[変換行列計算]
    
    Matrix --> LSQ["最小二乗法<br/>M = (X'X)^-1 X'y"]
    
    LSQ --> Transform["3x4変換行列"]
    
    %% 参照色
    Reference[参照色<br/>白,灰,黒<br/>赤,緑,青] --> LSQ
    
    Transform --> Apply{較正適用}
    Apply --> Corrected[補正済み画像]
    
    style Image fill:#E6F3FF
    style Corrected fill:#90EE90
```

## 7. 信頼度評価フロー

```mermaid
graph TB
    EyeImage[眼領域画像] --> Checks[品質チェック]
    
    Checks --> Size{サイズ}
    Size -->|"< 20x20px"| Low1[低信頼度 0.1]
    Size -->|">= 20x20px"| Brightness
    
    Checks --> Brightness{明るさ}
    Brightness -->|30-220| OK1[適正]
    Brightness -->|その他| Low2[低信頼度 0.3]
    
    Checks --> Contrast{コントラスト}
    Contrast -->|"σ > 20"| OK2[適正]
    Contrast -->|"σ <= 20"| Low3[低信頼度 0.4]
    
    Checks --> WhitePixels{白色ピクセル}
    WhitePixels -->|"> 5%"| OK3[適正]
    WhitePixels -->|"<= 5%"| Low4[低信頼度 0.5]
    
    OK1 --> Calculate[総合計算]
    OK2 --> Calculate
    OK3 --> Calculate
    
    Calculate --> Formula2["信頼度 = min(1.0, 0.3 + 白色比率*2 + σ/100)"]
    
    Formula2 --> Confidence[最終信頼度]
    
    Low1 --> Confidence
    Low2 --> Confidence
    Low3 --> Confidence
    Low4 --> Confidence
    
    style EyeImage fill:#E6F3FF
    style Confidence fill:#FFE6E6
```

## 8. エラー処理とフォールバック

```mermaid
graph TB
    Process[処理中] --> Error{エラー種別}
    
    Error -->|顔検出失敗| Fallback1[直接眼検出試行]
    Error -->|眼検出失敗| Fallback2[エラーレスポンス]
    Error -->|結膜抽出失敗| Fallback3[眼領域全体を使用]
    Error -->|画像読込失敗| Fallback4[エラーレスポンス]
    
    Fallback1 --> Retry{再試行}
    Retry -->|成功| Continue[処理継続]
    Retry -->|失敗| ErrorResponse[エラーレスポンス]
    
    Fallback2 --> ErrorResponse
    Fallback3 --> ReducedAccuracy[精度低下警告付き<br/>処理継続]
    Fallback4 --> ErrorResponse
    
    ErrorResponse --> Response[レスポンス生成<br/>success: false<br/>error: メッセージ]
    
    style Error fill:#FF6B6B
    style ErrorResponse fill:#FFB6C1
    style Continue fill:#90EE90
```

## フロー図の見方

- **四角形**: 処理ステップ
- **ひし形**: 判断・分岐
- **角丸四角形**: 開始・終了
- **色の意味**:
  - 🟦 青系: 入力データ
  - 🟥 赤系: 出力・エラー
  - 🟩 緑系: 成功・正常処理
  - ⬜ 白: 通常処理

これらのフロー図により、各アルゴリズムの詳細な処理の流れと判断基準が明確になります。