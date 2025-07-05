#!/usr/bin/env python3
"""
Create algorithm flow diagrams as images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

def create_main_algorithm_flow():
    """Create the main algorithm flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Define colors
    start_color = '#90EE90'
    end_color = '#FFB6C1'
    process_color = '#E6F3FF'
    decision_color = '#FFE6CC'
    error_color = '#FF6B6B'
    
    # Helper function to add a box
    def add_box(x, y, w, h, text, color='#E6F3FF', style='round'):
        if style == 'round':
            box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                               boxstyle="round,pad=0.1",
                               facecolor=color,
                               edgecolor='black',
                               linewidth=1.5)
        elif style == 'diamond':
            # Create diamond shape
            points = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
            box = patches.Polygon(points, closed=True, 
                                facecolor=color,
                                edgecolor='black',
                                linewidth=1.5)
        else:
            box = Rectangle((x-w/2, y-h/2), w, h,
                          facecolor=color,
                          edgecolor='black',
                          linewidth=1.5)
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center', fontsize=10, 
                fontweight='bold' if style == 'round' else 'normal')
        
    # Helper function to add arrow
    def add_arrow(x1, y1, x2, y2, text=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', 
                              mutation_scale=20,
                              color='black',
                              linewidth=1.5)
        ax.add_patch(arrow)
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=8, style='italic')
    
    # Main flow
    add_box(5, 19, 2, 0.8, '開始', start_color, 'round')
    add_arrow(5, 18.5, 5, 17.5)
    
    add_box(5, 17, 2.5, 0.8, '画像入力', process_color)
    add_arrow(5, 16.5, 5, 15.5)
    
    add_box(5, 15, 2.5, 0.8, '前処理', process_color)
    add_arrow(5, 14.5, 5, 13.5)
    
    add_box(5, 13, 2.5, 1.2, '顔検出', decision_color, 'diamond')
    add_arrow(5, 12, 5, 11, '成功')
    add_arrow(6.5, 13, 8, 13, '失敗')
    
    add_box(5, 10.5, 2.5, 0.8, '眼検出', process_color)
    add_box(8, 13, 2.5, 0.8, '直接眼検出', process_color)
    add_arrow(8, 12.5, 8, 10.5)
    add_arrow(8, 10, 6.5, 9.5)
    add_arrow(5, 10, 5, 9)
    
    add_box(5, 8.5, 2.5, 0.8, '結膜抽出', process_color)
    add_arrow(5, 8, 5, 7)
    
    add_box(5, 6.5, 2.5, 0.8, '色解析', process_color)
    
    # Color space branches
    add_arrow(5, 6, 2, 5)
    add_arrow(5, 6, 5, 5)
    add_arrow(5, 6, 8, 5)
    
    add_box(2, 4.5, 1.5, 0.6, 'RGB', process_color)
    add_box(5, 4.5, 1.5, 0.6, 'HSV', process_color)
    add_box(8, 4.5, 1.5, 0.6, 'LAB', process_color)
    
    add_arrow(2, 4, 5, 3.5)
    add_arrow(5, 4, 5, 3.5)
    add_arrow(8, 4, 5, 3.5)
    
    add_box(5, 3, 2.5, 0.8, '特徴統合', process_color)
    add_arrow(5, 2.5, 5, 1.5)
    
    add_box(5, 1, 3, 0.8, 'ビリルビン推定', process_color)
    add_arrow(5, 0.5, 5, -0.5)
    
    add_box(5, -1, 2.5, 0.8, 'リスク評価', process_color)
    add_arrow(5, -1.5, 5, -2.5)
    
    add_box(5, -3, 2.5, 0.8, '結果出力', process_color)
    add_arrow(5, -3.5, 5, -4.5)
    
    add_box(5, -5, 2, 0.8, '終了', end_color, 'round')
    
    # Error path
    add_arrow(3.5, 8.5, 1, 8.5)
    add_box(0.5, 8.5, 1.5, 0.8, 'エラー', error_color)
    add_arrow(0.5, 8, 0.5, -4.5)
    add_arrow(0.5, -4.5, 4, -4.5)
    
    # Title
    ax.text(5, 20.5, 'ビリルビン検出システム アルゴリズムフロー', 
            fontsize=16, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig('docs/images/main_algorithm_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_color_analysis_flow():
    """Create color analysis flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    input_color = '#E6F3FF'
    process_color = '#FFE6E6'
    output_color = '#90EE90'
    
    def add_box(x, y, w, h, text, color='#E6F3FF'):
        box = Rectangle((x-w/2, y-h/2), w, h,
                      facecolor=color,
                      edgecolor='black',
                      linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
        
    def add_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', 
                              mutation_scale=15,
                              color='black')
        ax.add_patch(arrow)
    
    # Main flow
    add_box(7, 9, 2.5, 0.8, '結膜領域', input_color)
    add_arrow(7, 8.5, 7, 7.5)
    
    add_box(7, 7, 3, 0.8, 'マルチ色空間変換', process_color)
    
    # Branches
    add_arrow(5, 6.5, 3, 5.5)
    add_arrow(7, 6.5, 7, 5.5)
    add_arrow(9, 6.5, 11, 5.5)
    
    # RGB processing
    add_box(3, 5, 2, 0.8, 'RGB色空間', process_color)
    add_arrow(3, 4.5, 3, 3.5)
    add_box(3, 3, 2, 0.8, '統計量計算', process_color)
    add_arrow(3, 2.5, 3, 1.5)
    add_box(3, 1, 2, 0.8, 'R/B比率', process_color)
    
    # HSV processing
    add_box(7, 5, 2, 0.8, 'HSV色空間', process_color)
    add_arrow(7, 4.5, 7, 3.5)
    add_box(7, 3, 2.5, 0.8, '黄色マスク\nH: 20-40°', process_color)
    add_arrow(7, 2.5, 7, 1.5)
    add_box(7, 1, 2, 0.8, '黄色比率', process_color)
    
    # LAB processing
    add_box(11, 5, 2, 0.8, 'LAB色空間', process_color)
    add_arrow(11, 4.5, 11, 3.5)
    add_box(11, 3, 2, 0.8, 'b*チャンネル', process_color)
    add_arrow(11, 2.5, 11, 1.5)
    add_box(11, 1, 2, 0.8, '黄色度', process_color)
    
    # Combine
    add_arrow(3, 0.5, 7, -0.5)
    add_arrow(7, 0.5, 7, -0.5)
    add_arrow(11, 0.5, 7, -0.5)
    
    add_box(7, -1, 3, 0.8, '特徴ベクトル生成', output_color)
    
    # Title
    ax.text(7, 10, '色特徴抽出アルゴリズム', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Feature list
    feature_text = "抽出される特徴:\n" \
                  "• RGB平均/標準偏差\n" \
                  "• R/B比率\n" \
                  "• HSV黄色ピクセル比率\n" \
                  "• 彩度平均\n" \
                  "• LAB黄色度\n" \
                  "• 複合黄色度指標"
    ax.text(13, 3, feature_text, fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('docs/images/color_analysis_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_bilirubin_estimation_flow():
    """Create bilirubin estimation flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#E6F3FF'
    process_color = '#FFE6E6'
    output_color = '#90EE90'
    risk_colors = {'低': '#90EE90', '中': '#FFD700', '高': '#FFA500', '重大': '#FF6B6B'}
    
    def add_box(x, y, w, h, text, color='#E6F3FF'):
        box = Rectangle((x-w/2, y-h/2), w, h,
                      facecolor=color,
                      edgecolor='black',
                      linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9)
    
    def add_arrow(x1, y1, x2, y2, text=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', 
                              mutation_scale=15,
                              color='black')
        ax.add_patch(arrow)
        if text:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.5, mid_y, text, fontsize=8)
    
    # Input features
    features = [
        ('HSV黄色比率\nβ₁ = 50.0', 2),
        ('RGB R/B比率\nβ₂ = 25.0', 4),
        ('LAB黄色度\nβ₃ = 0.8', 6),
        ('複合指標\nβ₄ = 30.0', 8),
        ('彩度平均\nβ₅ = 10.0', 10)
    ]
    
    for i, (feature, x) in enumerate(features):
        add_box(x, 8, 1.8, 1, feature, input_color)
        add_arrow(x, 7.5, 6, 6)
    
    # Linear combination
    add_box(6, 5.5, 4, 1, '線形結合\nB = Σ(βᵢ × fᵢ) + β₀', process_color)
    add_box(6, 4, 2, 0.8, 'β₀ = -5.0', input_color)
    add_arrow(6, 5, 6, 3.5)
    
    # Clamping
    add_box(6, 3, 3, 0.8, '値域制限\n0-30 mg/dL', process_color)
    add_arrow(6, 2.5, 6, 1.5)
    
    # Output
    add_box(6, 1, 2.5, 0.8, 'ビリルビン値', output_color)
    
    # Risk assessment
    add_arrow(6, 0.5, 2, -0.5, '< 3')
    add_arrow(6, 0.5, 4.5, -0.5, '3-12')
    add_arrow(6, 0.5, 7.5, -0.5, '12-20')
    add_arrow(6, 0.5, 10, -0.5, '> 20')
    
    add_box(2, -1, 1.2, 0.6, '低', risk_colors['低'])
    add_box(4.5, -1, 1.2, 0.6, '中', risk_colors['中'])
    add_box(7.5, -1, 1.2, 0.6, '高', risk_colors['高'])
    add_box(10, -1, 1.2, 0.6, '重大', risk_colors['重大'])
    
    # Title
    ax.text(6, 9.5, 'ビリルビン推定アルゴリズム', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Formula
    formula_text = "推定式:\nB = 50.0×f₁ + 25.0×f₂ + 0.8×f₃ + 30.0×f₄ + 10.0×f₅ - 5.0"
    ax.text(11, 5.5, formula_text, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    plt.tight_layout()
    plt.savefig('docs/images/bilirubin_estimation_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all algorithm diagrams"""
    import os
    os.makedirs('docs/images', exist_ok=True)
    
    print("Creating algorithm flow diagrams...")
    
    create_main_algorithm_flow()
    print("  ✓ Main algorithm flow")
    
    create_color_analysis_flow()
    print("  ✓ Color analysis flow")
    
    create_bilirubin_estimation_flow()
    print("  ✓ Bilirubin estimation flow")
    
    print("\nDiagrams created successfully in docs/images/")

if __name__ == "__main__":
    main()