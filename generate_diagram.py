import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

def draw_box(x, y, width, height, text, facecolor):
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.2",
                                 facecolor=facecolor, edgecolor='black',
                                 linewidth=2)
    ax.add_patch(box)
    
    # Check if text is dark
    text_color = 'white' if facecolor in ['#2c3e50', '#2980b9', '#8e44ad', '#27ae60', '#d35400'] else 'black'
    
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=12, wrap=True, family='sans-serif', fontweight='bold', color=text_color)
    
    return {
        'top': (x + width/2, y + height),
        'bottom': (x + width/2, y),
        'left': (x, y + height/2),
        'right': (x + width, y + height/2)
    }

def draw_arrow(start, end, rad=0.0, style="->"):
    ax.annotate("",
                xy=end, xycoords='data',
                xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle=style, color='black', lw=2, connectionstyle=f"arc3,rad={rad}"))

# 1. Main Data
b_data = draw_box(5, 9, 4, 1, "Empirical Neural Data\n(Spikes, Behavior, Position)", '#2c3e50')

# 2. Real Modeling Branch
b_real_head = draw_box(2, 6.5, 3, 1, "Real Data Analysis", '#bdc3c7')
b_step12 = draw_box(1.5, 4, 4, 1.5, "Step 1: Fit Mean-Covariance Model\nStep 2: Interpret via SHAP Attributions", '#2980b9')

# 3. Synthetic Modeling Branch
b_syn_head = draw_box(8, 6.5, 4, 1, "Synthetic Verification Pipeline\n(Ground-Truth Validation)", '#bdc3c7')
b_mean_val = draw_box(6, 3, 4, 2, "Mean Model Validation\n--------------------\nStep 3: 2-Var GLM Injection\nStep 4: 3-Var GLM Injection\nStep 5: Dense/Sparse Check\nStep 6: Empirical Cov. Noise", '#27ae60')
b_cov_val = draw_box(10.5, 3, 3, 2, "Covariance Model Validation\n--------------------\nStep 7: Kronecker Structure\nInjection & Latent Recovery", '#d35400')

# Arrows
draw_arrow(b_data['bottom'], b_real_head['top'], rad=0.1)
draw_arrow(b_data['bottom'], b_syn_head['top'], rad=-0.1)

draw_arrow(b_real_head['bottom'], b_step12['top'])
draw_arrow(b_syn_head['bottom'], b_mean_val['top'], rad=0.1)
draw_arrow(b_syn_head['bottom'], b_cov_val['top'], rad=-0.1)

# Extracted parameters to synthetic
ax.annotate("", xy=b_syn_head['left'], xytext=b_real_head['right'],
            arrowprops=dict(arrowstyle="->", ls="--", color='gray', lw=2))
ax.text(5.5, 7.3, "Extract Empirical Parameters\nfor Synthetic Injection", ha='center', va='bottom', fontsize=10, color='gray', style='italic')

plt.title("Parallel Mean-Covariance Model: End-to-End Methodology", fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.savefig('assets/methodology_diagram.png', dpi=300, bbox_inches='tight')
print("Saved methodology_diagram.png")
