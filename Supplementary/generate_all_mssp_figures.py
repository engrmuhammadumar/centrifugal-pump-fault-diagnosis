# ============================================================================
# ADD THIS CELL AT THE END OF YOUR CODE TO GENERATE ALL MSSP FIGURES
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure publication style
plt.style.use('default')
sns.set_style("white")

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 0,  # NO TITLES for publication
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'axes.linewidth': 2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 1.0,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
})

# Create output directory
FIG_DIR = r"E:\4 Paper\New Implementation_final\MSSP_Publication_Figures"
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300  # Publication quality

print("\n" + "="*80)
print(" " * 20 + "GENERATING MSSP PUBLICATION FIGURES")
print("="*80)
print(f"Output Directory: {FIG_DIR}")
print(f"Resolution: {DPI} DPI | Style: White Background, Bold Axes, No Titles")
print("="*80 + "\n")

# Helper function
def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f"{FIG_DIR}/{name}.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  ✓ {name}")

def bold_ticks(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

# ============================================================================
# FIGURE 1: Training Loss Curve
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
epochs = np.arange(1, len(history['train_loss']) + 1)

ax.plot(epochs, history['train_loss'], 'o-', color='#1f77b4', 
        label='Training', markersize=5, markevery=max(1, len(epochs)//15))
ax.plot(epochs, history['val_loss'], 's-', color='#d62728', 
        label='Validation', markersize=5, markevery=max(1, len(epochs)//15))

best_epoch = np.argmin(history['val_loss']) + 1
ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7, linewidth=2,
          label=f'Best Epoch ({best_epoch})')

ax.set_xlabel('Epoch', fontweight='bold', fontsize=16)
ax.set_ylabel('Loss', fontweight='bold', fontsize=16)
ax.legend(frameon=True, loc='best', framealpha=0.9, fontsize=13)
ax.grid(True, alpha=0.3)
bold_ticks(ax)

save_fig('Fig01_Training_Loss_Curve')

# ============================================================================
# FIGURE 2: Training RMSE (Wear & RUL)
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
epochs = np.arange(1, len(history['train_loss']) + 1)

# Wear RMSE
axes[0].plot(epochs, history['train_rmse_wear'], 'o-', color='#1f77b4',
             label='Training', markevery=max(1, len(epochs)//15))
axes[0].plot(epochs, history['val_rmse_wear'], 's-', color='#d62728',
             label='Validation', markevery=max(1, len(epochs)//15))
axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=16)
axes[0].set_ylabel('Wear RMSE (0.001 mm)', fontweight='bold', fontsize=16)
axes[0].legend(frameon=True, loc='best', fontsize=13)
axes[0].grid(True, alpha=0.3)
bold_ticks(axes[0])

# RUL RMSE
axes[1].plot(epochs, history['train_rmse_rul'], 'o-', color='#1f77b4',
             label='Training', markevery=max(1, len(epochs)//15))
axes[1].plot(epochs, history['val_rmse_rul'], 's-', color='#d62728',
             label='Validation', markevery=max(1, len(epochs)//15))
axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=16)
axes[1].set_ylabel('RUL RMSE', fontweight='bold', fontsize=16)
axes[1].legend(frameon=True, loc='best', fontsize=13)
axes[1].grid(True, alpha=0.3)
bold_ticks(axes[1])

save_fig('Fig02_Training_RMSE_Metrics')

# ============================================================================
# FIGURE 3: Training R² Scores
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
epochs = np.arange(1, len(history['train_loss']) + 1)

# Wear R²
axes[0].plot(epochs, history['train_r2_wear'], 'o-', color='#1f77b4',
             label='Training', markevery=max(1, len(epochs)//15))
axes[0].plot(epochs, history['val_r2_wear'], 's-', color='#d62728',
             label='Validation', markevery=max(1, len(epochs)//15))
axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
axes[0].set_xlabel('Epoch', fontweight='bold', fontsize=16)
axes[0].set_ylabel('Wear R²', fontweight='bold', fontsize=16)
axes[0].set_ylim([-0.1, 1.05])
axes[0].legend(frameon=True, loc='best', fontsize=13)
axes[0].grid(True, alpha=0.3)
bold_ticks(axes[0])

# RUL R²
axes[1].plot(epochs, history['train_r2_rul'], 'o-', color='#1f77b4',
             label='Training', markevery=max(1, len(epochs)//15))
axes[1].plot(epochs, history['val_r2_rul'], 's-', color='#d62728',
             label='Validation', markevery=max(1, len(epochs)//15))
axes[1].axhline(1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
axes[1].set_xlabel('Epoch', fontweight='bold', fontsize=16)
axes[1].set_ylabel('RUL R²', fontweight='bold', fontsize=16)
axes[1].set_ylim([-0.1, 1.05])
axes[1].legend(frameon=True, loc='best', fontsize=13)
axes[1].grid(True, alpha=0.3)
bold_ticks(axes[1])

save_fig('Fig03_Training_R2_Scores')

# ============================================================================
# FIGURE 4: Learning Rate Schedule
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
epochs = np.arange(1, len(history['lr']) + 1)

ax.plot(epochs, history['lr'], 'o-', color='#9467bd', markersize=6)
ax.set_xlabel('Epoch', fontweight='bold', fontsize=16)
ax.set_ylabel('Learning Rate', fontweight='bold', fontsize=16)
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')
bold_ticks(ax)

save_fig('Fig04_Learning_Rate_Schedule')

# ============================================================================
# FIGURE 5: Validation Scatter - Wear
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

mask = val_df['wear_true'].notna()
y_true = val_df.loc[mask, 'wear_true'].values
y_pred = val_df.loc[mask, 'wear_pred'].values

ax.scatter(y_true, y_pred, alpha=0.6, s=80, color='#1f77b4', 
           edgecolors='white', linewidth=1.0)

lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=3, alpha=0.8, label='Perfect Prediction')

rmse = np.sqrt(np.mean((y_pred - y_true)**2))
r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
mae = np.mean(np.abs(y_pred - y_true))

textstr = f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props, fontweight='bold')

ax.set_xlabel('True Wear (0.001 mm)', fontweight='bold', fontsize=16)
ax.set_ylabel('Predicted Wear (0.001 mm)', fontweight='bold', fontsize=16)
ax.legend(frameon=True, loc='lower right', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
bold_ticks(ax)

save_fig('Fig05_Validation_Scatter_Wear')

# ============================================================================
# FIGURE 6: Validation Scatter - RUL
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 8))

mask = val_df['rul_true'].notna()
y_true = val_df.loc[mask, 'rul_true'].values
y_pred = val_df.loc[mask, 'rul_pred'].values

ax.scatter(y_true, y_pred, alpha=0.6, s=80, color='#d62728', 
           edgecolors='white', linewidth=1.0)

lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=3, alpha=0.8, label='Perfect Prediction')

rmse = np.sqrt(np.mean((y_pred - y_true)**2))
r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
mae = np.mean(np.abs(y_pred - y_true))

textstr = f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props, fontweight='bold')

ax.set_xlabel('True RUL', fontweight='bold', fontsize=16)
ax.set_ylabel('Predicted RUL', fontweight='bold', fontsize=16)
ax.legend(frameon=True, loc='lower right', fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
bold_ticks(ax)

save_fig('Fig06_Validation_Scatter_RUL')

# ============================================================================
# FIGURE 7: Validation Wear Trajectories (Individual Cutters)
# ============================================================================

cutters = sorted(val_df['cutter'].unique())
n = len(cutters)
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
if n == 1:
    axes = [axes]

for idx, cutter in enumerate(cutters):
    ax = axes[idx]
    df_c = val_df[val_df['cutter'] == cutter].sort_values('cut_number')
    
    x = df_c['cut_number'].values
    y_true = df_c['wear_true'].values
    y_pred = df_c['wear_pred'].values
    y_std = df_c['wear_std'].values
    
    ax.plot(x, y_true, 'k-', linewidth=3.5, label='Ground Truth', zorder=3, 
            marker='o', markersize=7)
    ax.plot(x, y_pred, '-', linewidth=2.5, color=colors[idx % len(colors)], 
            label='Predicted', marker='s', markersize=6, zorder=2)
    ax.fill_between(x, np.maximum(y_pred - 2*y_std, 0), y_pred + 2*y_std,
                    alpha=0.25, color=colors[idx % len(colors)], label='±2σ', zorder=1)
    
    ax.set_xlabel('Cut Number', fontweight='bold', fontsize=16)
    if idx == 0:
        ax.set_ylabel('Wear (0.001 mm)', fontweight='bold', fontsize=16)
    ax.legend(frameon=True, loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.95, 0.05, cutter.upper(), transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                     linewidth=2, alpha=0.9))
    bold_ticks(ax)

save_fig('Fig07_Validation_Wear_Trajectories')

# ============================================================================
# FIGURE 8: Validation RUL Trajectories (Individual Cutters)
# ============================================================================

colors_rul = ['#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
if n == 1:
    axes = [axes]

for idx, cutter in enumerate(cutters):
    ax = axes[idx]
    df_c = val_df[val_df['cutter'] == cutter].sort_values('cut_number')
    
    x = df_c['cut_number'].values
    y_true = df_c['rul_true'].values
    y_pred = df_c['rul_pred'].values
    y_std = df_c['rul_std'].values
    
    ax.plot(x, y_true, 'k-', linewidth=3.5, label='Ground Truth', zorder=3, 
            marker='o', markersize=7)
    ax.plot(x, y_pred, '-', linewidth=2.5, color=colors_rul[idx % len(colors_rul)], 
            label='Predicted', marker='s', markersize=6, zorder=2)
    ax.fill_between(x, np.maximum(y_pred - 2*y_std, 0), y_pred + 2*y_std,
                    alpha=0.25, color=colors_rul[idx % len(colors_rul)], label='±2σ', zorder=1)
    ax.axhline(0, color='darkred', linestyle='--', linewidth=2.5, alpha=0.8, label='Failure')
    
    ax.set_xlabel('Cut Number', fontweight='bold', fontsize=16)
    if idx == 0:
        ax.set_ylabel('RUL', fontweight='bold', fontsize=16)
    ax.legend(frameon=True, loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.95, 0.95, cutter.upper(), transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                     linewidth=2, alpha=0.9))
    bold_ticks(ax)

save_fig('Fig08_Validation_RUL_Trajectories')

# ============================================================================
# FIGURE 9: Validation Phase Space (Wear vs RUL)
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

cutters = sorted(val_df['cutter'].unique())

for idx, cutter in enumerate(cutters):
    df_c = val_df[val_df['cutter'] == cutter].sort_values('cut_number')
    
    wear = df_c['wear_true'].values
    rul = df_c['rul_true'].values
    cuts = df_c['cut_number'].values
    
    scatter = ax.scatter(wear, rul, s=100, alpha=0.7, c=cuts, cmap='viridis',
                        edgecolors='black', linewidth=1.5, label=cutter.upper())

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Cut Number', fontweight='bold', fontsize=14)
cbar.ax.tick_params(labelsize=12)

ax.set_xlabel('Wear (0.001 mm)', fontweight='bold', fontsize=16)
ax.set_ylabel('RUL', fontweight='bold', fontsize=16)
ax.legend(frameon=True, loc='upper right', fontsize=13)
ax.grid(True, alpha=0.3)
bold_ticks(ax)

save_fig('Fig09_Validation_Phase_Space')

# ============================================================================
# FIGURE 10: Test Wear Predictions
# ============================================================================

test_cutters = sorted(test_seq_predictions['cutter'].unique())
n_test = len(test_cutters)

fig, axes = plt.subplots(1, n_test, figsize=(7*n_test, 6))
if n_test == 1:
    axes = [axes]

colors_test = ['#1f77b4', '#2ca02c', '#ff7f0e']

for idx, cutter in enumerate(test_cutters):
    ax = axes[idx]
    df_c = test_seq_predictions[test_seq_predictions['cutter'] == cutter].sort_values('cut_number')
    
    x = df_c['cut_number'].values
    y_pred = df_c['wear_pred'].values
    y_std = df_c['wear_std'].values
    eol = df_c['estimated_eol'].iloc[0]
    
    ax.plot(x, y_pred, 'o-', linewidth=2.5, color=colors_test[idx % len(colors_test)],
            label='Predicted Wear', markersize=6)
    ax.fill_between(x, np.maximum(y_pred - 2*y_std, 0), y_pred + 2*y_std,
                    alpha=0.25, color=colors_test[idx % len(colors_test)], label='±2σ')
    ax.axhline(eol, color='darkred', linestyle='--', linewidth=2.5, alpha=0.8,
              label=f'Est. EOL: {eol:.1f}')
    
    ax.set_xlabel('Cut Number', fontweight='bold', fontsize=16)
    if idx == 0:
        ax.set_ylabel('Wear (0.001 mm)', fontweight='bold', fontsize=16)
    ax.legend(frameon=True, loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.95, 0.05, cutter.upper(), transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                     linewidth=2, alpha=0.9))
    bold_ticks(ax)

save_fig('Fig10_Test_Wear_Predictions')

# ============================================================================
# FIGURE 11: Test RUL Predictions
# ============================================================================

colors_test_rul = ['#d62728', '#9467bd', '#8c564b']

fig, axes = plt.subplots(1, n_test, figsize=(7*n_test, 6))
if n_test == 1:
    axes = [axes]

for idx, cutter in enumerate(test_cutters):
    ax = axes[idx]
    df_c = test_seq_predictions[test_seq_predictions['cutter'] == cutter].sort_values('cut_number')
    
    x = df_c['cut_number'].values
    y_pred = df_c['rul_pred'].values
    y_std = df_c['rul_std'].values
    
    ax.plot(x, y_pred, 's-', linewidth=2.5, color=colors_test_rul[idx % len(colors_test_rul)],
            label='Predicted RUL', markersize=6)
    ax.fill_between(x, np.maximum(y_pred - 2*y_std, 0), y_pred + 2*y_std,
                    alpha=0.25, color=colors_test_rul[idx % len(colors_test_rul)], label='±2σ')
    ax.axhline(0, color='darkred', linestyle='--', linewidth=2.5, alpha=0.8,
              label='Failure Threshold')
    
    ax.set_xlabel('Cut Number', fontweight='bold', fontsize=16)
    if idx == 0:
        ax.set_ylabel('RUL', fontweight='bold', fontsize=16)
    ax.legend(frameon=True, loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    ax.text(0.95, 0.95, cutter.upper(), transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='right', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', 
                     linewidth=2, alpha=0.9))
    bold_ticks(ax)

save_fig('Fig11_Test_RUL_Predictions')

# ============================================================================
# FIGURE 12: Residual Analysis - Wear
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

mask = val_df['wear_true'].notna()
y_true = val_df.loc[mask, 'wear_true'].values
y_pred = val_df.loc[mask, 'wear_pred'].values
residuals = y_pred - y_true

# Residual plot
axes[0].scatter(y_pred, residuals, alpha=0.6, s=80, color='#1f77b4',
               edgecolors='white', linewidth=1.0)
axes[0].axhline(0, color='r', linestyle='--', linewidth=3)
axes[0].set_xlabel('Predicted Wear (0.001 mm)', fontweight='bold', fontsize=16)
axes[0].set_ylabel('Residuals', fontweight='bold', fontsize=16)
axes[0].grid(True, alpha=0.3)
bold_ticks(axes[0])

# Histogram
axes[1].hist(residuals, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].axvline(0, color='r', linestyle='--', linewidth=3)
axes[1].axvline(residuals.mean(), color='orange', linestyle=':', linewidth=3,
               label=f'Mean: {residuals.mean():.2f}')
axes[1].set_xlabel('Residuals', fontweight='bold', fontsize=16)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=16)
axes[1].legend(frameon=True, fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')
bold_ticks(axes[1])

save_fig('Fig12_Residual_Analysis_Wear')

# ============================================================================
# FIGURE 13: Residual Analysis - RUL
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

mask = val_df['rul_true'].notna()
y_true = val_df.loc[mask, 'rul_true'].values
y_pred = val_df.loc[mask, 'rul_pred'].values
residuals = y_pred - y_true

# Residual plot
axes[0].scatter(y_pred, residuals, alpha=0.6, s=80, color='#d62728',
               edgecolors='white', linewidth=1.0)
axes[0].axhline(0, color='r', linestyle='--', linewidth=3)
axes[0].set_xlabel('Predicted RUL', fontweight='bold', fontsize=16)
axes[0].set_ylabel('Residuals', fontweight='bold', fontsize=16)
axes[0].grid(True, alpha=0.3)
bold_ticks(axes[0])

# Histogram
axes[1].hist(residuals, bins=30, color='#d62728', alpha=0.7, edgecolor='black', linewidth=1.5)
axes[1].axvline(0, color='r', linestyle='--', linewidth=3)
axes[1].axvline(residuals.mean(), color='orange', linestyle=':', linewidth=3,
               label=f'Mean: {residuals.mean():.2f}')
axes[1].set_xlabel('Residuals', fontweight='bold', fontsize=16)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=16)
axes[1].legend(frameon=True, fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')
bold_ticks(axes[1])

save_fig('Fig13_Residual_Analysis_RUL')

# ============================================================================
# FIGURE 14: Uncertainty Calibration Analysis
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Wear uncertainty
mask_w = val_df['wear_true'].notna()
wear_std = val_df.loc[mask_w, 'wear_std'].values
wear_error = np.abs(val_df.loc[mask_w, 'wear_pred'].values - val_df.loc[mask_w, 'wear_true'].values)

axes[0].scatter(wear_std, wear_error, alpha=0.6, s=80, color='#1f77b4',
               edgecolors='white', linewidth=1.0)
max_val = max(wear_std.max(), wear_error.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=3, label='Perfect Calibration')
axes[0].set_xlabel('Predicted Std (Wear)', fontweight='bold', fontsize=16)
axes[0].set_ylabel('Absolute Error', fontweight='bold', fontsize=16)
axes[0].legend(frameon=True, fontsize=13)
axes[0].grid(True, alpha=0.3)
bold_ticks(axes[0])

# RUL uncertainty
mask_r = val_df['rul_true'].notna()
rul_std = val_df.loc[mask_r, 'rul_std'].values
rul_error = np.abs(val_df.loc[mask_r, 'rul_pred'].values - val_df.loc[mask_r, 'rul_true'].values)

axes[1].scatter(rul_std, rul_error, alpha=0.6, s=80, color='#d62728',
               edgecolors='white', linewidth=1.0)
max_val = max(rul_std.max(), rul_error.max())
axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=3, label='Perfect Calibration')
axes[1].set_xlabel('Predicted Std (RUL)', fontweight='bold', fontsize=16)
axes[1].set_ylabel('Absolute Error', fontweight='bold', fontsize=16)
axes[1].legend(frameon=True, fontsize=13)
axes[1].grid(True, alpha=0.3)
bold_ticks(axes[1])

save_fig('Fig14_Uncertainty_Calibration')

# ============================================================================
# FIGURE 15: Error Distribution Comparison
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Wear errors
mask_w = val_df['wear_true'].notna()
wear_errors = val_df.loc[mask_w, 'wear_pred'].values - val_df.loc[mask_w, 'wear_true'].values

axes[0].hist(wear_errors, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black', 
             linewidth=1.5, density=True)
axes[0].axvline(0, color='r', linestyle='--', linewidth=3)
axes[0].axvline(wear_errors.mean(), color='orange', linestyle=':', linewidth=3,
               label=f'Mean: {wear_errors.mean():.2f}')
axes[0].set_xlabel('Prediction Error (Wear)', fontweight='bold', fontsize=16)
axes[0].set_ylabel('Density', fontweight='bold', fontsize=16)
axes[0].legend(frameon=True, fontsize=13)
axes[0].grid(True, alpha=0.3, axis='y')
bold_ticks(axes[0])

# RUL errors
mask_r = val_df['rul_true'].notna()
rul_errors = val_df.loc[mask_r, 'rul_pred'].values - val_df.loc[mask_r, 'rul_true'].values

axes[1].hist(rul_errors, bins=30, color='#d62728', alpha=0.7, edgecolor='black', 
             linewidth=1.5, density=True)
axes[1].axvline(0, color='r', linestyle='--', linewidth=3)
axes[1].axvline(rul_errors.mean(), color='orange', linestyle=':', linewidth=3,
               label=f'Mean: {rul_errors.mean():.2f}')
axes[1].set_xlabel('Prediction Error (RUL)', fontweight='bold', fontsize=16)
axes[1].set_ylabel('Density', fontweight='bold', fontsize=16)
axes[1].legend(frameon=True, fontsize=13)
axes[1].grid(True, alpha=0.3, axis='y')
bold_ticks(axes[1])

save_fig('Fig15_Error_Distribution_Comparison')

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ ALL 15 MSSP PUBLICATION FIGURES GENERATED SUCCESSFULLY!")
print("="*80)
print(f"\nOutput Directory: {FIG_DIR}")
print(f"\nFormats: PNG ({DPI} DPI) + PDF (vector)")
print("\nGenerated Figures:")
print("  [1-4]  Training Dynamics (Loss, RMSE, R², Learning Rate)")
print("  [5-6]  Validation Scatter Plots (Wear & RUL)")
print("  [7-8]  Validation Trajectories (Wear & RUL per Cutter)")
print("  [9]    Validation Phase Space (Wear vs RUL)")
print("  [10-11] Test Predictions (Wear & RUL per Cutter)")
print("  [12-13] Residual Analysis (Wear & RUL)")
print("  [14]   Uncertainty Calibration")
print("  [15]   Error Distributions")
print("\n" + "="*80)
print("🎓 Ready for MSSP submission!")
print("="*80 + "\n")
