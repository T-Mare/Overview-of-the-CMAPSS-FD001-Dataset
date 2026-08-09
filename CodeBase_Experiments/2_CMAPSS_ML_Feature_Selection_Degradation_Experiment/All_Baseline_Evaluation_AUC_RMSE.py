import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import config
from Utilities.config import ENGINE_COUNTS_HIGH, ENGINE_COUNTS_LOW

# ==========
# CONFIGURATION
# ==========

RESULTS_DIR = project_root / "Results" / "Phase1_Baseline" / "FD001"
OUTPUT_DIR = RESULTS_DIR / "Evaluation_Plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Deep Learning models
DL_MODELS = ['ANN', 'BiLSTM', 'CNN', 'GRU', 'LSTM', 'RNN', 'TCN', 'Transformer']

# ==========
# LOAD DATA
# ==========

def load_all_results():
    print("Loading Phase 1 baseline results...")
    
    # Load regression results
    regression_df = pd.read_csv(RESULTS_DIR / "regression_all_results.csv")
    print(f"Loaded {len(regression_df)} regression records")
    
    # Load tree-based results
    tree_df = pd.read_csv(RESULTS_DIR / "tree_all_results.csv")
    print(f"Loaded {len(tree_df)} tree-based records")
    
    # Load deep learning results from individual model folders
    all_dl_data = []
    for model_name in DL_MODELS:
        model_file = RESULTS_DIR / model_name / f"{model_name}_metrics_summary.csv"
        if model_file.exists():
            df = pd.read_csv(model_file)
            all_dl_data.append(df)
    
    dl_df = pd.concat(all_dl_data, ignore_index=True) if all_dl_data else pd.DataFrame()
    print(f"Loaded {len(dl_df)} deep learning records")
    
    # Standardize column names
    if 'dataset' in regression_df.columns:
        regression_df.rename(columns={'dataset': 'split'}, inplace=True)
    
    return regression_df, tree_df, dl_df

# ==========
# PLOTTING FUNCTIONS
# ==========

def plot_auc_rmse_evolution(df, model_family, output_dir, engine_range, range_label):
    print(f"\nPlotting AUC-RMSE evolution for {model_family} models ({range_label})...")
    
    # Filter to validation set only
    val_data = df[df['split'] == 'val'].copy()
    
    # Filter to specified engine range
    val_data = val_data[val_data['n_engines'].isin(engine_range)]
    
    if len(val_data) == 0:
        print(f"Warning: No validation data found for {model_family}")
        return
    
    # Get unique models
    models = val_data['model_name'].unique()
    print(f"Found {len(models)} models: {list(models)}")
    
    # Create figure
    plt.figure(figsize=(16, 8))
    
    # Color palette by family
    if model_family == 'Regression':
        cmap = plt.cm.Reds
    elif model_family == 'Tree-Based':
        cmap = plt.cm.Greens
    else:  # Deep Learning
        cmap = plt.cm.Blues
    
    colors = {model: cmap(0.4 + 0.6 * i / max(len(models) - 1, 1)) 
              for i, model in enumerate(sorted(models))}
    
    # Plot each model and collect endpoint data for labeling
    line_endpoints = []  # Store (auc_rmse_at_10pct, model_name, color) for labeling
    
    for model in sorted(models):
        model_data = val_data[val_data['model_name'] == model].sort_values('n_engines', ascending=False)
        
        if len(model_data) == 0:
            continue
        
        plt.plot(model_data['n_engines'], model_data['auc_rmse'],
                marker='o', linewidth=2.5, markersize=8,
                label=model, color=colors[model], alpha=0.9)
        
        # Get AUC-RMSE at lowest engine count in this range for labeling
        min_engines = min(engine_range)
        auc_rmse_at_min = model_data[model_data['n_engines'] == min_engines]['auc_rmse'].values
        if len(auc_rmse_at_min) > 0:
            line_endpoints.append((auc_rmse_at_min[0], model, colors[model]))
    
    # Add biology-style labels aligned vertically on the right with connector lines
    # Sort by AUC-RMSE to keep similar values closer
    line_endpoints.sort(key=lambda x: x[0])
    
    # Calculate evenly spaced positions for labels
    n_labels = len(line_endpoints)
    min_engines = min(engine_range)
    if range_label == "high_engines":
        label_y_positions = np.linspace(11, 24, n_labels)
        label_x = min_engines - 2  # X position for labels (right side, beyond lowest data point)
        line_start_x = min_engines  # Start connector from lowest engine count in range
    else:  # low_engines
        label_y_positions = np.linspace(15, 75, n_labels)  # More space for low engine range
        label_x = min_engines - 1.5  # X position for labels (left side for low engines)
        line_start_x = min_engines  # Start connector from lowest engine count
    
    for idx, (auc_rmse_val, model, color) in enumerate(line_endpoints):
        label_y = label_y_positions[idx]
        
        # Draw connector line from actual data point to label
        plt.plot([line_start_x, label_x], [auc_rmse_val, label_y], 
                linestyle=':', linewidth=1, color=color, alpha=0.6)
        
        # Add label with color-coded background
        plt.text(label_x, label_y, f' {model} ', 
                verticalalignment='center', horizontalalignment='left' if label_x > 0 else 'right',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                         edgecolor='white', linewidth=1.5, alpha=0.95))
    
    # Formatting
    plt.xlabel('Number of Training Engines', fontsize=13, fontweight='bold')
    plt.ylabel('AUC-RMSE (Validation Set)', fontsize=13, fontweight='bold')
    
    range_title = "High Engine Counts (80-10)" if range_label == "high_engines" else "Low Engine Counts (10-1)"
    plt.title(f'{model_family} Models: AUC-RMSE Evolution - {range_title}',
              fontsize=15, fontweight='bold', pad=20)
    
    if model_family == 'Regression':
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10, frameon=True)
    elif model_family == 'Tree-Based':
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=10, frameon=True)
    else:  # Deep Learning
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=10, frameon=True)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    # Let matplotlib handle x-ticks automatically based on n_engines data
    
    # Set y-axis limits based on range
    min_engines = min(engine_range)
    if range_label == "high_engines":
        plt.ylim(10, 25)
        plt.xlim(max(engine_range) + 5, min_engines - 4)  # Extra space on right for labels
    else:  # low_engines
        plt.ylim(10, 80)
        plt.xlim(max(engine_range) + 1, min_engines - 3)  # Extra space on left for labels
    
    plt.tight_layout()
    
    # Save
    filename = f"{model_family.lower().replace('-', '_').replace(' ', '_')}_auc_rmse_evolution_{range_label}.png"
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")
    plt.close()

def plot_combined_baseline_evaluation_auc_rmse(regression_df, tree_df, dl_df, output_dir, engine_range, range_label):
    print(f"\nPlotting combined baseline AUC-RMSE evaluation (All 3 families) ({range_label})...")
    
    # Filter to validation set only
    reg_val = regression_df[regression_df['split'] == 'val'].copy()
    tree_val = tree_df[tree_df['split'] == 'val'].copy()
    dl_val = dl_df[dl_df['split'] == 'val'].copy()
    
    # Filter to specified engine range
    reg_val = reg_val[reg_val['n_engines'].isin(engine_range)]
    tree_val = tree_val[tree_val['n_engines'].isin(engine_range)]
    dl_val = dl_val[dl_val['n_engines'].isin(engine_range)]
    
    # Add family labels
    reg_val['family'] = 'Regression'
    tree_val['family'] = 'Tree-Based'
    dl_val['family'] = 'Deep Learning'
    
    # Combine
    val_data = pd.concat([reg_val, tree_val, dl_val], ignore_index=True)
    
    if len(val_data) == 0:
        print(f"Warning: No validation data found")
        return
    
    # Get unique models
    models = val_data['model_name'].unique()
    print(f"Found {len(models)} models total")
    
    # Create figure
    plt.figure(figsize=(18, 10))
    
    # Color palette by family
    reg_models = reg_val['model_name'].unique()
    tree_models = tree_val['model_name'].unique()
    dl_models = dl_val['model_name'].unique()
    
    reg_colors = {model: plt.cm.Reds(0.4 + 0.6 * i / max(len(reg_models) - 1, 1)) 
                  for i, model in enumerate(sorted(reg_models))}
    tree_colors = {model: plt.cm.Greens(0.4 + 0.6 * i / max(len(tree_models) - 1, 1)) 
                   for i, model in enumerate(sorted(tree_models))}
    dl_colors = {model: plt.cm.Blues(0.4 + 0.6 * i / max(len(dl_models) - 1, 1)) 
                 for i, model in enumerate(sorted(dl_models))}
    
    colors = {**reg_colors, **tree_colors, **dl_colors}
    
    # Plot each model and collect endpoint data for labeling
    line_endpoints = []  # Store (auc_rmse_at_10pct, model_name, color) for labeling
    
    for model in sorted(models):
        model_data = val_data[val_data['model_name'] == model].sort_values('n_engines', ascending=False)
        
        if len(model_data) == 0:
            continue
        
        family = model_data['family'].iloc[0]
        
        plt.plot(model_data['n_engines'], model_data['auc_rmse'],
                marker='o', linewidth=2.5, markersize=8,
                label=f"{model} ({family})", color=colors[model], alpha=0.9)
        
        # Get AUC-RMSE at lowest engine count in this range for labeling
        min_engines = min(engine_range)
        auc_rmse_at_min = model_data[model_data['n_engines'] == min_engines]['auc_rmse'].values
        if len(auc_rmse_at_min) > 0:
            line_endpoints.append((auc_rmse_at_min[0], model, colors[model]))
    
    # Add biology-style labels aligned vertically on the right with connector lines
    # Sort by AUC-RMSE to keep similar values closer
    line_endpoints.sort(key=lambda x: x[0])
    
    # Calculate evenly spaced positions for labels
    n_labels = len(line_endpoints)
    min_engines = min(engine_range)
    if range_label == "high_engines":
        label_y_positions = np.linspace(10.5, 24.5, n_labels)
        label_x = min_engines - 2  # X position for labels (right side, beyond lowest data point)
        line_start_x = min_engines  # Start connector from lowest engine count in range
    else:  # low_engines
        label_y_positions = np.linspace(15, 75, n_labels)  # More space for low engine range
        label_x = min_engines - 1.5  # X position for labels (left side for low engines)
        line_start_x = min_engines  # Start connector from lowest engine count
    
    for idx, (auc_rmse_val, model, color) in enumerate(line_endpoints):
        label_y = label_y_positions[idx]
        
        # Draw connector line from actual data point to label
        plt.plot([line_start_x, label_x], [auc_rmse_val, label_y], 
                linestyle=':', linewidth=1, color=color, alpha=0.6)
        
        # Add label with color-coded background
        plt.text(label_x, label_y, f' {model} ', 
                verticalalignment='center', horizontalalignment='left' if label_x > 0 else 'right',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                         edgecolor='white', linewidth=1.2, alpha=0.95))
    
    # Formatting
    plt.xlabel('Number of Training Engines', fontsize=14, fontweight='bold')
    plt.ylabel('AUC-RMSE (Validation Set)', fontsize=14, fontweight='bold')
    
    range_title = "High Engine Counts (80-10)" if range_label == "high_engines" else "Low Engine Counts (10-1)"
    plt.title(f'All Baseline Models: AUC-RMSE Evolution - {range_title}\n(Regression, Tree-Based, and Deep Learning)',
              fontsize=16, fontweight='bold', pad=20)
    
    # Create 3 separate legend boxes grouped by family
    from matplotlib.lines import Line2D
    
    # Get handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Group by family
    reg_handles, reg_labels = [], []
    tree_handles, tree_labels = [], []
    dl_handles, dl_labels = [], []
    
    for handle, label in zip(handles, labels):
        if '(Regression)' in label:
            reg_handles.append(handle)
            reg_labels.append(label.replace(' (Regression)', ''))
        elif '(Tree-Based)' in label:
            tree_handles.append(handle)
            tree_labels.append(label.replace(' (Tree-Based)', ''))
        elif '(Deep Learning)' in label:
            dl_handles.append(handle)
            dl_labels.append(label.replace(' (Deep Learning)', ''))
    
    # Create 3 legend boxes side by side at the bottom
    # Regression (left)
    leg1 = plt.legend(reg_handles, reg_labels, 
                     loc='upper left', bbox_to_anchor=(0.02, -0.08),
                     ncol=3, fontsize=11, frameon=True, title='Regression',
                     title_fontproperties={'weight': 'bold', 'size': 12})
    plt.gca().add_artist(leg1)
    
    # Tree-Based (center)
    leg2 = plt.legend(tree_handles, tree_labels,
                     loc='upper center', bbox_to_anchor=(0.5, -0.08),
                     ncol=3, fontsize=11, frameon=True, title='Tree-Based',
                     title_fontproperties={'weight': 'bold', 'size': 12})
    plt.gca().add_artist(leg2)
    
    # Deep Learning (right)
    leg3 = plt.legend(dl_handles, dl_labels,
                     loc='upper right', bbox_to_anchor=(0.98, -0.08),
                     ncol=4, fontsize=11, frameon=True, title='Deep Learning',
                     title_fontproperties={'weight': 'bold', 'size': 12})
    
    plt.grid(True, alpha=0.3, linestyle='--')
    # Let matplotlib handle x-ticks automatically based on n_engines data
    
    # Set y-axis limits based on range
    min_engines = min(engine_range)
    if range_label == "high_engines":
        plt.ylim(10, 25)
        plt.xlim(max(engine_range) + 5, min_engines - 4)  # Extra space on right for labels
    else:  # low_engines
        plt.ylim(10, 80)
        plt.xlim(max(engine_range) + 1, min_engines - 3)  # Extra space on left for labels
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'combined_baseline_evaluation_auc_rmse_{range_label}.png'
    plt.savefig(output_file, dpi=300)
    print(f"Saved: {output_file}")
    plt.close()

# ==========
# MAIN EXECUTION
# ==========

def main():
    print("="*40)
    print("ALL BASELINE MODELS EVALUATION - AUC-RMSE")
    print("="*40)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Load data
    regression_df, tree_df, dl_df = load_all_results()
    
    print("\n" + "="*40)
    print("GENERATING HIGH ENGINE COUNT PLOTS (80-10 engines)")
    print("="*40)
    
    # Plot AUC-RMSE evolution for Regression models (high engines)
    plot_auc_rmse_evolution(regression_df, 'Regression', OUTPUT_DIR, ENGINE_COUNTS_HIGH, 'high_engines')
    
    # Plot AUC-RMSE evolution for Tree-Based models (high engines)
    plot_auc_rmse_evolution(tree_df, 'Tree-Based', OUTPUT_DIR, ENGINE_COUNTS_HIGH, 'high_engines')
    
    # Plot AUC-RMSE evolution for Deep Learning models (high engines)
    plot_auc_rmse_evolution(dl_df, 'Deep Learning', OUTPUT_DIR, ENGINE_COUNTS_HIGH, 'high_engines')
    
    # Plot combined evaluation (high engines)
    plot_combined_baseline_evaluation_auc_rmse(regression_df, tree_df, dl_df, OUTPUT_DIR, ENGINE_COUNTS_HIGH, 'high_engines')
    
    print("\n" + "="*40)
    print("GENERATING LOW ENGINE COUNT PLOTS (10-1 engines)")
    print("="*40)
    
    # Plot AUC-RMSE evolution for Regression models (low engines)
    plot_auc_rmse_evolution(regression_df, 'Regression', OUTPUT_DIR, ENGINE_COUNTS_LOW, 'low_engines')
    
    # Plot AUC-RMSE evolution for Tree-Based models (low engines)
    plot_auc_rmse_evolution(tree_df, 'Tree-Based', OUTPUT_DIR, ENGINE_COUNTS_LOW, 'low_engines')
    
    # Plot AUC-RMSE evolution for Deep Learning models (low engines)
    plot_auc_rmse_evolution(dl_df, 'Deep Learning', OUTPUT_DIR, ENGINE_COUNTS_LOW, 'low_engines')
    
    # Plot combined evaluation (low engines)
    plot_combined_baseline_evaluation_auc_rmse(regression_df, tree_df, dl_df, OUTPUT_DIR, ENGINE_COUNTS_LOW, 'low_engines')
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE!")
    print("- 4 High engine count plots (80-10 engines)")
    print("- 4 Low engine count plots (10-1 engines)")
    print("="*40)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("1. regression_auc_rmse_evolution.png")
    print("2. tree_based_auc_rmse_evolution.png")
    print("3. deep_learning_auc_rmse_evolution.png")
    print("4. combined_baseline_evaluation_auc_rmse.png (ALL MODELS)")
    print("="*40)

if __name__ == "__main__":
    main()

