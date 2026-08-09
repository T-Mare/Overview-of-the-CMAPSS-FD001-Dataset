import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import utilities
from Utilities.config import RANDOM_SEED
from sklearn.ensemble import RandomForestRegressor

# ==========
# CONFIGURATION
# ==========

# Data paths
DATA_PATH = Path(project_root) / "CodeBase_Experiments" / "0_Data_Processing" / "Data_CMAPSS" / "2_Cleaned_Data" / "Non_Windowed"

# Output paths
OUTPUT_BASE = Path(project_root) / "Results" / "Phase2_Feature_Selection" / "FD001" / "Feature_Analysis"
CORR_DIR = OUTPUT_BASE / "correlation_based"
TREE_DIR = OUTPUT_BASE / "tree_based"
COMPARISON_DIR = OUTPUT_BASE / "feature_comparison"

# Create output directories
for dir_path in [CORR_DIR, TREE_DIR, COMPARISON_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Cumulative importance threshold
IMPORTANCE_THRESHOLD = 0.99

# Random seed
np.random.seed(RANDOM_SEED)

# ==========
# DATA LOADING
# ==========

def load_100pct_training_data():
    print("Loading FD001 100% training data...")
    
    train_features = pd.read_csv(DATA_PATH / "FD001_train_features.csv")
    train_ids = pd.read_csv(DATA_PATH / "FD001_train_ids.csv")
    
    # Combine features and IDs
    train_df = pd.concat([train_ids, train_features], axis=1)
    
    # Separate features and target
    # Exclude: engine, engine_id, cycle, RUL, time_cycles (identifiers, not features!)
    exclude_cols = ['engine', 'engine_id', 'cycle', 'RUL', 'time_cycles']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['RUL']
    
    print(f"Loaded {len(X_train)} samples with {X_train.shape[1]} features")
    print(f"Features: {list(X_train.columns)}")
    
    return X_train, y_train, feature_cols

# ==========
# FEATURE SELECTION METHODS
# ==========

def apply_correlation_fs(X_train, y_train, threshold=0.99):
    print("\n" + "="*40)
    print("CORRELATION-BASED FEATURE SELECTION")
    print("="*40)
    
    # Calculate Pearson correlation with RUL
    correlations = X_train.corrwith(y_train).abs()
    
    # Sort by correlation (descending)
    correlations = correlations.sort_values(ascending=False)
    
    # Calculate cumulative percentage
    cumulative_pct = correlations.cumsum() / correlations.sum()
    
    # Create DataFrame
    correlations_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values,
        'cumulative_pct': cumulative_pct.values
    })
    
    # Select features until threshold
    selected_features = correlations_df[correlations_df['cumulative_pct'] <= threshold]['feature'].tolist()
    
    # Print summary
    print(f"\n  Total features: {len(correlations_df)}")
    print(f"Selected features: {len(selected_features)} ({len(selected_features)/len(correlations_df)*100:.1f}%)")
    print(f"Cumulative correlation captured: {cumulative_pct[selected_features[-1]]:.1%}")
    print(f"\n  Top 5 features by correlation:")
    for idx, row in correlations_df.head(5).iterrows():
        print(f"{row['feature']:20s} | Corr: {row['correlation']:.4f} | Cum: {row['cumulative_pct']:.1%}")
    
    print(f"\n  Selected features: {selected_features}")
    
    return selected_features, correlations_df

def apply_tree_based_fs(X_train, y_train, threshold=0.99):
    print("\n" + "="*40)
    print("TREE-BASED FEATURE SELECTION (Random Forest)")
    print("="*40)
    
    # Train Random Forest on 100% data
    print("\n  Training Random Forest for feature importance...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("Training complete")
    
    # Get feature importances
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    
    # Sort by importance (descending)
    importances = importances.sort_values(ascending=False)
    
    # Calculate cumulative percentage
    cumulative_pct = importances.cumsum() / importances.sum()
    
    # Create DataFrame
    importances_df = pd.DataFrame({
        'feature': importances.index,
        'importance': importances.values,
        'cumulative_pct': cumulative_pct.values
    })
    
    # Select features until threshold
    selected_features = importances_df[importances_df['cumulative_pct'] <= threshold]['feature'].tolist()
    
    # Print summary
    print(f"\n  Total features: {len(importances_df)}")
    print(f"Selected features: {len(selected_features)} ({len(selected_features)/len(importances_df)*100:.1f}%)")
    print(f"Cumulative importance captured: {cumulative_pct[selected_features[-1]]:.1%}")
    print(f"\n  Top 5 features by importance:")
    for idx, row in importances_df.head(5).iterrows():
        print(f"{row['feature']:20s} | Imp: {row['importance']:.4f} | Cum: {row['cumulative_pct']:.1%}")
    
    print(f"\n  Selected features: {selected_features}")
    
    return selected_features, importances_df

# ==========
# VISUALIZATION
# ==========

def plot_cumulative_importance(importance_df, threshold, method_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    n_features = len(importance_df)
    x = np.arange(1, n_features + 1)
    y = importance_df['cumulative_pct'].values
    
    # Plot cumulative curve
    ax.plot(x, y, linewidth=2, marker='o', markersize=4, label='Cumulative Importance')
    
    # Add threshold line
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'{threshold:.0%} Threshold')
    
    # Mark selected features
    n_selected = (y <= threshold).sum()
    ax.axvline(x=n_selected, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Selected: {n_selected} features')
    ax.fill_between(x[:n_selected], 0, 1, alpha=0.2, color='green')
    
    # Labels and formatting
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Cumulative Importance', fontsize=12)
    ax.set_title(f'{method_name} - Cumulative Feature Importance (99% Threshold)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, n_features + 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_importance_bars(importance_df, method_name, metric_name, output_path, top_n=None):
    
    # Use all features if top_n not specified
    if top_n is None:
        top_n = len(importance_df)
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))  # Dynamic height based on number of features
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    # Create bar plot
    bars = ax.barh(range(len(top_features)), top_features[metric_name].values)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()  # Highest at top
    
    # Color bars (gradient)
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_features)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Labels
    ax.set_xlabel(metric_name.capitalize(), fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    title_suffix = f'Top {top_n}' if top_n < len(importance_df) else 'All'
    ax.set_title(f'{method_name} - {title_suffix} Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_feature_overlap_venn(corr_features, tree_features, output_path):
    try:
        from matplotlib_venn import venn2
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create Venn diagram
        venn = venn2(
            [set(corr_features), set(tree_features)],
            set_labels=('Correlation-Based', 'Tree-Based'),
            ax=ax
        )
        
        # Customize colors
        venn.get_patch_by_id('10').set_color('#ff9999')
        venn.get_patch_by_id('01').set_color('#99cc99')
        venn.get_patch_by_id('11').set_color('#ffcc99')
        
        # Title
        ax.set_title('Feature Selection Overlap (99% Cumulative Threshold)', fontsize=14, fontweight='bold')
        
        # Add overlap details as text
        overlap = set(corr_features) & set(tree_features)
        overlap_text = f"\nCommon features ({len(overlap)}):\n" + "\n".join(f"   {f}" for f in sorted(overlap))
        
        plt.figtext(0.5, -0.05, overlap_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
        
    except ImportError:
        print("matplotlib-venn not installed. Skipping Venn diagram.")
        print("Install with: pip install matplotlib-venn")

# ==========
# SAVE RESULTS
# ==========

def save_feature_list(features, output_path):
    with open(output_path, 'w') as f:
        for feature in features:
            f.write(f"{feature}\n")
    print(f"Saved: {output_path}")

def save_importance_table(importance_df, output_path):
    importance_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

def save_summary_report(corr_features, tree_features, output_path):
    overlap = set(corr_features) & set(tree_features)
    corr_only = set(corr_features) - set(tree_features)
    tree_only = set(tree_features) - set(corr_features)
    
    with open(output_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("FEATURE SELECTION SUMMARY - 99% CUMULATIVE IMPORTANCE THRESHOLD\n")
        f.write("="*40 + "\n\n")
        
        f.write(f"Correlation-Based FS: {len(corr_features)} features selected\n")
        f.write(f"Tree-Based FS: {len(tree_features)} features selected\n\n")
        
        f.write(f"Common features: {len(overlap)}\n")
        for feature in sorted(overlap):
            f.write(f"{feature}\n")
        f.write("\n")
        
        f.write(f"Correlation-only features: {len(corr_only)}\n")
        for feature in sorted(corr_only):
            f.write(f"{feature}\n")
        f.write("\n")
        
        f.write(f"Tree-only features: {len(tree_only)}\n")
        for feature in sorted(tree_only):
            f.write(f"{feature}\n")
    
    print(f"Saved: {output_path}")

# ==========
# MAIN EXECUTION
# ==========

def main():
    print("\n" + "="*40)
    print("FEATURE SELECTION - 99% CUMULATIVE IMPORTANCE THRESHOLD")
    print("="*40)
    print(f"\nOutput directory: {OUTPUT_BASE}")
    print(f"Importance threshold: {IMPORTANCE_THRESHOLD:.0%}\n")
    
    # Load data
    X_train, y_train, all_features = load_100pct_training_data()
    
    # Apply Correlation-Based FS
    corr_features, corr_df = apply_correlation_fs(X_train, y_train, IMPORTANCE_THRESHOLD)
    
    print("\nSaving Correlation-Based FS results...")
    save_feature_list(corr_features, CORR_DIR / "selected_features.txt")
    save_importance_table(corr_df, CORR_DIR / "feature_correlations.csv")
    plot_cumulative_importance(corr_df, IMPORTANCE_THRESHOLD, "Correlation-Based FS", CORR_DIR / "cumulative_correlation_plot.png")
    plot_feature_importance_bars(corr_df, "Correlation-Based FS", "correlation", CORR_DIR / "feature_importance_barplot.png")
    
    # Apply Tree-Based FS
    tree_features, tree_df = apply_tree_based_fs(X_train, y_train, IMPORTANCE_THRESHOLD)
    
    print("\nSaving Tree-Based FS results...")
    save_feature_list(tree_features, TREE_DIR / "selected_features.txt")
    save_importance_table(tree_df, TREE_DIR / "feature_importances.csv")
    plot_cumulative_importance(tree_df, IMPORTANCE_THRESHOLD, "Tree-Based FS (Random Forest)", TREE_DIR / "cumulative_importance_plot.png")
    plot_feature_importance_bars(tree_df, "Tree-Based FS (Random Forest)", "importance", TREE_DIR / "feature_importance_barplot.png")
    
    # Feature comparison
    print("\n" + "="*40)
    print("FEATURE COMPARISON")
    print("="*40)
    overlap = set(corr_features) & set(tree_features)
    print(f"\n  Common features: {len(overlap)} / {len(set(corr_features) | set(tree_features))} total unique")
    print(f"Overlap percentage: {len(overlap)/len(set(corr_features) | set(tree_features))*100:.1f}%")
    
    print("\nSaving comparison results...")
    plot_feature_overlap_venn(corr_features, tree_features, COMPARISON_DIR / "feature_overlap_venn.png")
    save_summary_report(corr_features, tree_features, COMPARISON_DIR / "selected_features_summary.txt")
    
    # Save consolidated summary
    summary = {
        'importance_threshold': IMPORTANCE_THRESHOLD,
        'total_features': len(all_features),
        'correlation_based': {
            'n_selected': len(corr_features),
            'features': corr_features
        },
        'tree_based': {
            'n_selected': len(tree_features),
            'features': tree_features
        },
        'overlap': {
            'n_common': len(overlap),
            'common_features': list(overlap)
        }
    }
    
    with open(OUTPUT_BASE / "feature_selection_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved: {OUTPUT_BASE / 'feature_selection_summary.json'}")
    
    print("\n" + "="*40)
    print(" FEATURE SELECTION COMPLETE!")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_BASE}")
    print("\nNext step: Run Phase 2 experiments with selected features")
    print("Command: python Automation_Scripts/run_phase2_fs_experiments.py")
    print("="*40)

if __name__ == "__main__":
    main()

