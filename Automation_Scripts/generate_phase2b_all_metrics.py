import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import config
from Utilities.config import ENGINE_COUNTS_ALL

# ==========
# CONFIGURATION
# ==========

SELECTED_MODELS = ['Poly2', 'Lasso', 'RF', 'XGB', 'GRU', 'BiLSTM', 'LSTM']

MODEL_FAMILIES = {
    'Poly2': 'Regression',
    'Lasso': 'Regression',
    'RF': 'Tree',
    'XGB': 'Tree',
    'GRU': 'Deep Learning',
    'BiLSTM': 'Deep Learning',
    'LSTM': 'Deep Learning'
}

FAMILY_COLORS = {
    'Regression': '#d62728',  # Red
    'Tree': '#2ca02c',        # Green
    'Deep Learning': '#1f77b4' # Blue
}

# Metrics configuration
METRICS = {
    'rmse': {'name': 'RMSE', 'lower_is_better': True, 'ylabel': 'RMSE (Lower is Better)'},
    'mae': {'name': 'MAE', 'lower_is_better': True, 'ylabel': 'MAE (Lower is Better)'},
    'r2': {'name': 'R²', 'lower_is_better': False, 'ylabel': 'R² (Higher is Better)'},
    'cmapss': {'name': 'CMAPSS Score', 'lower_is_better': True, 'ylabel': 'CMAPSS Score (Lower is Better)'},
    'auc_rmse': {'name': 'AUC-RMSE', 'lower_is_better': True, 'ylabel': 'AUC-RMSE (Lower is Better)'}
}

ENGINE_COUNTS = ENGINE_COUNTS_ALL  # [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

PHASE1_DIR = PROJECT_ROOT / 'Results' / 'Phase1_Baseline' / 'FD001'
PHASE2_DIR = PROJECT_ROOT / 'Results' / 'Phase2_Feature_Selection' / 'FD001'
OUTPUT_DIR = PHASE2_DIR / 'Analysis' / 'Phase2B_All_Metrics'

# ==========
# DATA LOADING (Same as Phase 2A)
# ==========

def load_phase1_baseline(model_name):
    csv_path = PHASE1_DIR / model_name / f'{model_name}_metrics_summary.csv'
    
    if not csv_path.exists():
        print(f" Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Handle different column names for split/dataset
    if 'split' in df.columns:
        split_col = 'split'
    elif 'dataset' in df.columns:
        split_col = 'dataset'
    else:
        print(f" Warning: No split/dataset column found in {model_name}")
        return None
    
    # Get validation metrics (from val split)
    val_df = df[df[split_col] == 'val'][['model_name', 'n_engines', 'rmse', 'mae', 'r2', 'cmapss_score', 'auc_rmse']].copy()
    
    # Rename cmapss_score to cmapss for consistency
    if 'cmapss_score' in val_df.columns:
        val_df.rename(columns={'cmapss_score': 'cmapss'}, inplace=True)
    
    # Get training time (from train split)
    train_df = df[df[split_col] == 'train'][['model_name', 'n_engines', 'training_time_sec']].copy()
    
    # Merge validation metrics with training time
    result_df = pd.merge(val_df, train_df, on=['model_name', 'n_engines'], how='left')
    
    # Add method column
    result_df['method'] = 'Baseline'
    
    return result_df

def load_phase2_fs(model_name, fs_method):
    if fs_method == 'Correlation_FS':
        csv_path = PHASE2_DIR / 'Correlation_FS' / model_name / f'{model_name}_metrics_summary.csv'
    elif fs_method == 'Tree_FS':
        csv_path = PHASE2_DIR / 'Tree_FS' / model_name / f'{model_name}_metrics_summary.csv'
    else:
        raise ValueError(f"Unknown FS method: {fs_method}")
    
    if not csv_path.exists():
        print(f" Warning: {csv_path} not found")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Rename columns for consistency
    df['method'] = fs_method
    if 'cmapss_score' in df.columns:
        df.rename(columns={'cmapss_score': 'cmapss'}, inplace=True)
    
    # Select relevant columns
    columns = ['model_name', 'n_engines', 'method', 'val_rmse', 'val_mae', 'val_r2', 'val_cmapss', 'val_auc_rmse', 'training_time_sec']
    
    # Rename val_* to match baseline
    rename_map = {
        'val_rmse': 'rmse',
        'val_mae': 'mae',
        'val_r2': 'r2',
        'val_cmapss': 'cmapss',
        'val_auc_rmse': 'auc_rmse'
    }
    
    df.rename(columns=rename_map, inplace=True)
    
    return df[['model_name', 'n_engines', 'method', 'rmse', 'mae', 'r2', 'cmapss', 'auc_rmse', 'training_time_sec']]

def load_all_results():
    print("="*40)
    print("LOADING PHASE 1 & PHASE 2 RESULTS (ALL METRICS)")
    print("="*40)
    
    all_dfs = []
    
    for model in SELECTED_MODELS:
        print(f"\nLoading {model}...")
        
        # Load Phase 1 Baseline
        baseline_df = load_phase1_baseline(model)
        if baseline_df is not None:
            all_dfs.append(baseline_df)
            print(f"Baseline: {len(baseline_df)} records")
        
        # Load Phase 2 Correlation_FS
        corr_df = load_phase2_fs(model, 'Correlation_FS')
        if corr_df is not None:
            all_dfs.append(corr_df)
            print(f"Correlation_FS: {len(corr_df)} records")
        
        # Load Phase 2 Tree_FS
        tree_df = load_phase2_fs(model, 'Tree_FS')
        if tree_df is not None:
            all_dfs.append(tree_df)
            print(f"Tree_FS: {len(tree_df)} records")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Add model family column
    combined_df['model_family'] = combined_df['model_name'].map(MODEL_FAMILIES)
    
    print(f"\n Total records loaded: {len(combined_df)}")
    print(f" Models: {combined_df['model_name'].nunique()}")
    print(f" Methods: {combined_df['method'].unique().tolist()}")
    print(f" Metrics: rmse, mae, r2, cmapss, auc_rmse")
    
    return combined_df

# ==========
# MULTI-METRIC ANALYSIS
# ==========

def generate_improvement_heatmap_for_metric(df, metric_col, metric_info, output_dir):
    lower_is_better = metric_info['lower_is_better']
    
    # Prepare data for heatmap
    improvement_data = []
    
    for model in SELECTED_MODELS:
        model_improvements = []
        
        for n_eng in ENGINE_COUNTS:
            # Get baseline value
            baseline_val = df[(df['model_name'] == model) & 
                             (df['n_engines'] == n_eng) & 
                             (df['method'] == 'Baseline')][metric_col].values
            
            if len(baseline_val) == 0 or pd.isna(baseline_val[0]):
                model_improvements.append(np.nan)
                continue
            
            baseline_val = baseline_val[0]
            
            # Get best FS value
            fs_vals = df[(df['model_name'] == model) & 
                        (df['n_engines'] == n_eng) & 
                        (df['method'].isin(['Correlation_FS', 'Tree_FS']))][metric_col].values
            
            if len(fs_vals) == 0:
                model_improvements.append(np.nan)
                continue
            
            # Remove NaN values
            fs_vals = fs_vals[~pd.isna(fs_vals)]
            if len(fs_vals) == 0:
                model_improvements.append(np.nan)
                continue
            
            # Get best FS value (min for lower_is_better, max for higher_is_better)
            if lower_is_better:
                best_fs_val = min(fs_vals)
                # Calculate % improvement (positive = better)
                improvement_percent = ((baseline_val - best_fs_val) / abs(baseline_val)) * 100
            else:
                best_fs_val = max(fs_vals)
                # For R², positive change is good
                improvement_percent = ((best_fs_val - baseline_val) / abs(baseline_val)) * 100
            
            model_improvements.append(improvement_percent)
        
        improvement_data.append(model_improvements)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(improvement_data,
                             index=SELECTED_MODELS,
                             columns=[f'{n}eng' for n in ENGINE_COUNTS])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, vmin=-10, vmax=10,
                cbar_kws={'label': '% Improvement from Baseline'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title(f'{metric_info["name"]} Improvement from Baseline (%)\n(Positive = Better with Feature Selection)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Training Engines', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Add family separators
    regression_count = sum(1 for m in SELECTED_MODELS if MODEL_FAMILIES[m] == 'Regression')
    tree_count = sum(1 for m in SELECTED_MODELS if MODEL_FAMILIES[m] == 'Tree')
    
    ax.axhline(y=regression_count, color='black', linewidth=2)
    ax.axhline(y=regression_count + tree_count, color='black', linewidth=2)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{metric_col}_improvement_heatmap.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved: {output_path.name}")
    
    return heatmap_df

def analyze_cross_metric_consistency(df, output_dir):
    print("\nAnalyzing cross-metric consistency...")
    
    # For each model and data %, calculate if FS helps across all metrics
    consistency_data = []
    
    for model in SELECTED_MODELS:
        for n_eng in ENGINE_COUNTS:
            row = {'model': model, 'n_engines': n_eng}
            
            # Check each metric
            for metric_col, metric_info in METRICS.items():
                baseline_val = df[(df['model_name'] == model) & 
                                 (df['n_engines'] == n_eng) & 
                                 (df['method'] == 'Baseline')][metric_col].values
                
                if len(baseline_val) == 0 or pd.isna(baseline_val[0]):
                    row[f'{metric_col}_benefit'] = np.nan
                    continue
                
                baseline_val = baseline_val[0]
                
                fs_vals = df[(df['model_name'] == model) & 
                            (df['n_engines'] == n_eng) & 
                            (df['method'].isin(['Correlation_FS', 'Tree_FS']))][metric_col].values
                
                fs_vals = fs_vals[~pd.isna(fs_vals)]
                
                if len(fs_vals) == 0:
                    row[f'{metric_col}_benefit'] = np.nan
                    continue
                
                # Check if FS improves
                if metric_info['lower_is_better']:
                    improves = min(fs_vals) < baseline_val
                else:
                    improves = max(fs_vals) > baseline_val
                
                row[f'{metric_col}_benefit'] = improves
            
            consistency_data.append(row)
    
    consistency_df = pd.DataFrame(consistency_data)
    
    # Calculate how many metrics benefit for each model-pct combo
    metric_cols = [f'{m}_benefit' for m in METRICS.keys()]
    consistency_df['num_metrics_benefit'] = consistency_df[metric_cols].sum(axis=1)
    consistency_df['all_metrics_benefit'] = consistency_df['num_metrics_benefit'] == 5
    
    # Save
    output_path = output_dir / 'cross_metric_consistency.csv'
    consistency_df.to_csv(output_path, index=False)
    print(f" Saved: {output_path.name}")
    
    return consistency_df

def generate_phase2b_summary(df, all_heatmaps, consistency_df, output_dir):
    print("\nGenerating Phase 2B summary report...")
    
    report_lines = []
    report_lines.append("="*40)
    report_lines.append("PHASE 2B: MULTI-METRIC ANALYSIS SUMMARY")
    report_lines.append("="*40)
    report_lines.append("")
    report_lines.append("Metrics analyzed: RMSE, MAE, R², CMAPSS Score, AUC-RMSE")
    report_lines.append("")
    
    # 1. Average improvement across all metrics
    report_lines.append("1. AVERAGE IMPROVEMENT ACROSS ALL METRICS")
    report_lines.append("-" * 70)
    
    for metric_col, metric_info in METRICS.items():
        heatmap_df = all_heatmaps[metric_col]
        avg_improvement = heatmap_df.mean().mean()
        report_lines.append(f"   {metric_info['name']:15}: {avg_improvement:+6.2f}% average improvement")
    
    report_lines.append("")
    
    # 2. Models ranked by consistency
    report_lines.append("2. MODELS RANKED BY CROSS-METRIC CONSISTENCY")
    report_lines.append("-" * 70)
    report_lines.append("   (How often does FS improve across ALL 5 metrics?)")
    report_lines.append("")
    
    # Count how often each model benefits across all metrics
    model_consistency = []
    for model in SELECTED_MODELS:
        model_df = consistency_df[consistency_df['model'] == model]
        all_benefit_count = model_df['all_metrics_benefit'].sum()
        total_count = len(model_df[model_df['all_metrics_benefit'].notna()])
        pct_consistent = (all_benefit_count / total_count * 100) if total_count > 0 else 0
        model_consistency.append((model, all_benefit_count, total_count, pct_consistent))
    
    model_consistency.sort(key=lambda x: x[3], reverse=True)
    
    for rank, (model, count, total, pct) in enumerate(model_consistency, 1):
        family = MODEL_FAMILIES[model]
        report_lines.append(f"   {rank}. {model:12} ({family:15}): {count}/{total} cases ({pct:.1f}%)")
    
    report_lines.append("")
    
    # 3. Low-data consistency
    report_lines.append("3. LOW-DATA PERFORMANCE ACROSS METRICS (10%, 20%, 30%)")
    report_lines.append("-" * 70)
    
    low_data_engines = [1, 2, 3]
    for n_eng in low_data_engines:
        eng_df = consistency_df[consistency_df['n_engines'] == n_eng]
        all_benefit = eng_df['all_metrics_benefit'].sum()
        total = len(eng_df[eng_df['all_metrics_benefit'].notna()])
        report_lines.append(f"   {pct:3}%: {all_benefit}/{total} models benefit across ALL metrics")
    
    report_lines.append("")
    report_lines.append("="*40)
    
    # Write report
    report_text = "\n".join(report_lines)
    output_path = output_dir / 'phase2b_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f" Saved: {output_path.name}")
    print("\n" + report_text)

# ==========
# MAIN EXECUTION
# ==========

def main():
    print("\n" + "="*40)
    print("PHASE 2B: MULTI-METRIC FEATURE SELECTION ANALYSIS")
    print("="*40)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all results
    df = load_all_results()
    
    # Generate heatmaps for all metrics
    print("\n" + "="*40)
    print("GENERATING IMPROVEMENT HEATMAPS FOR ALL METRICS")
    print("="*40)
    
    all_heatmaps = {}
    for metric_col, metric_info in METRICS.items():
        print(f"\nGenerating heatmap for {metric_info['name']}...")
        heatmap_df = generate_improvement_heatmap_for_metric(df, metric_col, metric_info, OUTPUT_DIR)
        all_heatmaps[metric_col] = heatmap_df
    
    # Analyze cross-metric consistency
    print("\n" + "="*40)
    print("ANALYZING CROSS-METRIC CONSISTENCY")
    print("="*40)
    
    consistency_df = analyze_cross_metric_consistency(df, OUTPUT_DIR)
    
    # Generate summary report
    print("\n" + "="*40)
    print("GENERATING SUMMARY REPORT")
    print("="*40)
    
    generate_phase2b_summary(df, all_heatmaps, consistency_df, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*40)
    print(" PHASE 2B ANALYSIS COMPLETE!")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"5 improvement heatmaps (one per metric)")
    print(f"1 cross-metric consistency CSV")
    print(f"1 comprehensive summary report")
    print("\nNext steps:")
    print("1. Compare findings across metrics")
    print("2. Identify metrics where FS helps most")
    print("3. Document Phase 2 findings for thesis")
    print()

if __name__ == '__main__':
    main()

