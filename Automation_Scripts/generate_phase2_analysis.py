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

ENGINE_COUNTS = ENGINE_COUNTS_ALL  # [80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

PHASE1_DIR = PROJECT_ROOT / 'Results' / 'Phase1_Baseline' / 'FD001'
PHASE2_DIR = PROJECT_ROOT / 'Results' / 'Phase2_Feature_Selection' / 'FD001'
OUTPUT_DIR = PHASE2_DIR / 'Analysis'

# ==========
# DATA LOADING
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
    
    # Get validation RMSE (from val split)
    val_df = df[df[split_col] == 'val'][['model_name', 'n_engines', 'rmse']].copy()
    val_df.rename(columns={'rmse': 'val_rmse'}, inplace=True)
    
    # Get training time (from train split)
    train_df = df[df[split_col] == 'train'][['model_name', 'n_engines', 'training_time_sec']].copy()
    
    # Merge validation RMSE with training time
    result_df = pd.merge(val_df, train_df, on=['model_name', 'n_engines'], how='left')
    
    # Add method column
    result_df['method'] = 'Baseline'
    
    # Select relevant columns in correct order
    result_df = result_df[['model_name', 'n_engines', 'method', 'val_rmse', 'training_time_sec']]
    
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
    
    # Select relevant columns (Phase 2 uses n_engines)
    df = df[['model_name', 'n_engines', 'method', 'val_rmse', 'training_time_sec']]
    
    return df

def load_all_results():
    print("="*40)
    print("LOADING PHASE 1 & PHASE 2 RESULTS")
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
    print(f" Engine counts: {sorted(combined_df['n_engines'].unique(), reverse=True)}")
    
    return combined_df

# ==========
# PLOTTING FUNCTIONS
# ==========

def plot_individual_rmse_evolution(df, model_name, output_dir):
    model_df = df[df['model_name'] == model_name].copy()
    model_family = MODEL_FAMILIES[model_name]
    color = FAMILY_COLORS[model_family]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot lines for each method
    line_styles = {'Baseline': '-', 'Correlation_FS': '--', 'Tree_FS': ':'}
    line_widths = {'Baseline': 2.5, 'Correlation_FS': 2.5, 'Tree_FS': 2.5}
    
    for method in ['Baseline', 'Correlation_FS', 'Tree_FS']:
        method_df = model_df[model_df['method'] == method].sort_values('n_engines', ascending=False)
        
        if len(method_df) > 0:
            ax.plot(method_df['n_engines'], method_df['val_rmse'],
                   color=color, linestyle=line_styles[method],
                   linewidth=line_widths[method], marker='o', markersize=6,
                   label=method, alpha=0.9)
    
    # Styling
    ax.set_xlabel('Number of Training Engines', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation RMSE (Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_title(f'{model_name} - RMSE Evolution Across Data Scarcity',
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(105, 5)  # Reverse x-axis
    ax.set_ylim(10, 25)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    
    # Add annotation for feature counts
    feature_counts = {
        'Baseline': 24,
        'Correlation_FS': 14,
        'Tree_FS': 12
    }
    
    y_pos = 24
    for method, count in feature_counts.items():
        ax.text(8, y_pos, f'{method}: {count} features',
               fontsize=10, style='italic', alpha=0.7)
        y_pos -= 0.8
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / f'{model_name}_rmse_evolution.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Saved: {output_path.name}")

def plot_improvement_heatmap(df, output_dir):
    print("\nGenerating improvement heatmap...")
    
    # Prepare data for heatmap
    improvement_data = []
    
    for model in SELECTED_MODELS:
        model_improvements = []
        
        for n_eng in ENGINE_COUNTS:
            # Get baseline RMSE
            baseline_rmse = df[(df['model_name'] == model) & 
                              (df['n_engines'] == n_eng) & 
                              (df['method'] == 'Baseline')]['val_rmse'].values
            
            if len(baseline_rmse) == 0:
                model_improvements.append(np.nan)
                continue
            
            baseline_rmse = baseline_rmse[0]
            
            # Get best FS RMSE
            fs_rmses = df[(df['model_name'] == model) & 
                         (df['n_engines'] == n_eng) & 
                         (df['method'].isin(['Correlation_FS', 'Tree_FS']))]['val_rmse'].values
            
            if len(fs_rmses) == 0:
                model_improvements.append(np.nan)
                continue
            
            best_fs_rmse = min(fs_rmses)
            
            # Calculate % improvement (positive = better)
            improvement_percent = ((baseline_rmse - best_fs_rmse) / baseline_rmse) * 100
            model_improvements.append(improvement_percent)
        
        improvement_data.append(model_improvements)
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(improvement_data,
                             index=SELECTED_MODELS,
                             columns=[f'{n}eng' for n in ENGINE_COUNTS])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use RdYlGn colormap: red for negative (worse), green for positive (better)
    sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='RdYlGn',
                center=0, vmin=-10, vmax=10,
                cbar_kws={'label': '% Improvement from Baseline'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title('RMSE Improvement from Baseline (%)\n(Positive = Better with Feature Selection)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Number of Training Engines', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    
    # Add family separators
    regression_count = sum(1 for m in SELECTED_MODELS if MODEL_FAMILIES[m] == 'Regression')
    tree_count = sum(1 for m in SELECTED_MODELS if MODEL_FAMILIES[m] == 'Tree')
    
    ax.axhline(y=regression_count, color='black', linewidth=2)
    ax.axhline(y=regression_count + tree_count, color='black', linewidth=2)
    
    plt.tight_layout()
    
    output_path = output_dir / 'rmse_improvement_heatmap.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f" Saved: {output_path.name}")
    
    return heatmap_df

def analyze_low_data_performance(df, heatmap_df, output_dir):
    print("\nAnalyzing low-data performance (1, 2, 3 engines)...")
    
    low_data_engines = [1, 2, 3]
    
    # Generate detailed table
    report_lines = []
    report_lines.append("\n" + "="*40)
    report_lines.append("LOW-DATA SCENARIO ANALYSIS (10%, 20%, 30%)")
    report_lines.append("="*40)
    report_lines.append("")
    report_lines.append("Who benefits from FS when data is scarce?")
    report_lines.append("")
    
    for n_eng in low_data_engines:
        report_lines.append(f"\n{'='*40}")
        report_lines.append(f"ENGINE COUNT: {n_eng} engines")
        report_lines.append(f"{'='*40}")
        
        # Get improvements for this engine count
        improvements = []
        for model in SELECTED_MODELS:
            improv = heatmap_df.loc[model, f'{n_eng}eng']
            if pd.notna(improv):
                improvements.append((model, improv))
        
        # Sort by improvement (best first)
        improvements.sort(key=lambda x: x[1], reverse=True)
        
        # Separate winners and losers
        winners = [x for x in improvements if x[1] > 0]
        losers = [x for x in improvements if x[1] <= 0]
        
        report_lines.append(f"\n Models that BENEFIT from FS ({len(winners)}/{len(improvements)}):")
        report_lines.append("-" * 70)
        
        if winners:
            for rank, (model, improv) in enumerate(winners, 1):
                family = MODEL_FAMILIES[model]
                
                # Get actual RMSE values
                baseline_rmse = df[(df['model_name'] == model) & 
                                  (df['n_engines'] == n_eng) & 
                                  (df['method'] == 'Baseline')]['val_rmse'].values[0]
                
                fs_rmses = df[(df['model_name'] == model) & 
                             (df['n_engines'] == n_eng) & 
                             (df['method'].isin(['Correlation_FS', 'Tree_FS']))]
                best_fs_rmse = fs_rmses['val_rmse'].min()
                best_fs_method = fs_rmses.loc[fs_rmses['val_rmse'].idxmin(), 'method']
                
                report_lines.append(f"   {rank}. {model:12} ({family:15}): {improv:+6.2f}%")
                report_lines.append(f"      Baseline RMSE: {baseline_rmse:6.2f}  Best FS: {best_fs_rmse:6.2f} ({best_fs_method})")
        else:
            report_lines.append("   None - all models degrade with FS at this engine count")
        
        report_lines.append(f"\n Models that DEGRADE with FS ({len(losers)}/{len(improvements)}):")
        report_lines.append("-" * 70)
        
        if losers:
            for rank, (model, improv) in enumerate(losers, 1):
                family = MODEL_FAMILIES[model]
                
                # Get actual RMSE values
                baseline_rmse = df[(df['model_name'] == model) & 
                                  (df['n_engines'] == n_eng) & 
                                  (df['method'] == 'Baseline')]['val_rmse'].values[0]
                
                fs_rmses = df[(df['model_name'] == model) & 
                             (df['n_engines'] == n_eng) & 
                             (df['method'].isin(['Correlation_FS', 'Tree_FS']))]
                best_fs_rmse = fs_rmses['val_rmse'].min()
                best_fs_method = fs_rmses.loc[fs_rmses['val_rmse'].idxmin(), 'method']
                
                report_lines.append(f"   {rank}. {model:12} ({family:15}): {improv:+6.2f}%")
                report_lines.append(f"      Baseline RMSE: {baseline_rmse:6.2f}  Best FS: {best_fs_rmse:6.2f} ({best_fs_method})")
    
    # Summary insights
    report_lines.append(f"\n{'='*40}")
    report_lines.append("KEY INSIGHTS FROM LOW-DATA ANALYSIS")
    report_lines.append(f"{'='*40}")
    report_lines.append("")
    
    # Count winners at each engine count
    for n_eng in low_data_engines:
        winners_count = sum(1 for m in SELECTED_MODELS if heatmap_df.loc[m, f'{n_eng}eng'] > 0)
        report_lines.append(f" {n_eng} engines: {winners_count}/7 models benefit from FS")
    
    report_lines.append("")
    
    # Find consistent performers
    consistent_winners = []
    consistent_losers = []
    
    for model in SELECTED_MODELS:
        low_data_scores = [heatmap_df.loc[model, f'{n}eng'] for n in low_data_engines if pd.notna(heatmap_df.loc[model, f'{n}eng'])]
        
        if all(score > 0 for score in low_data_scores):
            avg_improvement = np.mean(low_data_scores)
            consistent_winners.append((model, avg_improvement))
        elif all(score < 0 for score in low_data_scores):
            avg_degradation = np.mean(low_data_scores)
            consistent_losers.append((model, avg_degradation))
    
    if consistent_winners:
        report_lines.append("CONSISTENT WINNERS (benefit at ALL low engine counts):")
        consistent_winners.sort(key=lambda x: x[1], reverse=True)
        for model, avg_improv in consistent_winners:
            report_lines.append(f"   {model:12} ({MODEL_FAMILIES[model]:15}): {avg_improv:+6.2f}% average")
        report_lines.append("")
    
    if consistent_losers:
        report_lines.append("CONSISTENT LOSERS (degrade at ALL low engine counts):")
        consistent_losers.sort(key=lambda x: x[1])
        for model, avg_degrad in consistent_losers:
            report_lines.append(f"   {model:12} ({MODEL_FAMILIES[model]:15}): {avg_degrad:+6.2f}% average")
        report_lines.append("")
    
    report_lines.append("="*40)
    
    # Create visualization: Low-data focused plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.arange(len(SELECTED_MODELS))
    width = 0.25
    
    # Use distinct colors for each percentage, not model families
    eng_colors = ['#e74c3c', '#f39c12', '#3498db']  # Red, Orange, Blue for 1, 2, 3 engines
    
    for i, n_eng in enumerate(low_data_engines):
        improvements = [heatmap_df.loc[m, f'{n_eng}eng'] for m in SELECTED_MODELS]
        ax.bar(x + (i - 1) * width, improvements, width, 
               label=f'{n_eng} engines', alpha=0.8, color=eng_colors[i])
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('RMSE Improvement (%) - Positive = Better', fontsize=12, fontweight='bold')
    ax.set_title('Feature Selection Impact at Low Data Percentages\n(3 bars per model: 10%, 20%, 30%)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(SELECTED_MODELS, rotation=45, ha='right')
    
    # Add model family labels as text annotations
    y_pos = ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.15
    for i, model in enumerate(SELECTED_MODELS):
        family = MODEL_FAMILIES[model]
        ax.text(i, y_pos, family[0], ha='center', fontsize=9, 
               style='italic', color=FAMILY_COLORS[family])
    
    ax.legend(fontsize=11, title='Training Data %', loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'low_data_performance_analysis.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f" Saved: {output_path.name}")
    
    return "\n".join(report_lines)

# ==========
# ANALYSIS FUNCTIONS
# ==========

def generate_summary_statistics(df, heatmap_df, output_dir, training_time_report, low_data_report):
    print("\nGenerating summary statistics...")
    
    report_lines = []
    report_lines.append("="*40)
    report_lines.append("PHASE 2A: RMSE ANALYSIS SUMMARY")
    report_lines.append("="*40)
    report_lines.append("")
    
    # 1. Which models benefit most from FS?
    report_lines.append("1. MODELS RANKED BY FS BENEFIT (Average % Improvement)")
    report_lines.append("-" * 70)
    
    avg_improvements = heatmap_df.mean(axis=1).sort_values(ascending=False)
    
    for i, (model, improvement) in enumerate(avg_improvements.items(), 1):
        family = MODEL_FAMILIES[model]
        report_lines.append(f"   {i}. {model:12} ({family:15}): {improvement:6.2f}% improvement")
    
    report_lines.append("")
    
    # 2. At which data percentages does FS help most?
    report_lines.append("2. DATA PERCENTAGES RANKED BY FS BENEFIT")
    report_lines.append("-" * 70)
    
    avg_by_engines = heatmap_df.mean(axis=0).sort_values(ascending=False)
    
    for i, (eng_label, improvement) in enumerate(avg_by_engines.items(), 1):
        report_lines.append(f"   {i}. {eng_label:6}: {improvement:6.2f}% average improvement")
    
    report_lines.append("")
    
    # 3. Best FS method per model
    report_lines.append("3. BEST FEATURE SELECTION METHOD PER MODEL")
    report_lines.append("-" * 70)
    
    for model in SELECTED_MODELS:
        model_df = df[df['model_name'] == model]
        
        corr_avg = model_df[model_df['method'] == 'Correlation_FS']['val_rmse'].mean()
        tree_avg = model_df[model_df['method'] == 'Tree_FS']['val_rmse'].mean()
        
        if pd.notna(corr_avg) and pd.notna(tree_avg):
            if corr_avg < tree_avg:
                best_method = f"Correlation_FS (RMSE: {corr_avg:.2f})"
                diff = ((tree_avg - corr_avg) / tree_avg) * 100
            else:
                best_method = f"Tree_FS (RMSE: {tree_avg:.2f})"
                diff = ((corr_avg - tree_avg) / corr_avg) * 100
            
            report_lines.append(f"   {model:12}: {best_method:40} [{diff:.1f}% better]")
    
    report_lines.append("")
    
    # 4. Break-even point analysis
    report_lines.append("4. BREAK-EVEN ANALYSIS (When does FS become critical?)")
    report_lines.append("-" * 70)
    report_lines.append("   Threshold: >5% improvement indicates FS is beneficial")
    report_lines.append("")
    
    for n_eng in ENGINE_COUNTS:
        models_benefiting = (heatmap_df[f'{n_eng}eng'] > 5).sum()
        avg_improvement = heatmap_df[f'{n_eng}eng'].mean()
        
        status = " FS Critical" if avg_improvement > 5 else "  FS Optional"
        report_lines.append(f"   {n_eng:2} engines: {status} | {models_benefiting}/7 models benefit | Avg: {avg_improvement:5.2f}%")
    
    report_lines.append("")
    
    # 5. Overall findings
    report_lines.append("5. KEY FINDINGS")
    report_lines.append("-" * 70)
    
    # Find break-even point
    breakeven_engines = None
    for n_eng in sorted(ENGINE_COUNTS, reverse=True):
        if heatmap_df[f'{n_eng}eng'].mean() > 5:
            breakeven_engines = n_eng
            break
    
    if breakeven_engines:
        report_lines.append(f"    Feature Selection becomes critical below {breakeven_engines} training engines")
    
    best_model = avg_improvements.idxmax()
    best_improvement = avg_improvements.max()
    report_lines.append(f"    {best_model} benefits most from FS ({best_improvement:.1f}% avg improvement)")
    
    worst_model = avg_improvements.idxmin()
    worst_improvement = avg_improvements.min()
    report_lines.append(f"    {worst_model} benefits least from FS ({worst_improvement:.1f}% avg improvement)")
    
    best_engines = avg_by_engines.idxmax()
    report_lines.append(f"    FS helps most at {best_engines} ({avg_by_engines.max():.1f}% avg improvement)")
    
    report_lines.append("")
    report_lines.append("="*40)
    
    # Add training time analysis
    report_lines.append(training_time_report)
    
    # Add low-data analysis
    report_lines.append(low_data_report)
    
    # Write report
    report_text = "\n".join(report_lines)
    output_path = output_dir / 'phase2a_rmse_summary.txt'
    
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f" Saved: {output_path.name}")
    
    # Also print to console
    print("\n" + report_text)

def analyze_training_times(df, output_dir):
    print("\nAnalyzing training times...")
    
    # Calculate average training times per model and method
    time_summary = df.groupby(['model_name', 'method'])['training_time_sec'].mean().reset_index()
    time_pivot = time_summary.pivot(index='model_name', columns='method', values='training_time_sec')
    
    # Calculate speedup percentages
    time_pivot['corr_fs_speedup_percent'] = ((time_pivot['Baseline'] - time_pivot['Correlation_FS']) / time_pivot['Baseline']) * 100
    time_pivot['tree_fs_speedup_percent'] = ((time_pivot['Baseline'] - time_pivot['Tree_FS']) / time_pivot['Baseline']) * 100
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Absolute training times
    ax1 = axes[0]
    x = np.arange(len(SELECTED_MODELS))
    width = 0.25
    
    baseline_times = [time_pivot.loc[m, 'Baseline'] if m in time_pivot.index else 0 for m in SELECTED_MODELS]
    corr_times = [time_pivot.loc[m, 'Correlation_FS'] if m in time_pivot.index else 0 for m in SELECTED_MODELS]
    tree_times = [time_pivot.loc[m, 'Tree_FS'] if m in time_pivot.index else 0 for m in SELECTED_MODELS]
    
    colors = [FAMILY_COLORS[MODEL_FAMILIES[m]] for m in SELECTED_MODELS]
    
    bars1 = ax1.bar(x - width, baseline_times, width, label='Baseline (24 feat)', alpha=0.8, color=colors)
    bars2 = ax1.bar(x, corr_times, width, label='Correlation_FS (14 feat)', alpha=0.8, color=colors, hatch='//')
    bars3 = ax1.bar(x + width, tree_times, width, label='Tree_FS (12 feat)', alpha=0.8, color=colors, hatch='\\\\')
    
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Training Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Training Time Comparison\n(Average across all data percentages)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(SELECTED_MODELS, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Speedup percentages
    ax2 = axes[1]
    
    corr_speedups = [time_pivot.loc[m, 'corr_fs_speedup_percent'] if m in time_pivot.index else 0 for m in SELECTED_MODELS]
    tree_speedups = [time_pivot.loc[m, 'tree_fs_speedup_percent'] if m in time_pivot.index else 0 for m in SELECTED_MODELS]
    
    bars1 = ax2.bar(x - width/2, corr_speedups, width, label='Correlation_FS', alpha=0.8, color=colors)
    bars2 = ax2.bar(x + width/2, tree_speedups, width, label='Tree_FS', alpha=0.8, color=colors, hatch='\\\\')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time Reduction (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training Time Speedup with Feature Selection\n(Positive = Faster)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(SELECTED_MODELS, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'training_time_analysis.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f" Saved: {output_path.name}")
    
    # Generate text report
    report_lines = []
    report_lines.append("\n" + "="*40)
    report_lines.append("TRAINING TIME ANALYSIS")
    report_lines.append("="*40)
    report_lines.append("")
    report_lines.append("Average Training Time per Method (seconds):")
    report_lines.append("-" * 70)
    
    for model in SELECTED_MODELS:
        if model in time_pivot.index:
            baseline_t = time_pivot.loc[model, 'Baseline']
            corr_t = time_pivot.loc[model, 'Correlation_FS']
            tree_t = time_pivot.loc[model, 'Tree_FS']
            corr_speedup = time_pivot.loc[model, 'corr_fs_speedup_percent']
            tree_speedup = time_pivot.loc[model, 'tree_fs_speedup_percent']
            
            report_lines.append(f"\n{model} ({MODEL_FAMILIES[model]}):")
            report_lines.append(f"  Baseline (24 feat):      {baseline_t:8.2f}s")
            report_lines.append(f"  Correlation_FS (14 feat): {corr_t:8.2f}s  [{corr_speedup:+6.1f}% time]")
            report_lines.append(f"  Tree_FS (12 feat):        {tree_t:8.2f}s  [{tree_speedup:+6.1f}% time]")
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("\nTIME vs ACCURACY TRADE-OFF:")
    report_lines.append("-" * 70)
    
    # Calculate correlation between time saved and RMSE improvement
    for model in SELECTED_MODELS:
        if model in time_pivot.index:
            # Get average RMSE improvement for this model
            model_df = df[df['model_name'] == model]
            
            baseline_rmse = model_df[model_df['method'] == 'Baseline']['val_rmse'].mean()
            corr_rmse = model_df[model_df['method'] == 'Correlation_FS']['val_rmse'].mean()
            tree_rmse = model_df[model_df['method'] == 'Tree_FS']['val_rmse'].mean()
            
            corr_rmse_improv = ((baseline_rmse - corr_rmse) / baseline_rmse) * 100
            tree_rmse_improv = ((baseline_rmse - tree_rmse) / baseline_rmse) * 100
            
            corr_speedup = time_pivot.loc[model, 'corr_fs_speedup_percent']
            tree_speedup = time_pivot.loc[model, 'tree_fs_speedup_percent']
            
            report_lines.append(f"\n{model}:")
            report_lines.append(f"  Correlation_FS: {corr_speedup:+6.1f}% time | {corr_rmse_improv:+6.2f}% RMSE")
            report_lines.append(f"  Tree_FS:        {tree_speedup:+6.1f}% time | {tree_rmse_improv:+6.2f}% RMSE")
    
    report_lines.append("\n" + "="*40)
    
    return "\n".join(report_lines)

def save_master_csv(df, output_dir):
    print("\nGenerating master comparison CSV...")
    
    # Pivot data to get baseline, corr_fs, and tree_fs in columns
    comparison_rows = []
    
    for model in SELECTED_MODELS:
        for n_eng in ENGINE_COUNTS:
            row = {
                'model_name': model,
                'model_family': MODEL_FAMILIES[model],
                'n_engines': n_eng
            }
            
            # Get baseline values
            baseline = df[(df['model_name'] == model) & 
                         (df['n_engines'] == n_eng) & 
                         (df['method'] == 'Baseline')]
            
            if len(baseline) > 0:
                row['baseline_val_rmse'] = baseline['val_rmse'].values[0]
                row['training_time_baseline'] = baseline['training_time_sec'].values[0]
            else:
                row['baseline_val_rmse'] = np.nan
                row['training_time_baseline'] = np.nan
            
            # Get Correlation_FS values
            corr_fs = df[(df['model_name'] == model) & 
                        (df['n_engines'] == n_eng) & 
                        (df['method'] == 'Correlation_FS')]
            
            if len(corr_fs) > 0:
                row['corr_fs_val_rmse'] = corr_fs['val_rmse'].values[0]
                row['training_time_corr_fs'] = corr_fs['training_time_sec'].values[0]
            else:
                row['corr_fs_val_rmse'] = np.nan
                row['training_time_corr_fs'] = np.nan
            
            # Get Tree_FS values
            tree_fs = df[(df['model_name'] == model) & 
                        (df['n_engines'] == n_eng) & 
                        (df['method'] == 'Tree_FS')]
            
            if len(tree_fs) > 0:
                row['tree_fs_val_rmse'] = tree_fs['val_rmse'].values[0]
                row['training_time_tree_fs'] = tree_fs['training_time_sec'].values[0]
            else:
                row['tree_fs_val_rmse'] = np.nan
                row['training_time_tree_fs'] = np.nan
            
            # Determine best method
            rmse_values = {
                'Baseline': row.get('baseline_val_rmse', np.inf),
                'Correlation_FS': row.get('corr_fs_val_rmse', np.inf),
                'Tree_FS': row.get('tree_fs_val_rmse', np.inf)
            }
            
            row['best_method'] = min(rmse_values, key=rmse_values.get)
            row['best_val_rmse'] = rmse_values[row['best_method']]
            
            # Calculate improvement
            if pd.notna(row['baseline_val_rmse']) and pd.notna(row['best_val_rmse']):
                row['improvement_percent'] = ((row['baseline_val_rmse'] - row['best_val_rmse']) / 
                                         row['baseline_val_rmse']) * 100
            else:
                row['improvement_percent'] = np.nan
            
            comparison_rows.append(row)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Save CSV
    output_path = output_dir / 'phase2a_rmse_comparison.csv'
    comparison_df.to_csv(output_path, index=False)
    
    print(f" Saved: {output_path.name}")
    print(f"Total rows: {len(comparison_df)}")

# ==========
# MAIN EXECUTION
# ==========

def main():
    print("\n" + "="*40)
    print("PHASE 2A: RMSE-FOCUSED FEATURE SELECTION ANALYSIS")
    print("="*40)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Create output directories
    rmse_plots_dir = OUTPUT_DIR / 'RMSE_Plots'
    rmse_plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load all results
    df = load_all_results()
    
    # Step 2: Generate individual model plots
    print("\n" + "="*40)
    print("GENERATING INDIVIDUAL MODEL RMSE PLOTS")
    print("="*40)
    
    for model in SELECTED_MODELS:
        print(f"\nPlotting {model}...")
        plot_individual_rmse_evolution(df, model, rmse_plots_dir)
    
    # Step 3: Generate improvement heatmap
    print("\n" + "="*40)
    print("GENERATING IMPROVEMENT HEATMAP")
    print("="*40)
    
    heatmap_df = plot_improvement_heatmap(df, OUTPUT_DIR)
    
    # Step 4: Analyze training times
    print("\n" + "="*40)
    print("ANALYZING TRAINING TIMES")
    print("="*40)
    
    training_time_report = analyze_training_times(df, OUTPUT_DIR)
    
    # Step 5: Analyze low-data performance
    print("\n" + "="*40)
    print("ANALYZING LOW-DATA PERFORMANCE")
    print("="*40)
    
    low_data_report = analyze_low_data_performance(df, heatmap_df, OUTPUT_DIR)
    
    # Step 6: Generate summary statistics
    print("\n" + "="*40)
    print("GENERATING SUMMARY STATISTICS")
    print("="*40)
    
    generate_summary_statistics(df, heatmap_df, OUTPUT_DIR, training_time_report, low_data_report)
    
    # Step 7: Save master CSV
    print("\n" + "="*40)
    print("SAVING MASTER COMPARISON CSV")
    print("="*40)
    
    save_master_csv(df, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*40)
    print(" PHASE 2A ANALYSIS COMPLETE!")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
