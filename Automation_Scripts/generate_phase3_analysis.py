import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PHASE1_DIR = PROJECT_ROOT / 'Results' / 'Phase1_Baseline' / 'FD001'
PHASE2_DIR = PROJECT_ROOT / 'Results' / 'Phase2_Feature_Selection' / 'FD001'
PHASE3_DIR = PROJECT_ROOT / 'Results' / 'Phase3_Feature_Extraction' / 'FD001'
OUTPUT_DIR = PHASE3_DIR / 'Analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model colors (biology-style: 3 color families)
MODEL_COLORS = {
    'LSTM': '#2E86AB',           # Blue family - Baseline
    'LSTM_Correlation_FS': '#A23B72',  # Purple family - Feature Selection
    'LSTM_Tree_FS': '#C73E1D',         # Red family - Feature Selection
    'CNN_LSTM': '#06A77D',       # Green family - Feature Extraction
    'TCN_LSTM': '#2A9D8F',       # Teal family - Feature Extraction
    'AE_LSTM': '#0FA3B1',        # Cyan family - Feature Extraction
}

METRICS = ['rmse', 'mae', 'r2', 'cmapss', 'auc_rmse']
METRIC_LABELS = {
    'rmse': 'RMSE (lower is better)',
    'mae': 'MAE (lower is better)',
    'r2': 'R² (higher is better)',
    'cmapss': 'CMAPSS Score (lower is better)',
    'auc_rmse': 'AUC-RMSE (lower is better)'
}

DATA_PERCENTAGES = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

# ==========
# LOAD DATA
# ==========

def load_phase1_lstm_baseline():
    print("Loading Phase 1 LSTM baseline...")
    lstm_file = PHASE1_DIR / 'LSTM' / 'LSTM_metrics_summary.csv'
    
    if not lstm_file.exists():
        print(f"LSTM baseline not found: {lstm_file}")
        return None
    
    df = pd.read_csv(lstm_file)
    
    # Check for 'split' or 'dataset' column
    if 'split' in df.columns:
        split_col = 'split'
    elif 'dataset' in df.columns:
        split_col = 'dataset'
    else:
        print(f"No 'split' or 'dataset' column found in {lstm_file}")
        return None
    
    # Filter for validation results
    df = df[df[split_col] == 'val'].copy()
    
    # Ensure we have data_pct column
    if 'data_pct' not in df.columns:
        print(f"No 'data_pct' column in {lstm_file}")
        return None
    
    df['model_name'] = 'LSTM'
    df['source'] = 'Phase1_Baseline'
    
    print(f"Loaded {len(df)} validation records")
    return df

def load_phase2_lstm_fs():
    print("\nLoading Phase 2 LSTM Feature Selection results...")
    
    results = []
    fs_methods = ['Correlation_FS', 'Tree_FS']
    
    for fs_method in fs_methods:
        # Correct path: Phase2_Feature_Selection/FD001/{fs_method}/LSTM/LSTM_metrics_summary.csv
        fs_file = PHASE2_DIR / fs_method / 'LSTM' / 'LSTM_metrics_summary.csv'
        
        if not fs_file.exists():
            print(f"{fs_method} not found: {fs_file}")
            continue
        
        df = pd.read_csv(fs_file)
        
        # Phase 2 files have train/val/test metrics in same row
        # We only need the validation metrics, so rename columns
        # Expected columns: val_rmse, val_mae, val_r2, val_cmapss, val_auc_rmse, data_pct
        
        if 'val_rmse' not in df.columns or 'data_pct' not in df.columns:
            print(f"Missing required columns in {fs_file}")
            continue
        
        df['model_name'] = f'LSTM_{fs_method}'
        df['source'] = 'Phase2_FeatureSelection'
        results.append(df)
        print(f"Loaded {len(df)} records for {fs_method}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        print("No Phase 2 Feature Selection results found")
        return None

def load_phase3_feature_extraction():
    print("\nLoading Phase 3 Feature Extraction results...")
    
    results = []
    models = ['CNN_LSTM', 'TCN_LSTM', 'AE_LSTM']
    
    for model in models:
        model_dir = PHASE3_DIR / model
        results_file = model_dir / f'{model.lower()}_all_results.csv'
        
        if not results_file.exists():
            print(f"{model} not found: {results_file}")
            continue
        
        df = pd.read_csv(results_file)
        df['model_name'] = model
        df['source'] = 'Phase3_FeatureExtraction'
        results.append(df)
        print(f"Loaded {len(df)} records for {model}")
    
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        print("No Phase 3 Feature Extraction results found")
        return None

def combine_all_results():
    print("\n" + "="*40)
    print("LOADING ALL RESULTS")
    print("="*40)
    
    # Load all phases
    phase1_df = load_phase1_lstm_baseline()
    phase2_df = load_phase2_lstm_fs()
    phase3_df = load_phase3_feature_extraction()
    
    # Combine
    dfs = []
    if phase1_df is not None:
        dfs.append(phase1_df)
    if phase2_df is not None:
        dfs.append(phase2_df)
    if phase3_df is not None:
        dfs.append(phase3_df)
    
    if not dfs:
        print("\n No data loaded!")
        return None
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n Combined data: {len(combined_df)} records")
    print(f"Models: {combined_df['model_name'].unique().tolist()}")
    print(f"Data percentages: {sorted(combined_df['data_pct'].unique().tolist(), reverse=True)}")
    
    return combined_df

# ==========
# PLOTTING FUNCTIONS
# ==========

def plot_rmse_evolution_all_models(df):
    print("\nGenerating RMSE evolution plot (all models)...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model].sort_values('data_pct', ascending=False)
        
        color = MODEL_COLORS.get(model, '#333333')
        marker = 'o' if 'LSTM' in model else 's'
        linestyle = '-' if model == 'LSTM' else '--' if 'FS' in model else '-.'
        
        ax.plot(model_data['data_pct'], model_data['val_rmse'], 
                marker=marker, linestyle=linestyle, linewidth=2, markersize=8,
                label=model, color=color, alpha=0.8)
    
    ax.set_xlabel('Training Data Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation RMSE', fontsize=14, fontweight='bold')
    ax.set_title('Phase 3: Feature Extraction RMSE Evolution\n(Compared with Baseline & Feature Selection)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'rmse_evolution_all_models.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_metric_comparison(df, metric):
    print(f"\nGenerating {metric.upper()} evolution plot...")
    
    val_col = f'val_{metric}'
    if val_col not in df.columns:
        print(f"Column {val_col} not found, skipping...")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model].sort_values('data_pct', ascending=False)
        
        color = MODEL_COLORS.get(model, '#333333')
        marker = 'o' if 'LSTM' in model else 's'
        linestyle = '-' if model == 'LSTM' else '--' if 'FS' in model else '-.'
        
        ax.plot(model_data['data_pct'], model_data[val_col], 
                marker=marker, linestyle=linestyle, linewidth=2, markersize=8,
                label=model, color=color, alpha=0.8)
    
    ax.set_xlabel('Training Data Percentage (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=14, fontweight='bold')
    ax.set_title(f'Phase 3: {metric.upper()} Evolution Across All Models', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / f'{metric}_evolution_all_models.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_training_time_comparison(df):
    print("\nGenerating training time comparison...")
    
    if 'training_time_sec' not in df.columns:
        print("No training_time_sec column, skipping...")
        return
    
    # Calculate average training time per model
    time_summary = df.groupby('model_name')['training_time_sec'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [MODEL_COLORS.get(model, '#333333') for model in time_summary.index]
    bars = ax.barh(range(len(time_summary)), time_summary.values, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(time_summary)))
    ax.set_yticklabels(time_summary.index, fontsize=12)
    ax.set_xlabel('Average Training Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Average Training Time per Model (All Data Percentages)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, time_summary.values)):
        ax.text(val + 5, i, f'{val:.1f}s', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'training_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_low_data_comparison(df):
    print("\nGenerating low-data scenario comparison...")
    
    low_data_pcts = [10, 20, 30]
    low_data_df = df[df['data_pct'].isin(low_data_pcts)].copy()
    
    if low_data_df.empty:
        print("No data for low percentages, skipping...")
        return
    
    # Calculate improvement vs baseline LSTM
    baseline_df = df[df['model_name'] == 'LSTM'].copy()
    
    improvements = []
    for pct in low_data_pcts:
        baseline_rmse = baseline_df[baseline_df['data_pct'] == pct]['val_rmse'].values
        if len(baseline_rmse) == 0:
            continue
        baseline_rmse = baseline_rmse[0]
        
        for model in low_data_df['model_name'].unique():
            if model == 'LSTM':
                continue
            
            model_data = low_data_df[(low_data_df['model_name'] == model) & 
                                     (low_data_df['data_pct'] == pct)]
            if model_data.empty:
                continue
            
            model_rmse = model_data['val_rmse'].values[0]
            improvement_pct = ((baseline_rmse - model_rmse) / baseline_rmse) * 100
            
            improvements.append({
                'model': model,
                'data_pct': pct,
                'improvement_pct': improvement_pct
            })
    
    if not improvements:
        print("No improvements calculated, skipping...")
        return
    
    imp_df = pd.DataFrame(improvements)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = imp_df['model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    colors_by_pct = {10: '#E63946', 20: '#F77F00', 30: '#06A77D'}
    
    for i, pct in enumerate(low_data_pcts):
        pct_data = imp_df[imp_df['data_pct'] == pct].set_index('model')
        values = [pct_data.loc[model, 'improvement_pct'] if model in pct_data.index else 0 
                  for model in models]
        
        ax.bar(x + i*width, values, width, label=f'{pct}% Data', 
               color=colors_by_pct[pct], alpha=0.8)
    
    ax.set_ylabel('RMSE Improvement vs Baseline LSTM (%)', fontsize=14, fontweight='bold')
    ax.set_title('Low-Data Performance: Improvement Over Baseline LSTM\n(Positive = Better than baseline)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=11)
    ax.legend(loc='best', fontsize=12, framealpha=0.9, title='Data Percentage')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'low_data_performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

def plot_all_metrics_heatmap(df):
    print("\nGenerating all-metrics heatmap...")
    
    # Focus on key data percentages
    key_pcts = [100, 70, 50, 30, 10]
    df_subset = df[df['data_pct'].isin(key_pcts)].copy()
    
    if df_subset.empty:
        print("No data for key percentages, skipping...")
        return
    
    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 6))
    
    for idx, metric in enumerate(METRICS):
        val_col = f'val_{metric}'
        if val_col not in df_subset.columns:
            continue
        
        # Pivot for heatmap
        pivot_df = df_subset.pivot_table(
            values=val_col, 
            index='model_name', 
            columns='data_pct'
        )
        
        # Sort columns descending
        pivot_df = pivot_df[sorted(pivot_df.columns, reverse=True)]
        
        # Choose colormap (reversed for "lower is better" metrics)
        if metric in ['rmse', 'mae', 'cmapss', 'auc_rmse']:
            cmap = 'RdYlGn_r'  # Red (bad) to Green (good), reversed
        else:
            cmap = 'RdYlGn'    # Red (bad) to Green (good)
        
        sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap=cmap, 
                    ax=axes[idx], cbar_kws={'label': METRIC_LABELS[metric]},
                    linewidths=0.5, linecolor='gray')
        
        axes[idx].set_title(metric.upper(), fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Data %', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Model' if idx == 0 else '', fontsize=11, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=0)
    
    plt.suptitle('Phase 3: All Metrics Heatmap (Key Data Percentages)', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'all_metrics_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")

# ==========
# SUMMARY STATISTICS
# ==========

def generate_summary_statistics(df):
    print("\nGenerating summary statistics...")
    
    summary_stats = []
    
    for model in df['model_name'].unique():
        model_data = df[df['model_name'] == model]
        
        stats = {
            'Model': model,
            'Source': model_data['source'].iloc[0],
            'Avg_Val_RMSE': model_data['val_rmse'].mean(),
            'Avg_Val_MAE': model_data['val_mae'].mean(),
            'Avg_Val_R2': model_data['val_r2'].mean(),
            'Avg_Val_CMAPSS': model_data['val_cmapss'].mean(),
            'Avg_Val_AUC_RMSE': model_data['val_auc_rmse'].mean(),
            'RMSE_at_100pct': model_data[model_data['data_pct'] == 100]['val_rmse'].values[0] if len(model_data[model_data['data_pct'] == 100]) > 0 else np.nan,
            'RMSE_at_10pct': model_data[model_data['data_pct'] == 10]['val_rmse'].values[0] if len(model_data[model_data['data_pct'] == 10]) > 0 else np.nan,
        }
        
        # Calculate degradation
        if not np.isnan(stats['RMSE_at_100pct']) and not np.isnan(stats['RMSE_at_10pct']):
            stats['RMSE_Degradation_100to10'] = ((stats['RMSE_at_10pct'] - stats['RMSE_at_100pct']) / stats['RMSE_at_100pct']) * 100
        else:
            stats['RMSE_Degradation_100to10'] = np.nan
        
        # Average training time
        if 'training_time_sec' in model_data.columns:
            stats['Avg_Training_Time_sec'] = model_data['training_time_sec'].mean()
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('Avg_Val_RMSE')
    
    # Save
    output_file = OUTPUT_DIR / 'summary_statistics.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    # Print to console
    print("\n" + "="*40)
    print("SUMMARY STATISTICS")
    print("="*40)
    print(summary_df.to_string(index=False))
    print("="*40)
    
    return summary_df

def identify_best_model(df):
    print("\n" + "="*40)
    print("BEST MODEL IDENTIFICATION")
    print("="*40)
    
    # Average validation RMSE across all percentages
    avg_rmse = df.groupby('model_name')['val_rmse'].mean().sort_values()
    
    print("\nAverage Validation RMSE (across all data percentages):")
    for i, (model, rmse) in enumerate(avg_rmse.items(), 1):
        print(f"{i}. {model:25s}  {rmse:.4f}")
    
    best_model = avg_rmse.index[0]
    best_rmse = avg_rmse.values[0]
    
    print(f"\n BEST OVERALL MODEL: {best_model} (Avg RMSE: {best_rmse:.4f})")
    print("="*40)
    
    return best_model

# ==========
# MAIN ANALYSIS
# ==========

def main():
    print("\n" + "="*40)
    print("PHASE 3: FEATURE EXTRACTION ANALYSIS")
    print("="*40)
    
    # Load all data
    df = combine_all_results()
    
    if df is None or df.empty:
        print("\n No data to analyze!")
        return
    
    # Generate plots
    print("\n" + "="*40)
    print("GENERATING PLOTS")
    print("="*40)
    
    plot_rmse_evolution_all_models(df)
    
    for metric in METRICS:
        plot_metric_comparison(df, metric)
    
    plot_training_time_comparison(df)
    plot_low_data_comparison(df)
    plot_all_metrics_heatmap(df)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(df)
    
    # Identify best model
    best_model = identify_best_model(df)
    
    # Save combined data
    combined_file = OUTPUT_DIR / 'combined_all_phases.csv'
    df.to_csv(combined_file, index=False)
    print(f"\n Combined data saved: {combined_file}")
    
    print("\n" + "="*40)
    print(" PHASE 3 ANALYSIS COMPLETE!")
    print("="*40)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("rmse_evolution_all_models.png")
    print("mae_evolution_all_models.png")
    print("r2_evolution_all_models.png")
    print("cmapss_evolution_all_models.png")
    print("auc_rmse_evolution_all_models.png")
    print("training_time_comparison.png")
    print("low_data_performance_comparison.png")
    print("all_metrics_heatmap.png")
    print("summary_statistics.csv")
    print("combined_all_phases.csv")
    print()

if __name__ == '__main__':
    main()

