import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# ==========
# CONFIGURATION
# ==========

RESULTS_DIR = project_root / "Results" / "Phase1_Baseline" / "FD001"
OUTPUT_DIR = RESULTS_DIR / "Model_Selection"
OUTPUT_DIR.mkdir(exist_ok=True)

# Deep Learning models
DL_MODELS = ['ANN', 'BiLSTM', 'CNN', 'GRU', 'LSTM', 'RNN', 'TCN', 'Transformer']

# ==========
# LOAD DATA
# ==========

def load_all_results():
    print("="*40)
    print("RANK-BASED MODEL SELECTION (Borda Count Method)")
    print("="*40)
    print("\nLoading Phase 1 baseline results...")
    
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
    if 'cmapss_score' in regression_df.columns:
        regression_df.rename(columns={'cmapss_score': 'cmapss'}, inplace=True)
    if 'cmapss_score' in tree_df.columns:
        tree_df.rename(columns={'cmapss_score': 'cmapss'}, inplace=True)
    if 'cmapss_score' in dl_df.columns:
        dl_df.rename(columns={'cmapss_score': 'cmapss'}, inplace=True)
    
    # Add family labels
    regression_df['family'] = 'Regression'
    tree_df['family'] = 'Tree-Based'
    dl_df['family'] = 'Deep Learning'
    
    # Combine all data
    all_data = pd.concat([regression_df, tree_df, dl_df], ignore_index=True)
    
    return all_data

# ==========
# RANKING FUNCTIONS
# ==========

def calculate_average_metrics(df):
    print("\nCalculating average metrics across all data percentages...")
    
    # Filter to validation set only
    val_data = df[df['split'] == 'val'].copy()
    
    # Group by model and calculate mean metrics
    avg_metrics = val_data.groupby(['model_name', 'family']).agg({
        'rmse': 'mean',
        'mae': 'mean',
        'r2': 'mean',
        'cmapss': 'mean',
        'auc_rmse': 'mean'
    }).reset_index()
    
    print(f"Calculated averages for {len(avg_metrics)} models")
    
    return avg_metrics

def rank_models(avg_metrics_df):
    print("\nRanking models for each metric...")
    
    ranks_df = avg_metrics_df[['model_name', 'family']].copy()
    
    # Rank each metric (1 = worst, 17 = best)
    # For lower-is-better metrics: rank descending (lowest value gets highest rank)
    ranks_df['rmse_rank'] = avg_metrics_df['rmse'].rank(method='average', ascending=False)
    ranks_df['mae_rank'] = avg_metrics_df['mae'].rank(method='average', ascending=False)
    ranks_df['cmapss_rank'] = avg_metrics_df['cmapss'].rank(method='average', ascending=False)
    ranks_df['auc_rmse_rank'] = avg_metrics_df['auc_rmse'].rank(method='average', ascending=False)
    
    # For higher-is-better metrics: rank ascending (highest value gets highest rank)
    ranks_df['r2_rank'] = avg_metrics_df['r2'].rank(method='average', ascending=True)
    
    # Calculate total rank (sum of all metric ranks)
    ranks_df['total_rank'] = (ranks_df['rmse_rank'] + 
                               ranks_df['mae_rank'] + 
                               ranks_df['r2_rank'] + 
                               ranks_df['cmapss_rank'] + 
                               ranks_df['auc_rmse_rank'])
    
    # Add average metrics for reference
    ranks_df['avg_rmse'] = avg_metrics_df['rmse']
    ranks_df['avg_mae'] = avg_metrics_df['mae']
    ranks_df['avg_r2'] = avg_metrics_df['r2']
    ranks_df['avg_cmapss'] = avg_metrics_df['cmapss']
    ranks_df['avg_auc_rmse'] = avg_metrics_df['auc_rmse']
    
    # Sort by total rank (descending - highest is best)
    ranks_df = ranks_df.sort_values('total_rank', ascending=False).reset_index(drop=True)
    ranks_df['overall_rank'] = range(1, len(ranks_df) + 1)
    
    print(f"Ranked {len(ranks_df)} models")
    print(f"Rank range: {ranks_df['total_rank'].min():.1f} to {ranks_df['total_rank'].max():.1f}")
    
    return ranks_df

def select_models_for_phase2(ranks_df):
    print("\nSelecting models for Phase 2 (Feature Selection)...")
    
    selected_models = {}
    
    for family in ['Regression', 'Tree-Based', 'Deep Learning']:
        family_models = ranks_df[ranks_df['family'] == family].copy()
        
        if len(family_models) == 0:
            continue
        
        # Select top models by rank
        if family == 'Deep Learning':
            n_select = 3  # Top 3 DL models
        else:
            n_select = 2  # Top 2 Regression and Tree models
        
        top_models = family_models.head(n_select)
        selected_models[family] = top_models['model_name'].tolist()
        
        print(f"\n  {family}:")
        for idx, row in top_models.iterrows():
            print(f"{int(row['overall_rank'])}. {row['model_name']:<12} | "
                  f"Total Rank: {row['total_rank']:.1f} | "
                  f"RMSE: {row['avg_rmse']:.2f} | R²: {row['avg_r2']:.3f}")
    
    total_selected = sum(len(models) for models in selected_models.values())
    print(f"\n   Total models selected: {total_selected}")
    
    return selected_models

# ==========
# OUTPUT FUNCTIONS
# ==========

def save_results(ranks_df, selected_models):
    print(f"\nSaving results to: {OUTPUT_DIR}")
    
    # Save full ranking table
    output_file = OUTPUT_DIR / 'model_rankings_borda_count.csv'
    ranks_df.to_csv(output_file, index=False)
    print(f"Saved: model_rankings_borda_count.csv")
    
    # Save selected models
    selected_df = pd.DataFrame([
        {'family': family, 'model_name': model, 'rank': i+1}
        for family, models in selected_models.items()
        for i, model in enumerate(models)
    ])
    output_file = OUTPUT_DIR / 'selected_models_phase2.csv'
    selected_df.to_csv(output_file, index=False)
    print(f"Saved: selected_models_phase2.csv")
    
    # Save detailed report
    report_file = OUTPUT_DIR / 'model_selection_report.txt'
    with open(report_file, 'w') as f:
        f.write("="*40 + "\n")
        f.write("MODEL SELECTION FOR PHASE 2: FEATURE SELECTION\n")
        f.write("Method: Borda Count (Rank-Based Voting)\n")
        f.write("="*40 + "\n\n")
        
        f.write("METHODOLOGY:\n")
        f.write("-" * 70 + "\n")
        f.write("1. Calculate average metrics across all data percentages (100%-10%)\n")
        f.write("2. Rank models on each metric (1=worst, 17=best):\n")
        f.write("- RMSE, MAE, CMAPSS, AUC-RMSE: lower is better\n")
        f.write("- R²: higher is better\n")
        f.write("3. Sum ranks across all 5 metrics\n")
        f.write("4. Select top-ranked models from each family\n\n")
        
        f.write("VALIDATION SET ONLY (No Test Leakage)\n")
        f.write("-" * 70 + "\n\n")
        
        f.write("TOP 10 MODELS BY TOTAL RANK:\n")
        f.write("="*40 + "\n")
        for idx, row in ranks_df.head(10).iterrows():
            f.write(f"{int(row['overall_rank']):2d}. {row['model_name']:<12} ({row['family']:<15}) | "
                   f"Total: {row['total_rank']:5.1f} | "
                   f"RMSE: {row['avg_rmse']:6.2f} | "
                   f"R²: {row['avg_r2']:.3f}\n")
        
        f.write("\n" + "="*40 + "\n")
        f.write("SELECTED MODELS FOR PHASE 2:\n")
        f.write("="*40 + "\n")
        for family, models in selected_models.items():
            f.write(f"\n{family}:\n")
            for model in models:
                model_row = ranks_df[ranks_df['model_name'] == model].iloc[0]
                f.write(f"- {model:<12} | Rank: {int(model_row['overall_rank']):2d} | "
                       f"Total: {model_row['total_rank']:5.1f}\n")
        
        f.write("\n" + "="*40 + "\n")
        f.write("INDIVIDUAL METRIC RANKINGS:\n")
        f.write("="*40 + "\n\n")
        
        for idx, row in ranks_df.iterrows():
            f.write(f"{int(row['overall_rank']):2d}. {row['model_name']:<12} | ")
            f.write(f"RMSE: {row['rmse_rank']:4.1f} | ")
            f.write(f"MAE: {row['mae_rank']:4.1f} | ")
            f.write(f"R²: {row['r2_rank']:4.1f} | ")
            f.write(f"CMAPSS: {row['cmapss_rank']:4.1f} | ")
            f.write(f"AUC: {row['auc_rmse_rank']:4.1f}\n")
    
    print(f"Saved: model_selection_report.txt")

# ==========
# MAIN EXECUTION
# ==========

def main():
    # Load data
    all_data = load_all_results()
    
    # Calculate average metrics
    avg_metrics = calculate_average_metrics(all_data)
    
    # Rank models
    ranks_df = rank_models(avg_metrics)
    
    # Select models for Phase 2
    selected_models = select_models_for_phase2(ranks_df)
    
    # Save results
    save_results(ranks_df, selected_models)
    
    print("\n" + "="*40)
    print("MODEL SELECTION COMPLETE!")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nKey files:")
    print("1. model_rankings_borda_count.csv - Full ranking table")
    print("2. selected_models_phase2.csv - Models for Feature Selection")
    print("3. model_selection_report.txt - Detailed report")
    print("\nNext Step: Phase 2 - Feature Selection experiments")
    print("="*40)

if __name__ == "__main__":
    main()

