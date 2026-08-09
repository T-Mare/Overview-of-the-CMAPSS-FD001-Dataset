import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from Utilities.config import ENGINE_COUNTS_HIGH, ENGINE_COUNTS_LOW

# ==========
# CONFIGURATION
# ==========

PHASE1_RESULTS = project_root / "Results" / "Phase1_Baseline" / "FD001"
OUTPUT_DIR = PHASE1_RESULTS / "Model_Rankings"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========
# LOAD DATA
# ==========

def load_phase1_results():
    print("="*40)
    print("PHASE 1 MODEL RANKINGS (Separate for High & Low Engine Counts)")
    print("="*40)
    print("\nLoading Phase 1 baseline results...")
    
    all_data = []
    models_found = []
    
    # Iterate through all subdirectories in Phase 1 results
    for model_dir in PHASE1_RESULTS.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Skip special directories
        if model_dir.name in ['Model_Selection', 'Evaluation_Plots', 'Model_Rankings']:
            continue
        
        # Look for metrics summary file
        metrics_file = model_dir / f"{model_dir.name}_metrics_summary.csv"
        
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            df['model_name'] = model_dir.name
            
            # Normalize column names (regression models use 'dataset', DL models use 'split')
            if 'dataset' in df.columns and 'split' not in df.columns:
                df['split'] = df['dataset']
            
            all_data.append(df)
            models_found.append(model_dir.name)
    
    if not all_data:
        print("No Phase 1 results found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} total records")
    print(f"Models found: {len(models_found)}")
    print(f"Model list: {', '.join(sorted(models_found))}")
    
    return combined_df

# ==========
# RANKING FUNCTIONS
# ==========

def calculate_average_metrics(df, engine_range, range_name, split_type='val'):
    print(f"\nCalculating averages for {range_name} ({min(engine_range)}-{max(engine_range)} engines) - {split_type.upper()} set...")
    
    # Filter to specified split and engine range
    filtered_data = df[(df['split'] == split_type) & 
                       (df['n_engines'].isin(engine_range))].copy()
    
    if len(filtered_data) == 0:
        print(f"No {split_type} data found for {range_name}")
        return pd.DataFrame()
    
    # Get training time from train split
    train_data = df[(df['split'] == 'train') & 
                    (df['n_engines'].isin(engine_range))].copy()
    
    # Group by model and calculate mean of metrics
    avg_metrics = filtered_data.groupby('model_name').agg({
        'rmse': 'mean',
        'mae': 'mean',
        'r2': 'mean',
        'cmapss_score': 'mean',
        'auc_rmse': 'mean'
    }).reset_index()
    
    # Add average training time if available
    if not train_data.empty and 'training_time_sec' in train_data.columns:
        avg_time = train_data.groupby('model_name')['training_time_sec'].mean().reset_index()
        avg_metrics = avg_metrics.merge(avg_time, on='model_name', how='left')
    else:
        avg_metrics['training_time_sec'] = np.nan
    
    # Rename cmapss_score to cmapss for consistency
    avg_metrics.rename(columns={'cmapss_score': 'cmapss'}, inplace=True)
    
    print(f"Calculated averages for {len(avg_metrics)} models")
    
    return avg_metrics

def rank_models(avg_metrics_df, range_name):
    print(f"\nRanking models for {range_name}...")
    
    if len(avg_metrics_df) == 0:
        return pd.DataFrame()
    
    ranks_df = avg_metrics_df[['model_name']].copy()
    
    # Rank each metric (higher rank = better performance)
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
    ranks_df['avg_rmse'] = avg_metrics_df['rmse'].round(2)
    ranks_df['avg_mae'] = avg_metrics_df['mae'].round(2)
    ranks_df['avg_r2'] = avg_metrics_df['r2'].round(3)
    ranks_df['avg_cmapss'] = avg_metrics_df['cmapss'].round(0)
    ranks_df['avg_auc_rmse'] = avg_metrics_df['auc_rmse'].round(2)
    ranks_df['avg_train_time'] = avg_metrics_df['training_time_sec'].round(1)
    
    # Sort by total rank (descending - highest is best)
    ranks_df = ranks_df.sort_values('total_rank', ascending=False).reset_index(drop=True)
    ranks_df['overall_rank'] = range(1, len(ranks_df) + 1)
    
    print(f"Ranked {len(ranks_df)} models")
    print(f"Total rank range: {ranks_df['total_rank'].min():.1f} to {ranks_df['total_rank'].max():.1f}")
    
    return ranks_df

def categorize_models(ranks_df):
    regression_models = ['MLR', 'Ridge', 'Lasso', 'ElasticNet', 'Poly2', 'Poly3']
    tree_models = ['RF', 'XGB', 'LGBM']
    dl_models = ['ANN', 'RNN', 'LSTM', 'GRU', 'BiLSTM', 'CNN', 'TCN', 'Transformer']
    
    def get_family(model_name):
        if model_name in regression_models:
            return 'Regression'
        elif model_name in tree_models:
            return 'Tree-Based'
        elif model_name in dl_models:
            return 'Deep Learning'
        else:
            return 'Unknown'
    
    ranks_df['model_family'] = ranks_df['model_name'].apply(get_family)
    return ranks_df

def print_all_models(ranks_df, range_name, split_type='VAL'):
    print(f"\n{'='*40}")
    print(f"ALL MODELS RANKED - {range_name.upper()} ({split_type} SET)")
    print(f"{'='*40}")
    print(f"{'Rank':<6} {'Model':<13} {'Family':<13} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}")
    print("-"*130)
    
    for idx, row in ranks_df.iterrows():
        time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
        print(f"{row['overall_rank']:<6} {row['model_name']:<13} {row['model_family']:<13} "
              f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
              f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}")
    
    print("-"*130)
    print(f"\nTotal models ranked: {len(ranks_df)}")

def print_family_best(ranks_df, range_name, split_type='VAL'):
    print(f"\nBEST MODEL PER FAMILY - {range_name.upper()} ({split_type} SET)")
    print("="*40)
    print(f"{'Family':<15} {'Model':<13} {'Rank':<6} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}")
    print("-"*130)
    
    for family in ['Regression', 'Tree-Based', 'Deep Learning']:
        family_models = ranks_df[ranks_df['model_family'] == family]
        if len(family_models) > 0:
            best = family_models.iloc[0]
            time_str = f"{best['avg_train_time']:.1f}" if not pd.isna(best['avg_train_time']) else "N/A"
            print(f"{family:<15} {best['model_name']:<13} {best['overall_rank']:<6} "
                  f"{best['avg_rmse']:<7.2f} {best['avg_mae']:<7.2f} {best['avg_r2']:<7.3f} "
                  f"{best['avg_cmapss']:<8.0f} {best['avg_auc_rmse']:<7.2f} {time_str:<9}")

def save_results(high_val_ranks, low_val_ranks, high_test_ranks, low_test_ranks):
    print(f"\n{'='*40}")
    print("SAVING RESULTS")
    print(f"{'='*40}")
    
    # Save validation rankings
    if not high_val_ranks.empty:
        high_val_file = OUTPUT_DIR / "phase1_high_engines_val_rankings.csv"
        high_val_ranks.to_csv(high_val_file, index=False)
        print(f"High engine VAL rankings saved: {high_val_file}")
    
    if not low_val_ranks.empty:
        low_val_file = OUTPUT_DIR / "phase1_low_engines_val_rankings.csv"
        low_val_ranks.to_csv(low_val_file, index=False)
        print(f"Low engine VAL rankings saved: {low_val_file}")
    
    # Save test rankings
    if not high_test_ranks.empty:
        high_test_file = OUTPUT_DIR / "phase1_high_engines_test_rankings.csv"
        high_test_ranks.to_csv(high_test_file, index=False)
        print(f"High engine TEST rankings saved: {high_test_file}")
    
    if not low_test_ranks.empty:
        low_test_file = OUTPUT_DIR / "phase1_low_engines_test_rankings.csv"
        low_test_ranks.to_csv(low_test_file, index=False)
        print(f"Low engine TEST rankings saved: {low_test_file}")
    
    # Create comprehensive text report
    report_file = OUTPUT_DIR / "phase1_rankings_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*40 + "\n")
        f.write("PHASE 1 BASELINE MODEL RANKINGS REPORT\n")
        f.write("Separate Rankings for High (80-10) vs Low (10-1) Engine Counts\n")
        f.write("="*40 + "\n\n")
        
        # VALIDATION SET RANKINGS
        f.write("\n" + "="*40 + "\n")
        f.write("VALIDATION SET RANKINGS (for decision-making)\n")
        f.write("="*40 + "\n\n")
        
        # High engines - validation
        if not high_val_ranks.empty:
            f.write("HIGH ENGINE COUNTS (80-10 engines) - VALIDATION\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model':<13} {'Family':<13} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*130 + "\n")
            
            for idx, row in high_val_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_name']:<13} {row['model_family']:<13} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
            
            # Best per family for high engines - validation
            f.write("BEST PER FAMILY - HIGH ENGINE COUNTS (VALIDATION)\n")
            f.write("-"*130 + "\n")
            f.write(f"{'Family':<15} {'Model':<13} {'Rank':<6} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*130 + "\n")
            for family in ['Regression', 'Tree-Based', 'Deep Learning']:
                family_models = high_val_ranks[high_val_ranks['model_family'] == family]
                if len(family_models) > 0:
                    best = family_models.iloc[0]
                    time_str = f"{best['avg_train_time']:.1f}" if not pd.isna(best['avg_train_time']) else "N/A"
                    f.write(f"{family:<15} {best['model_name']:<13} {best['overall_rank']:<6} "
                           f"{best['avg_rmse']:<7.2f} {best['avg_mae']:<7.2f} {best['avg_r2']:<7.3f} "
                           f"{best['avg_cmapss']:<8.0f} {best['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            f.write("\n" + "="*40 + "\n\n")
        
        # Low engines - validation
        if not low_val_ranks.empty:
            f.write("LOW ENGINE COUNTS (10-1 engines) - VALIDATION\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model':<13} {'Family':<13} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*130 + "\n")
            
            for idx, row in low_val_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_name']:<13} {row['model_family']:<13} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
            
            # Best per family for low engines - validation
            f.write("BEST PER FAMILY - LOW ENGINE COUNTS (VALIDATION)\n")
            f.write("-"*130 + "\n")
            f.write(f"{'Family':<15} {'Model':<13} {'Rank':<6} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*130 + "\n")
            for family in ['Regression', 'Tree-Based', 'Deep Learning']:
                family_models = low_val_ranks[low_val_ranks['model_family'] == family]
                if len(family_models) > 0:
                    best = family_models.iloc[0]
                    time_str = f"{best['avg_train_time']:.1f}" if not pd.isna(best['avg_train_time']) else "N/A"
                    f.write(f"{family:<15} {best['model_name']:<13} {best['overall_rank']:<6} "
                           f"{best['avg_rmse']:<7.2f} {best['avg_mae']:<7.2f} {best['avg_r2']:<7.3f} "
                           f"{best['avg_cmapss']:<8.0f} {best['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            f.write("\n" + "="*40 + "\n\n")
        
        # TEST SET RANKINGS
        f.write("\n" + "="*40 + "\n")
        f.write("TEST SET RANKINGS (for thesis results only)\n")
        f.write("="*40 + "\n\n")
        
        # High engines - test
        if not high_test_ranks.empty:
            f.write("HIGH ENGINE COUNTS (80-10 engines) - TEST\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model':<13} {'Family':<13} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*130 + "\n")
            
            for idx, row in high_test_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_name']:<13} {row['model_family']:<13} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # Low engines - test
        if not low_test_ranks.empty:
            f.write("LOW ENGINE COUNTS (10-1 engines) - TEST\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model':<13} {'Family':<13} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*130 + "\n")
            
            for idx, row in low_test_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_name']:<13} {row['model_family']:<13} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # Key insights (based on validation set)
        f.write("KEY INSIGHTS & PHASE 2 RECOMMENDATIONS (Based on VALIDATION set)\n")
        f.write("="*40 + "\n\n")
        
        if not high_val_ranks.empty:
            best_high = high_val_ranks.iloc[0]
            time_str_h = f"{best_high['avg_train_time']:.1f}s" if not pd.isna(best_high['avg_train_time']) else "N/A"
            f.write(f"Best for HIGH engine counts (80-10):\n")
            f.write(f"{best_high['model_name']} ({best_high['model_family']})\n")
            f.write(f"RMSE: {best_high['avg_rmse']:.2f} | MAE: {best_high['avg_mae']:.2f} | R²: {best_high['avg_r2']:.3f}\n")
            f.write(f"CMAPSS: {best_high['avg_cmapss']:.0f} | AUC-RMSE: {best_high['avg_auc_rmse']:.2f} | Training Time: {time_str_h}\n\n")
        
        if not low_val_ranks.empty:
            best_low = low_val_ranks.iloc[0]
            time_str_l = f"{best_low['avg_train_time']:.1f}s" if not pd.isna(best_low['avg_train_time']) else "N/A"
            f.write(f"Best for LOW engine counts (10-1):\n")
            f.write(f"{best_low['model_name']} ({best_low['model_family']})\n")
            f.write(f"RMSE: {best_low['avg_rmse']:.2f} | MAE: {best_low['avg_mae']:.2f} | R²: {best_low['avg_r2']:.3f}\n")
            f.write(f"CMAPSS: {best_low['avg_cmapss']:.0f} | AUC-RMSE: {best_low['avg_auc_rmse']:.2f} | Training Time: {time_str_l}\n\n")
        
        # Analyze what's in Phase 2 (based on validation rankings)
        if not high_val_ranks.empty and not low_val_ranks.empty:
            phase2_models = ['BiLSTM', 'GRU', 'LSTM', 'Lasso', 'Poly2', 'RF', 'XGB']
            
            f.write("CURRENTLY IN PHASE 2:\n")
            f.write("-"*80 + "\n")
            for model in phase2_models:
                high_rank = high_val_ranks[high_val_ranks['model_name'] == model]
                low_rank = low_val_ranks[low_val_ranks['model_name'] == model]
                if not high_rank.empty and not low_rank.empty:
                    f.write(f"{model:<15} | High: Rank {high_rank.iloc[0]['overall_rank']:<3} "
                           f"| Low: Rank {low_rank.iloc[0]['overall_rank']:<3}\n")
            
            f.write("\n\nCONSIDER ADDING TO PHASE 2:\n")
            f.write("-"*80 + "\n")
            
            top_high = high_val_ranks[~high_val_ranks['model_name'].isin(phase2_models)].head(5)
            top_low = low_val_ranks[~low_val_ranks['model_name'].isin(phase2_models)].head(5)
            
            candidates = set(top_high['model_name'].tolist() + top_low['model_name'].tolist())
            
            for model in sorted(candidates):
                high_rank = high_val_ranks[high_val_ranks['model_name'] == model]
                low_rank = low_val_ranks[low_val_ranks['model_name'] == model]
                if not high_rank.empty and not low_rank.empty:
                    f.write(f"{model:<15} | High: Rank {high_rank.iloc[0]['overall_rank']:<3} "
                           f"(RMSE: {high_rank.iloc[0]['avg_rmse']:.2f}) | "
                           f"Low: Rank {low_rank.iloc[0]['overall_rank']:<3} "
                           f"(RMSE: {low_rank.iloc[0]['avg_rmse']:.2f})\n")
    
    print(f"Comprehensive report saved: {report_file}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")

# ==========
# MAIN EXECUTION
# ==========

def main():
    
    # Load Phase 1 results
    all_data = load_phase1_results()
    
    if all_data.empty:
        print("\n No data to rank. Exiting.")
        return
    
    print("\n" + "="*40)
    print("VALIDATION SET RANKINGS (for decision-making)")
    print("="*40)
    
    # Calculate and rank VALIDATION set - HIGH engine counts
    high_val_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_HIGH, "High Engine Counts", split_type='val')
    high_val_ranks = rank_models(high_val_avg, "High Engine Counts (80-10) - VALIDATION")
    if not high_val_ranks.empty:
        high_val_ranks = categorize_models(high_val_ranks)
        print_all_models(high_val_ranks, "High Engine Counts (80-10)", split_type='VAL')
        print_family_best(high_val_ranks, "High Engine Counts (80-10)", split_type='VAL')
    
    # Calculate and rank VALIDATION set - LOW engine counts
    low_val_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_LOW, "Low Engine Counts", split_type='val')
    low_val_ranks = rank_models(low_val_avg, "Low Engine Counts (10-1) - VALIDATION")
    if not low_val_ranks.empty:
        low_val_ranks = categorize_models(low_val_ranks)
        print_all_models(low_val_ranks, "Low Engine Counts (10-1)", split_type='VAL')
        print_family_best(low_val_ranks, "Low Engine Counts (10-1)", split_type='VAL')
    
    print("\n" + "="*40)
    print("TEST SET RANKINGS (for thesis results section)")
    print("="*40)
    
    # Calculate and rank TEST set - HIGH engine counts
    high_test_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_HIGH, "High Engine Counts", split_type='test')
    high_test_ranks = rank_models(high_test_avg, "High Engine Counts (80-10) - TEST")
    if not high_test_ranks.empty:
        high_test_ranks = categorize_models(high_test_ranks)
        print_all_models(high_test_ranks, "High Engine Counts (80-10)", split_type='TEST')
        print_family_best(high_test_ranks, "High Engine Counts (80-10)", split_type='TEST')
    
    # Calculate and rank TEST set - LOW engine counts
    low_test_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_LOW, "Low Engine Counts", split_type='test')
    low_test_ranks = rank_models(low_test_avg, "Low Engine Counts (10-1) - TEST")
    if not low_test_ranks.empty:
        low_test_ranks = categorize_models(low_test_ranks)
        print_all_models(low_test_ranks, "Low Engine Counts (10-1)", split_type='TEST')
        print_family_best(low_test_ranks, "Low Engine Counts (10-1)", split_type='TEST')
    
    # Save results (both validation and test)
    save_results(high_val_ranks, low_val_ranks, high_test_ranks, low_test_ranks)
    
    print("\n" + "="*40)
    print("PHASE 1 RANKING COMPLETE!")
    print("="*40)

if __name__ == "__main__":
    main()

