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

PHASE2_RESULTS = project_root / "Results" / "Phase2_Feature_Selection" / "FD001"
OUTPUT_DIR = PHASE2_RESULTS / "Model_Rankings"
OUTPUT_DIR.mkdir(exist_ok=True)

# Feature selection methods
FS_METHODS = ['No_FS', 'Correlation_FS', 'Tree_FS']

# ==========
# LOAD DATA
# ==========

def load_phase2_results():
    print("="*40)
    print("PHASE 2 MODEL RANKINGS (Separate for High & Low Engine Counts)")
    print("="*40)
    print("\nLoading Phase 2 results...")
    
    all_data = []
    
    for fs_method in FS_METHODS:
        fs_dir = PHASE2_RESULTS / fs_method
        if not fs_dir.exists():
            print(f"Warning: {fs_method} directory not found, skipping...")
            continue
        
        # Look for all model subdirectories
        for model_dir in fs_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            # Load metrics summary
            metrics_file = model_dir / f"{model_dir.name}_metrics_summary.csv"
            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                df['fs_method'] = fs_method
                df['model_name'] = model_dir.name
                all_data.append(df)
    
    if not all_data:
        print("No Phase 2 results found!")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined_df)} total records")
    print(f"Models found: {combined_df['model_name'].nunique()}")
    print(f"FS methods: {combined_df['fs_method'].nunique()}")
    
    return combined_df

# ==========
# RANKING FUNCTIONS
# ==========

def calculate_average_metrics(df, engine_range, range_name, split_type='val'):
    print(f"\nCalculating averages for {range_name} ({min(engine_range)}-{max(engine_range)} engines) - {split_type.upper()} set...")
    
    # Filter to specified engine range
    filtered_data = df[df['n_engines'].isin(engine_range)].copy()
    
    if len(filtered_data) == 0:
        print(f"No data found for {range_name}")
        return pd.DataFrame()
    
    # Group by model + FS method and calculate mean metrics
    avg_metrics = filtered_data.groupby(['model_name', 'fs_method']).agg({
        f'{split_type}_rmse': 'mean',
        f'{split_type}_mae': 'mean',
        f'{split_type}_r2': 'mean',
        f'{split_type}_cmapss': 'mean',
        f'{split_type}_auc_rmse': 'mean',
        'training_time_sec': 'mean'
    }).reset_index()
    
    # Rename columns to match ranking function expectations
    avg_metrics.rename(columns={
        f'{split_type}_rmse': 'rmse',
        f'{split_type}_mae': 'mae',
        f'{split_type}_r2': 'r2',
        f'{split_type}_cmapss': 'cmapss',
        f'{split_type}_auc_rmse': 'auc_rmse'
    }, inplace=True)
    
    # Create combined identifier
    avg_metrics['model_fs'] = avg_metrics['model_name'] + ' + ' + avg_metrics['fs_method']
    
    print(f"Calculated averages for {len(avg_metrics)} model+FS combinations")
    
    return avg_metrics

def rank_models(avg_metrics_df, range_name):
    print(f"\nRanking models for {range_name}...")
    
    if len(avg_metrics_df) == 0:
        return pd.DataFrame()
    
    ranks_df = avg_metrics_df[['model_name', 'fs_method', 'model_fs']].copy()
    
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
    
    print(f"Ranked {len(ranks_df)} model+FS combinations")
    print(f"Total rank range: {ranks_df['total_rank'].min():.1f} to {ranks_df['total_rank'].max():.1f}")
    
    return ranks_df

def print_all_models(ranks_df, range_name, split_type='VAL'):
    print(f"\n{'='*40}")
    print(f"ALL MODELS RANKED - {range_name.upper()} ({split_type} SET)")
    print(f"{'='*40}")
    print(f"{'Rank':<6} {'Model + FS':<30} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}")
    print("-"*140)
    
    for idx, row in ranks_df.iterrows():
        time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
        print(f"{row['overall_rank']:<6} {row['model_fs']:<30} "
              f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
              f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}")
    
    print("-"*140)
    print(f"\nTotal model+FS combinations ranked: {len(ranks_df)}")

def save_results(high_val_ranks, low_val_ranks, high_test_ranks, low_test_ranks):
    print(f"\n{'='*40}")
    print("SAVING RESULTS")
    print(f"{'='*40}")
    
    # Save validation rankings
    if not high_val_ranks.empty:
        high_val_file = OUTPUT_DIR / "phase2_high_engines_val_rankings.csv"
        high_val_ranks.to_csv(high_val_file, index=False)
        print(f"High engine VAL rankings saved: {high_val_file}")
    
    if not low_val_ranks.empty:
        low_val_file = OUTPUT_DIR / "phase2_low_engines_val_rankings.csv"
        low_val_ranks.to_csv(low_val_file, index=False)
        print(f"Low engine VAL rankings saved: {low_val_file}")
    
    # Save test rankings
    if not high_test_ranks.empty:
        high_test_file = OUTPUT_DIR / "phase2_high_engines_test_rankings.csv"
        high_test_ranks.to_csv(high_test_file, index=False)
        print(f"High engine TEST rankings saved: {high_test_file}")
    
    if not low_test_ranks.empty:
        low_test_file = OUTPUT_DIR / "phase2_low_engines_test_rankings.csv"
        low_test_ranks.to_csv(low_test_file, index=False)
        print(f"Low engine TEST rankings saved: {low_test_file}")
    
    # Create comprehensive text report
    report_file = OUTPUT_DIR / "phase2_rankings_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*40 + "\n")
        f.write("PHASE 2 MODEL RANKINGS REPORT\n")
        f.write("Separate Rankings for High (80-10) vs Low (10-1) Engine Counts\n")
        f.write("VALIDATION and TEST sets shown separately\n")
        f.write("="*40 + "\n\n")
        
        f.write("NOTE: Use VALIDATION rankings for decision-making\n")
        f.write("TEST rankings are for thesis results section only\n")
        f.write("="*40 + "\n\n")
        
        # VALIDATION SET RANKINGS
        f.write("\n" + "="*40 + "\n")
        f.write("VALIDATION SET RANKINGS (for decision-making)\n")
        f.write("="*40 + "\n\n")
        
        # High engines - validation
        if not high_val_ranks.empty:
            f.write("HIGH ENGINE COUNTS (80-10 engines) - VALIDATION\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model + FS':<30} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*140 + "\n")
            
            for idx, row in high_val_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_fs']:<30} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # Low engines - validation
        if not low_val_ranks.empty:
            f.write("LOW ENGINE COUNTS (10-1 engines) - VALIDATION\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model + FS':<30} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*140 + "\n")
            
            for idx, row in low_val_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_fs']:<30} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # TEST SET RANKINGS
        f.write("\n" + "="*40 + "\n")
        f.write("TEST SET RANKINGS (for thesis results only)\n")
        f.write("="*40 + "\n\n")
        
        # High engines - test
        if not high_test_ranks.empty:
            f.write("HIGH ENGINE COUNTS (80-10 engines) - TEST\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model + FS':<30} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*140 + "\n")
            
            for idx, row in high_test_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_fs']:<30} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # Low engines - test
        if not low_test_ranks.empty:
            f.write("LOW ENGINE COUNTS (10-1 engines) - TEST\n")
            f.write("="*40 + "\n")
            f.write(f"{'Rank':<6} {'Model + FS':<30} {'Total':<7} {'RMSE':<7} {'MAE':<7} {'R²':<7} {'CMAPSS':<8} {'AUC-R':<7} {'Time(s)':<9}\n")
            f.write("-"*140 + "\n")
            
            for idx, row in low_test_ranks.iterrows():
                time_str = f"{row['avg_train_time']:.1f}" if not pd.isna(row['avg_train_time']) else "N/A"
                f.write(f"{row['overall_rank']:<6} {row['model_fs']:<30} "
                       f"{row['total_rank']:<7.1f} {row['avg_rmse']:<7.2f} {row['avg_mae']:<7.2f} "
                       f"{row['avg_r2']:<7.3f} {row['avg_cmapss']:<8.0f} {row['avg_auc_rmse']:<7.2f} {time_str:<9}\n")
            
            f.write("\n" + "="*40 + "\n\n")
        
        # Key insights (based on validation)
        f.write("KEY INSIGHTS (Based on VALIDATION set)\n")
        f.write("="*40 + "\n\n")
        
        if not high_val_ranks.empty:
            best_high = high_val_ranks.iloc[0]
            time_str_h = f"{best_high['avg_train_time']:.1f}s" if not pd.isna(best_high['avg_train_time']) else "N/A"
            f.write(f"Best for HIGH engine counts (80-10):\n")
            f.write(f"{best_high['model_fs']}\n")
            f.write(f"RMSE: {best_high['avg_rmse']:.2f} | MAE: {best_high['avg_mae']:.2f} | R²: {best_high['avg_r2']:.3f}\n")
            f.write(f"CMAPSS: {best_high['avg_cmapss']:.0f} | AUC-RMSE: {best_high['avg_auc_rmse']:.2f} | Training Time: {time_str_h}\n\n")
        
        if not low_val_ranks.empty:
            best_low = low_val_ranks.iloc[0]
            time_str_l = f"{best_low['avg_train_time']:.1f}s" if not pd.isna(best_low['avg_train_time']) else "N/A"
            f.write(f"Best for LOW engine counts (10-1):\n")
            f.write(f"{best_low['model_fs']}\n")
            f.write(f"RMSE: {best_low['avg_rmse']:.2f} | MAE: {best_low['avg_mae']:.2f} | R²: {best_low['avg_r2']:.3f}\n")
            f.write(f"CMAPSS: {best_low['avg_cmapss']:.0f} | AUC-RMSE: {best_low['avg_auc_rmse']:.2f} | Training Time: {time_str_l}\n\n")
    
    print(f"Comprehensive report saved: {report_file}")
    print(f"\nAll results saved to: {OUTPUT_DIR}")

# ==========
# MAIN EXECUTION
# ==========

def print_detailed_breakdown(df, range_name, split_type='val'):
    print(f"\n{'='*40}")
    print(f"DETAILED BREAKDOWN BY ENGINE COUNT - {range_name.upper()} ({split_type.upper()} SET)")
    print(f"{'='*40}")
    
    # Get sorted engine counts
    engine_counts = sorted(df['n_engines'].unique(), reverse=True)
    
    for n_engines in engine_counts:
        print(f"\n{n_engines} Engines:")
        print("-"*150)
        print(f"{'Model + FS':<35} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'CMAPSS':<10} {'AUC-RMSE':<10} {'Time(s)':<10}")
        print("-"*150)
        
        engine_data = df[df['n_engines'] == n_engines].sort_values(f'{split_type}_rmse')
        
        for idx, row in engine_data.iterrows():
            model_fs = f"{row['model_name']} + {row['fs_method']}"
            time_str = f"{row['training_time_sec']:.1f}" if not pd.isna(row['training_time_sec']) else "N/A"
            print(f"{model_fs:<35} "
                  f"{row[f'{split_type}_rmse']:<8.2f} {row[f'{split_type}_mae']:<8.2f} "
                  f"{row[f'{split_type}_r2']:<8.3f} {row[f'{split_type}_cmapss']:<10.0f} "
                  f"{row[f'{split_type}_auc_rmse']:<10.2f} {time_str:<10}")

def main():
    
    # Load Phase 2 results
    all_data = load_phase2_results()
    
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
        print_all_models(high_val_ranks, "High Engine Counts (80-10)", split_type='VAL')
    
    # Calculate and rank VALIDATION set - LOW engine counts
    low_val_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_LOW, "Low Engine Counts", split_type='val')
    low_val_ranks = rank_models(low_val_avg, "Low Engine Counts (10-1) - VALIDATION")
    if not low_val_ranks.empty:
        print_all_models(low_val_ranks, "Low Engine Counts (10-1)", split_type='VAL')
    
    print("\n" + "="*40)
    print("TEST SET RANKINGS (for thesis results section)")
    print("="*40)
    
    # Calculate and rank TEST set - HIGH engine counts
    high_test_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_HIGH, "High Engine Counts", split_type='test')
    high_test_ranks = rank_models(high_test_avg, "High Engine Counts (80-10) - TEST")
    if not high_test_ranks.empty:
        print_all_models(high_test_ranks, "High Engine Counts (80-10)", split_type='TEST')
    
    # Calculate and rank TEST set - LOW engine counts
    low_test_avg = calculate_average_metrics(all_data, ENGINE_COUNTS_LOW, "Low Engine Counts", split_type='test')
    low_test_ranks = rank_models(low_test_avg, "Low Engine Counts (10-1) - TEST")
    if not low_test_ranks.empty:
        print_all_models(low_test_ranks, "Low Engine Counts (10-1)", split_type='TEST')
    
    # Show detailed breakdown by engine count - VALIDATION
    print("\n" + "="*40)
    print("DETAILED BREAKDOWN (VALIDATION SET)")
    print("="*40)
    
    # High engine counts detailed breakdown - validation
    high_engine_data = all_data[all_data['n_engines'].isin(ENGINE_COUNTS_HIGH)]
    print_detailed_breakdown(high_engine_data, "High Engine Counts (80-10)", split_type='val')
    
    # Low engine counts detailed breakdown - validation
    low_engine_data = all_data[all_data['n_engines'].isin(ENGINE_COUNTS_LOW)]
    print_detailed_breakdown(low_engine_data, "Low Engine Counts (10-1)", split_type='val')
    
    # Show detailed breakdown by engine count - TEST
    print("\n" + "="*40)
    print("DETAILED BREAKDOWN (TEST SET)")
    print("="*40)
    
    # High engine counts detailed breakdown - test
    print_detailed_breakdown(high_engine_data, "High Engine Counts (80-10)", split_type='test')
    
    # Low engine counts detailed breakdown - test
    print_detailed_breakdown(low_engine_data, "Low Engine Counts (10-1)", split_type='test')
    
    # Save results (both validation and test)
    save_results(high_val_ranks, low_val_ranks, high_test_ranks, low_test_ranks)
    
    print("\n" + "="*40)
    print("PHASE 2 RANKING COMPLETE!")
    print("="*40)

if __name__ == "__main__":
    main()

