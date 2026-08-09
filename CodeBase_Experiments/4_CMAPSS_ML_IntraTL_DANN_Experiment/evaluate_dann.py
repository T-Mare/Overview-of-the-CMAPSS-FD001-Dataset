import pandas as pd
import numpy as np
from pathlib import Path

# ==========
# CONFIGURATION
# ==========

SOURCE_DATASET = 'FD002'
TARGET_DATASET = 'FD001'

# Paths
RESULTS_BASE = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / TARGET_DATASET
DANN_RESULTS_PATH = RESULTS_BASE / f'DANN_GRU_{SOURCE_DATASET}_all_results.csv'
BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase2_Feature_Selection' / TARGET_DATASET / 'Correlation_FS' / 'GRU' / 'GRU_metrics_summary.csv'

# Vanilla TL results paths
TL1_GRU_PATH = RESULTS_BASE / f'TL1_GRU_{SOURCE_DATASET}_all_results.csv'
TL2_GRU_PATH = RESULTS_BASE / f'TL2_GRU_{SOURCE_DATASET}_all_results.csv'
TL3_GRU_PATH = RESULTS_BASE / f'TL3_GRU_{SOURCE_DATASET}_all_results.csv'

# ==========
# COMPARISON FUNCTIONS
# ==========

def load_and_prepare_data():
    data = {}
    
    # Load DANN results
    if DANN_RESULTS_PATH.exists():
        data['DANN'] = pd.read_csv(DANN_RESULTS_PATH)
        print(f"Loaded DANN results: {len(data['DANN'])} experiments")
    else:
        print(f"DANN results not found: {DANN_RESULTS_PATH}")
        data['DANN'] = None
    
    # Load baseline
    if BASELINE_PATH.exists():
        baseline = pd.read_csv(BASELINE_PATH)
        # Remove duplicates - keep best (lowest RMSE)
        baseline = baseline.sort_values('val_rmse').drop_duplicates('n_engines', keep='first')
        baseline = baseline[baseline['fs_method'] == 'Correlation_FS']
        data['Baseline'] = baseline
        print(f"Loaded baseline: {len(data['Baseline'])} engine counts")
    else:
        print(f"Baseline not found: {BASELINE_PATH}")
        data['Baseline'] = None
    
    # Load vanilla TL results
    for name, path in [('TL1-GRU', TL1_GRU_PATH), ('TL2-GRU', TL2_GRU_PATH), ('TL3-GRU', TL3_GRU_PATH)]:
        if path.exists():
            data[name] = pd.read_csv(path)
            print(f"Loaded {name}: {len(data[name])} experiments")
        else:
            print(f"{name} not found: {path}")
            data[name] = None
    
    return data

def compare_dann_vs_baseline(data):
    if data['DANN'] is None or data['Baseline'] is None:
        print("\n Cannot compare DANN vs Baseline: Missing data")
        return
    
    print("\n" + "="*40)
    print(f"DANN vs BASELINE (No Transfer Learning)")
    print("="*40)
    print(f"{'Engines':<10}{'% Data':<10}{'Val RMSE':<30}{'Test RMSE':<30}{'Improvement':<20}")
    print(f"{'':10}{'':10}{'DANN':<12}{'Baseline':<12}{'Gain%':<10}{'DANN':<12}{'Baseline':<12}{'Gain%':<10}")
    print('-'*120)
    
    # Merge DANN and baseline
    comparison = pd.merge(
        data['DANN'][['n_engines', 'val_rmse', 'test_rmse']],
        data['Baseline'][['n_engines', 'val_rmse', 'test_rmse']],
        on='n_engines',
        suffixes=('_dann', '_baseline')
    )
    
    comparison['pct_data'] = (comparison['n_engines'] / 80 * 100).round(1)
    comparison['val_improvement'] = ((comparison['val_rmse_baseline'] - comparison['val_rmse_dann']) / comparison['val_rmse_baseline'] * 100).round(2)
    comparison['test_improvement'] = ((comparison['test_rmse_baseline'] - comparison['test_rmse_dann']) / comparison['test_rmse_baseline'] * 100).round(2)
    
    # Sort by n_engines
    comparison = comparison.sort_values('n_engines')
    
    for _, row in comparison.iterrows():
        val_emoji = '' if row['val_improvement'] > 0 else ''
        test_emoji = '' if row['test_improvement'] > 0 else ''
        print(f"{int(row['n_engines']):<10}{row['pct_data']:<10.1f}"
              f"{row['val_rmse_dann']:<12.2f}{row['val_rmse_baseline']:<12.2f}{row['val_improvement']:>8.1f}% {val_emoji:<4}"
              f"{row['test_rmse_dann']:<12.2f}{row['test_rmse_baseline']:<12.2f}{row['test_improvement']:>8.1f}% {test_emoji}")
    
    # Summary stats
    print(f"\n  Average Val Improvement:  {comparison['val_improvement'].mean():>6.2f}%")
    print(f"Average Test Improvement: {comparison['test_improvement'].mean():>6.2f}%")
    print(f"Win Rate (Val): {(comparison['val_improvement'] > 0).sum()}/{len(comparison)} ({(comparison['val_improvement'] > 0).sum()/len(comparison)*100:.1f}%)")
    print(f"Win Rate (Test): {(comparison['test_improvement'] > 0).sum()}/{len(comparison)} ({(comparison['test_improvement'] > 0).sum()/len(comparison)*100:.1f}%)")

def compare_dann_vs_vanilla_tl(data):
    if data['DANN'] is None:
        print("\n Cannot compare DANN vs Vanilla TL: DANN data missing")
        return
    
    print("\n" + "="*40)
    print(f"DANN vs VANILLA TRANSFER LEARNING (TL1, TL2, TL3)")
    print("="*40)
    print(f"{'Engines':<10}{'%Data':<8}{'Val RMSE (Lower is Better)':<60}{'Best Method':<20}")
    print(f"{'':10}{'':8}{'DANN':<12}{'TL1':<12}{'TL2':<12}{'TL3':<12}{'':20}")
    print('-'*140)
    
    # Prepare comparison data
    dann_df = data['DANN'][['n_engines', 'val_rmse', 'test_rmse']].copy()
    dann_df.columns = ['n_engines', 'val_rmse_dann', 'test_rmse_dann']
    
    comparison = dann_df.sort_values('n_engines')
    
    # Add TL methods
    for method_name in ['TL1-GRU', 'TL2-GRU', 'TL3-GRU']:
        if data[method_name] is not None:
            method_df = data[method_name][['n_engines', 'val_rmse', 'test_rmse']].copy()
            method_df.columns = ['n_engines', f'val_rmse_{method_name.lower().replace("-", "_")}', f'test_rmse_{method_name.lower().replace("-", "_")}']
            comparison = pd.merge(comparison, method_df, on='n_engines', how='left')
    
    comparison['pct_data'] = (comparison['n_engines'] / 80 * 100).round(1)
    
    # Print results
    for _, row in comparison.iterrows():
        val_scores = {
            'DANN': row.get('val_rmse_dann', np.nan),
            'TL1': row.get('val_rmse_tl1_gru', np.nan),
            'TL2': row.get('val_rmse_tl2_gru', np.nan),
            'TL3': row.get('val_rmse_tl3_gru', np.nan)
        }
        
        # Find best method
        valid_scores = {k: v for k, v in val_scores.items() if not np.isnan(v)}
        if valid_scores:
            best_method = min(valid_scores, key=valid_scores.get)
        else:
            best_method = 'N/A'
        
        print(f"{int(row['n_engines']):<10}{row['pct_data']:<8.1f}"
              f"{val_scores['DANN']:<12.2f}"
              f"{val_scores['TL1'] if not np.isnan(val_scores['TL1']) else 'N/A':<12}"
              f"{val_scores['TL2'] if not np.isnan(val_scores['TL2']) else 'N/A':<12}"
              f"{val_scores['TL3'] if not np.isnan(val_scores['TL3']) else 'N/A':<12}"
              f"{best_method:<20}")
    
    # Summary: DANN win rate
    dann_wins = 0
    total_comparisons = 0
    for _, row in comparison.iterrows():
        val_scores = {
            'DANN': row.get('val_rmse_dann', np.nan),
            'TL1': row.get('val_rmse_tl1_gru', np.nan),
            'TL2': row.get('val_rmse_tl2_gru', np.nan),
            'TL3': row.get('val_rmse_tl3_gru', np.nan)
        }
        valid_scores = {k: v for k, v in val_scores.items() if not np.isnan(v)}
        if len(valid_scores) > 1:
            total_comparisons += 1
            if valid_scores['DANN'] == min(valid_scores.values()):
                dann_wins += 1
    
    if total_comparisons > 0:
        print(f"\n  DANN Win Rate: {dann_wins}/{total_comparisons} ({dann_wins/total_comparisons*100:.1f}%)")

def analyze_low_data_regime(data):
    if data['DANN'] is None:
        print("\n Cannot analyze low-data regime: DANN data missing")
        return
    
    print("\n" + "="*40)
    print(f"LOW-DATA REGIME ANALYSIS (≤10 engines)")
    print("="*40)
    
    # Filter for low data
    dann_low = data['DANN'][data['DANN']['n_engines'] <= 10].copy()
    
    if data['Baseline'] is not None:
        baseline_low = data['Baseline'][data['Baseline']['n_engines'] <= 10].copy()
        
        # Calculate average improvement
        comparison_low = pd.merge(
            dann_low[['n_engines', 'val_rmse', 'test_rmse']],
            baseline_low[['n_engines', 'val_rmse', 'test_rmse']],
            on='n_engines',
            suffixes=('_dann', '_baseline')
        )
        
        comparison_low['val_improvement'] = ((comparison_low['val_rmse_baseline'] - comparison_low['val_rmse_dann']) / comparison_low['val_rmse_baseline'] * 100)
        comparison_low['test_improvement'] = ((comparison_low['test_rmse_baseline'] - comparison_low['test_rmse_dann']) / comparison_low['test_rmse_baseline'] * 100)
        
        print(f"\nDANN vs Baseline (Low Data):")
        print(f"Average Val Improvement:  {comparison_low['val_improvement'].mean():>6.2f}%")
        print(f"Average Test Improvement: {comparison_low['test_improvement'].mean():>6.2f}%")
        print(f"Win Rate (Val): {(comparison_low['val_improvement'] > 0).sum()}/{len(comparison_low)} ({(comparison_low['val_improvement'] > 0).sum()/len(comparison_low)*100:.1f}%)")
    
    # Show where DANN helps most
    print(f"\nBest DANN Improvements (Val RMSE):")
    if data['Baseline'] is not None and len(comparison_low) > 0:
        best_improvements = comparison_low.nlargest(5, 'val_improvement')
        for _, row in best_improvements.iterrows():
            print(f"{int(row['n_engines'])} engines: {row['val_improvement']:>6.2f}% improvement "
                  f"(DANN: {row['val_rmse_dann']:.2f} vs Baseline: {row['val_rmse_baseline']:.2f})")

def generate_thesis_summary(data):
    if data['DANN'] is None:
        print("\n Cannot generate thesis summary: DANN data missing")
        return
    
    print("\n" + "="*40)
    print(f"THESIS SUMMARY: DANN FOR {SOURCE_DATASET}{TARGET_DATASET} TRANSFER")
    print("="*40)
    
    dann_df = data['DANN']
    
    print(f"\nExperiments Completed: {len(dann_df)}")
    print(f"Engine Counts Tested: {sorted(dann_df['n_engines'].unique())}")
    
    print(f"\nDomain Discriminator Performance:")
    if 'final_domain_acc' in dann_df.columns:
        avg_domain_acc = dann_df['final_domain_acc'].mean()
        print(f"Average Domain Accuracy: {avg_domain_acc:.3f}")
        if 0.5 <= avg_domain_acc <= 0.6:
            print(f"Target range achieved (0.5-0.6) - Feature extractor successfully confuses discriminator")
        elif avg_domain_acc > 0.7:
            print(f"Discriminator too strong - Consider increasing lambda")
        else:
            print(f"Discriminator too weak - Consider decreasing lambda")
    
    print(f"\nOverall Performance:")
    print(f"Average Val RMSE: {dann_df['val_rmse'].mean():.4f} (±{dann_df['val_rmse'].std():.4f})")
    print(f"Average Test RMSE: {dann_df['test_rmse'].mean():.4f} (±{dann_df['test_rmse'].std():.4f})")
    print(f"Average Training Time: {dann_df['training_time_sec'].mean():.1f}s")
    
    if data['Baseline'] is not None:
        comparison = pd.merge(
            dann_df[['n_engines', 'val_rmse', 'test_rmse']],
            data['Baseline'][['n_engines', 'val_rmse', 'test_rmse']],
            on='n_engines',
            suffixes=('_dann', '_baseline')
        )
        comparison['val_improvement'] = ((comparison['val_rmse_baseline'] - comparison['val_rmse_dann']) / comparison['val_rmse_baseline'] * 100)
        
        print(f"\nTransfer Learning Effectiveness:")
        print(f"Average Improvement over Baseline: {comparison['val_improvement'].mean():>6.2f}%")
        print(f"Positive Transfer Rate: {(comparison['val_improvement'] > 0).sum()}/{len(comparison)} ({(comparison['val_improvement'] > 0).sum()/len(comparison)*100:.1f}%)")
        
        # Low vs high data regime
        low_data = comparison[comparison['n_engines'] <= 10]
        high_data = comparison[comparison['n_engines'] > 10]
        
        if len(low_data) > 0:
            print(f"\n  Low Data (≤10 engines): {low_data['val_improvement'].mean():>6.2f}% average improvement")
        if len(high_data) > 0:
            print(f"High Data (>10 engines): {high_data['val_improvement'].mean():>6.2f}% average improvement")

# ==========
# MAIN EXECUTION
# ==========

def main():
    
    print("="*40)
    print("DANN EVALUATION: COMPARING DOMAIN-ADVERSARIAL TRANSFER LEARNING")
    print("="*40)
    print(f"Source: {SOURCE_DATASET} (6 operational conditions)")
    print(f"Target: {TARGET_DATASET} (1 operational condition)")
    print(f"Architecture: GRU")
    
    # Load all data
    print("\nLoading data...")
    data = load_and_prepare_data()
    
    # Run comparisons
    compare_dann_vs_baseline(data)
    compare_dann_vs_vanilla_tl(data)
    analyze_low_data_regime(data)
    generate_thesis_summary(data)
    
    print("\n" + "="*40)
    print("EVALUATION COMPLETE")
    print("="*40)

if __name__ == '__main__':
    main()

