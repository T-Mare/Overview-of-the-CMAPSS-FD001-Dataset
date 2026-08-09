import pandas as pd
from pathlib import Path
import json
import sys

# Paths
BASELINE_PATH = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'Baseline_Very_Low_Data' / 'baseline_very_low_data_summary.csv'
TL3_BASE = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'FD001' / 'FD003_to_FD001' / 'TL3_Progressive_Unfreeze'

DATA_PERCENTAGES = [1, 2, 3, 4, 5]

print("\n" + "="*40)
print("BASELINE LSTM vs TL3 PROGRESSIVE UNFREEZING: VERY LOW DATA (1-5%)")
print("="*40)
print("Question: Can Transfer Learning rescue performance at extreme data scarcity?")
print("="*40 + "\n")

# Load baseline results
if not BASELINE_PATH.exists():
    print(f"Baseline results not found: {BASELINE_PATH}")
    sys.exit(1)

baseline_df = pd.read_csv(BASELINE_PATH)
print(f"Loaded baseline results: {len(baseline_df)} experiments")

# Load TL3 results
tl3_results = []
for pct in DATA_PERCENTAGES:
    tl3_file = TL3_BASE / f'tl3_metrics_{pct}pct.json'
    if tl3_file.exists():
        with open(tl3_file, 'r') as f:
            data = json.load(f)
            tl3_results.append({
                'data_pct': pct,
                'val_rmse': data['val_rmse'],
                'test_rmse': data['test_rmse'],
                'test_mae': data['test_mae'],
                'test_r2': data['test_r2'],
                'test_auc_rmse': data['test_auc_rmse'],
                'training_time_sec': data.get('total_training_time_sec', data.get('training_time_sec', 0))
            })
    else:
        print(f"Warning: TL3 results not found for {pct}%: {tl3_file}")

if not tl3_results:
    print(f"No TL3 results found in: {TL3_BASE}")
    sys.exit(1)

tl3_df = pd.DataFrame(tl3_results)
print(f"Loaded TL3 results: {len(tl3_df)} experiments\n")

# Merge results
comparison = baseline_df.merge(tl3_df, on='data_pct', suffixes=('_baseline', '_tl3'))

# Calculate improvements
comparison['val_rmse_gain'] = ((comparison['val_rmse_baseline'] - comparison['val_rmse_tl3']) / comparison['val_rmse_baseline'] * 100)
comparison['test_rmse_gain'] = ((comparison['test_rmse_baseline'] - comparison['test_rmse_tl3']) / comparison['test_rmse_baseline'] * 100)
comparison['test_auc_rmse_gain'] = ((comparison['test_auc_rmse_baseline'] - comparison['test_auc_rmse_tl3']) / comparison['test_auc_rmse_baseline'] * 100)

# Display comparison
print("="*40)
print("VALIDATION RMSE COMPARISON")
print("="*40)
print(f"{'Data %':<8} {'Engines':<10} {'Baseline':<12} {'TL3':<12} {'Gain (%)':<12} {'Status':<15}")
print("-"*100)

for _, row in comparison.iterrows():
    gain = row['val_rmse_gain']
    status = " POSITIVE TL" if gain > 0 else " NEGATIVE TL"
    print(f"{row['data_pct']:<8} {row['n_engines']:<10} {row['val_rmse_baseline']:<12.3f} {row['val_rmse_tl3']:<12.3f} {gain:<12.2f} {status:<15}")

print("\n" + "="*40)
print("TEST RMSE COMPARISON")
print("="*40)
print(f"{'Data %':<8} {'Engines':<10} {'Baseline':<12} {'TL3':<12} {'Gain (%)':<12} {'Status':<15}")
print("-"*100)

for _, row in comparison.iterrows():
    gain = row['test_rmse_gain']
    status = " POSITIVE TL" if gain > 0 else " NEGATIVE TL"
    print(f"{row['data_pct']:<8} {row['n_engines']:<10} {row['test_rmse_baseline']:<12.3f} {row['test_rmse_tl3']:<12.3f} {gain:<12.2f} {status:<15}")

print("\n" + "="*40)
print("TEST AUC-RMSE COMPARISON")
print("="*40)
print(f"{'Data %':<8} {'Engines':<10} {'Baseline':<12} {'TL3':<12} {'Gain (%)':<12} {'Status':<15}")
print("-"*100)

for _, row in comparison.iterrows():
    gain = row['test_auc_rmse_gain']
    status = " POSITIVE TL" if gain > 0 else " NEGATIVE TL"
    print(f"{row['data_pct']:<8} {row['n_engines']:<10} {row['test_auc_rmse_baseline']:<12.3f} {row['test_auc_rmse_tl3']:<12.3f} {gain:<12.2f} {status:<15}")

# Summary statistics
print("\n" + "="*40)
print("SUMMARY STATISTICS")
print("="*40)

positive_val = (comparison['val_rmse_gain'] > 0).sum()
positive_test = (comparison['test_rmse_gain'] > 0).sum()
positive_auc = (comparison['test_auc_rmse_gain'] > 0).sum()

print(f"Positive Transfer (Val RMSE):  {positive_val}/{len(comparison)} experiments")
print(f"Positive Transfer (Test RMSE): {positive_test}/{len(comparison)} experiments")
print(f"Positive Transfer (Test AUC):  {positive_auc}/{len(comparison)} experiments")

print(f"\nAverage Val RMSE Gain:  {comparison['val_rmse_gain'].mean():.2f}%")
print(f"Average Test RMSE Gain: {comparison['test_rmse_gain'].mean():.2f}%")
print(f"Average Test AUC Gain:  {comparison['test_auc_rmse_gain'].mean():.2f}%")

print(f"\nBest Val RMSE Gain:  {comparison['val_rmse_gain'].max():.2f}% @ {comparison.loc[comparison['val_rmse_gain'].idxmax(), 'data_pct']:.0f}%")
print(f"Best Test RMSE Gain: {comparison['test_rmse_gain'].max():.2f}% @ {comparison.loc[comparison['test_rmse_gain'].idxmax(), 'data_pct']:.0f}%")

print(f"\nWorst Val RMSE Gain:  {comparison['val_rmse_gain'].min():.2f}% @ {comparison.loc[comparison['val_rmse_gain'].idxmin(), 'data_pct']:.0f}%")
print(f"Worst Test RMSE Gain: {comparison['test_rmse_gain'].min():.2f}% @ {comparison.loc[comparison['test_rmse_gain'].idxmin(), 'data_pct']:.0f}%")

# Key insights
print("\n" + "="*40)
print("KEY INSIGHTS")
print("="*40)

if comparison['test_rmse_gain'].mean() > 10:
    print("EXCELLENT: TL3 shows strong positive transfer at very low data!")
    print(f"Average test RMSE improvement: {comparison['test_rmse_gain'].mean():.1f}%")
    print("This validates that Transfer Learning is critical for data scarcity scenarios.")
elif comparison['test_rmse_gain'].mean() > 0:
    print("POSITIVE: TL3 shows modest positive transfer at very low data.")
    print(f"Average test RMSE improvement: {comparison['test_rmse_gain'].mean():.1f}%")
    print("Transfer Learning provides some benefit, but improvements are limited.")
else:
    print("NEGATIVE: TL3 still shows negative transfer even at very low data.")
    print(f"Average test RMSE degradation: {comparison['test_rmse_gain'].mean():.1f}%")
    print("Consider:")
    print("- Different source dataset")
    print("- Domain adaptation techniques")
    print("- Literature-based TL approaches (TL4)")

# Save comparison
comparison_path = Path(__file__).parent.parent.parent / 'Results' / 'Phase4_Transfer_Learning' / 'Baseline_vs_TL3_low_data_comparison.csv'
comparison.to_csv(comparison_path, index=False)
print(f"\n Comparison saved: {comparison_path}")

print("="*40 + "\n")

