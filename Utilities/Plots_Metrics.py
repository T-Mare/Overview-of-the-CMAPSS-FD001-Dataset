import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

###### Metrics ######

#1. CMAPSS_Score

def cmapss_score(y_true, y_pred, reduction="sum"):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)

    if reduction == "mean":
        return float(np.mean(s))
    return float(np.sum(s))

#2. RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#3. MAE
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

#4. R²
def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

#5. MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# RMSE_by_bins_with_AUC

def rmse_by_bins_with_auc(y_true, y_pred, edges):
    # Convert to numpy arrays
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    edges  = np.asarray(edges,  dtype=np.float64)

    rmse_bins = [] #Store RMSE for each bin// will plot later

    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        # For last bin, include upper boundary to capture samples at max RUL
        if i == len(edges) - 2:  # Last bin
            mask = (y_true >= lo) & (y_true <= hi) # [lo, hi] inclusive
        else:
            mask = (y_true >= lo) & (y_true < hi) # [lo, hi) half-open
        rmse_bins.append( #Append the RMSE for this bin
            np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])) if mask.any() else np.nan # Compute MSE over samples in this bin (if no samples in the bin store as NaN)
        )

    rmse_bins = np.asarray(rmse_bins, dtype=np.float64) #Convert list of RMSE to np array for vector calc

    # Compute unweighted average AUC-RMSE (each bin treated equally regardless of sample count or bin width)
    valid = ~np.isnan(rmse_bins) # Only compute AUC if at least one bin has samples
    auc_rmse_norm = None 
    if valid.any():  
        auc_rmse_norm = float(np.mean(rmse_bins[valid])) # Simple average of bin RMSEs

    return rmse_bins.tolist(), auc_rmse_norm  # Return RMSE per bin and the unweighted average AUC summary

####Plots####

#1. Actual vs Predicted

def plot_actual_vs_predicted(y_actual, y_pred, dataset_name, model_name, save_path=None, show=False):
    y_actual_np = np.asarray(y_actual, dtype=np.float64).ravel()  # flatten to (n,)
    y_pred_np   = np.asarray(y_pred,   dtype=np.float64).ravel()  # flatten to (n,)

    if y_actual_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError(f"Length mismatch: y_actual={len(y_actual_np)} vs y_pred={len(y_pred_np)}")

    sample_index = np.arange(y_actual_np.shape[0])

    plt.figure(figsize=(12, 6))
    plt.plot(sample_index, y_actual_np, label="Actual RUL", linestyle='-')
    plt.plot(sample_index, y_pred_np,   label="Predicted RUL", linestyle='--')

    plt.xlabel("Sample Index (Test Set Order)")
    plt.ylabel("RUL (Remaining Useful Life)")
    plt.title(f"Test Set: Actual vs Predicted RUL - {model_name} on {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()

    plt.close()

#2. RMSE by bins
def plot_rmse_by_bins(edges, rmse_bins, auc_rmse_norm=None, model_name=None, save_path=None, show=False):
    edges = np.asarray(edges, dtype=np.float64)
    rmse_bins = np.asarray(rmse_bins, dtype=np.float64)

    valid = ~np.isnan(rmse_bins)
    if not valid.any():
        print("--- INFO: No valid RMSE values to plot. ---")
        return

    centers = (edges[:-1] + edges[1:]) / 2.0  # bin midpoints for plotting

    plt.figure(figsize=(8, 6))
    plt.plot(centers, rmse_bins, marker='o', linestyle='-')

    # label points
    for x, r in zip(centers, rmse_bins):
        if not np.isnan(r):
            plt.annotate(f"{r:.2f}", (x, r), textcoords="offset points", xytext=(0, 10), ha="center")

    plt.xlabel("RUL bin midpoint")
    plt.ylabel("RMSE (within bin)")

    title = "RMSE by RUL bin"
    if model_name:
        title += f" ({model_name})"
    if auc_rmse_norm is not None:
        title += f"\nAUC-RMSE: {auc_rmse_norm:.4f}"
    plt.title(title)

    plt.gca().invert_xaxis()  # matches your threshold plot style; remove if you want increasing left->right
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
        
    plt.close()

