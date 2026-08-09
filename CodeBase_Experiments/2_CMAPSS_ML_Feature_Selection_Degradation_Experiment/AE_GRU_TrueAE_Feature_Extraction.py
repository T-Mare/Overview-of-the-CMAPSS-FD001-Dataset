import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.pruners import MedianPruner
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from Utilities.Plots_Metrics import cmapss_score, rmse_by_bins_with_auc
from Utilities.config import (
    AE_LSTM_FIXED_PARAMS,
    AE_LSTM_SEARCH_SPACE,
    ENGINE_COUNTS_ALL,
    OPTUNA_N_TRIALS_PHASE3,
    RANDOM_SEED,
    RUL_BINS,
    WINDOW_SIZE,
)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========
# CONFIGURATION
# ==========

DATASET = "FD001"

DATA_DIR = (
    PROJECT_ROOT
    / "CodeBase_Experiments"
    / "0_Data_Processing"
    / "Data_CMAPSS"
    / "2_Cleaned_Data"
    / "Windowed"
)
PHASE2_DIR = PROJECT_ROOT / "Results" / "Phase2_Feature_Selection" / DATASET
OUTPUT_DIR = PROJECT_ROOT / "Results" / "Phase3_Feature_Extraction" / DATASET / "AE_GRU_TrueAE"

FS_METHOD = "Correlation_FS"
SELECTED_FEATURES_FILE = PHASE2_DIR / "Feature_Analysis" / "correlation_based" / "selected_features.txt"

GRU_FIXED_HPS = None

N_TRIALS = OPTUNA_N_TRIALS_PHASE3
BATCH_SIZE = AE_LSTM_FIXED_PARAMS["batch_size"]
MAX_EPOCHS = AE_LSTM_FIXED_PARAMS["max_epochs"]
PATIENCE = AE_LSTM_FIXED_PARAMS["early_stopping_patience"]
ENCODER_DROPOUT = AE_LSTM_FIXED_PARAMS["dropout"]
LAMBDA_RECON_SEARCH_SPACE = [0.01, 0.05, 0.1]
DEFAULT_LAMBDA_RECON = 0.1
ACTIVATION_SEARCH_SPACE = ["relu"]

# ==========
# LOAD PHASE 1 GRU HYPERPARAMETERS
# ==========

def load_phase1_gru_hps():
    print("\nLoading Phase 1 GRU hyperparameters...")

    gru_hp_file = PROJECT_ROOT / "Results" / "Hyperparameter_Tuning" / "GRU" / "GRU_best_hyperparameters.json"

    if gru_hp_file.exists():
        with open(gru_hp_file, "r") as f:
            raw_hps = json.load(f)

        gru_hps = {
            "units": raw_hps.get("units_layer1", 128),
            "num_layers": raw_hps.get("n_gru_layers", 1),
            "dropout": raw_hps.get("dropout_rate", 0.1),
            "learning_rate": raw_hps.get("learning_rate", 0.001),
            "batch_size": raw_hps.get("batch_size", 32),
        }
        print(f"Loaded GRU HPs from Hyperparameter_Tuning: {gru_hps}")
        return gru_hps

    print("Phase 1 GRU HPs not found, using defaults")
    return {
        "units": 128,
        "num_layers": 1,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 32,
    }

def load_selected_features():
    print("\nLoading selected features from Phase 2...")

    if SELECTED_FEATURES_FILE.exists():
        with open(SELECTED_FEATURES_FILE, "r") as f:
            features = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(features)} features: {features}")
        return features

    print("Selected features file not found")
    return None

# ==========
# DATA LOADING
# ==========

def load_windowed_data(n_engines, selected_features):
    print(f"\nLoading data with {n_engines} engines...")

    X_train_full = np.load(DATA_DIR / f"{DATASET}_X_train_windowed.npy")
    y_train_full = np.load(DATA_DIR / f"{DATASET}_y_train_windowed.npy")
    X_val = np.load(DATA_DIR / f"{DATASET}_X_val_windowed.npy")
    y_val = np.load(DATA_DIR / f"{DATASET}_y_val_windowed.npy")
    X_test = np.load(DATA_DIR / f"{DATASET}_X_test_windowed.npy")
    y_test = np.load(DATA_DIR / f"{DATASET}_y_test_windowed.npy")

    train_ids = pd.read_csv(DATA_DIR / f"{DATASET}_train_ids_windowed.csv")

    print(f"Full train: {X_train_full.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    unique_engines = train_ids["engine"].unique()
    n_total_engines = len(unique_engines)

    if n_engines < n_total_engines:
        n_selected = min(n_engines, n_total_engines)

        np.random.seed(RANDOM_SEED)
        sampled_engines = np.random.choice(unique_engines, size=n_selected, replace=False)
        sampled_engines = sorted(sampled_engines)

        mask = train_ids["engine"].isin(sampled_engines).values
        train_X = X_train_full[mask]
        train_y = y_train_full[mask]
        print(f"Using {n_engines} engines: {train_X.shape}")
    else:
        train_X = X_train_full
        train_y = y_train_full
        print(f"Using all {n_total_engines} engines: {train_X.shape}")

    all_feature_names = [
        "os1", "os2", "os3", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8",
        "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18",
        "s19", "s20", "s21",
    ]

    feature_indices = [i for i, feat in enumerate(all_feature_names) if feat in selected_features]

    if len(feature_indices) != len(selected_features):
        print(f"Warning: Only found {len(feature_indices)}/{len(selected_features)} features")
        print(f"Available features: {all_feature_names}")
        print(f"Requested features: {selected_features}")
    else:
        print(f"Selected {len(feature_indices)} features: {[all_feature_names[i] for i in feature_indices]}")

    train_X = train_X[:, :, feature_indices]
    val_X = X_val[:, :, feature_indices]
    test_X = X_test[:, :, feature_indices]

    print(f"Final shape: {train_X.shape} (samples, timesteps={WINDOW_SIZE}, features={len(feature_indices)})")

    return (train_X, train_y), (val_X, y_val), (test_X, y_test)

# ==========
# MODEL BUILDING
# ==========

def build_true_ae_gru_model(input_shape, ae_config, gru_config, lambda_recon=DEFAULT_LAMBDA_RECON):
    _, n_features = input_shape

    inputs = keras.Input(shape=input_shape, name="input_sequence")

    # Encoder: feature vector at each timestep -> latent vector at each timestep.
    encoded = inputs
    for i in range(ae_config["encoder_layers"]):
        encoded = layers.TimeDistributed(
            layers.Dense(
                ae_config["encoder_units"],
                activation=ae_config["activation"],
                name=f"encoder_dense_{i + 1}",
            ),
            name=f"td_encoder_{i + 1}",
        )(encoded)

        if ENCODER_DROPOUT > 0:
            encoded = layers.TimeDistributed(
                layers.Dropout(ENCODER_DROPOUT, name=f"encoder_dropout_{i + 1}"),
                name=f"td_encoder_dropout_{i + 1}",
            )(encoded)

    latent = layers.TimeDistributed(
        layers.Dense(
            ae_config["bottleneck_dim"],
            activation=ae_config["activation"],
            name="bottleneck",
        ),
        name="latent_sequence",
    )(encoded)

    # Decoder: latent sequence -> reconstructed selected-feature sequence.
    decoded = latent
    for i in range(ae_config["encoder_layers"]):
        decoded = layers.TimeDistributed(
            layers.Dense(
                ae_config["encoder_units"],
                activation=ae_config["activation"],
                name=f"decoder_dense_{i + 1}",
            ),
            name=f"td_decoder_{i + 1}",
        )(decoded)

    reconstruction_output = layers.TimeDistributed(
        layers.Dense(n_features, activation="linear", name="decoder_reconstruction_dense"),
        name="reconstruction_output",
    )(decoded)

    # GRU predictor: latent sequence -> RUL prediction.
    x = latent
    for i in range(gru_config["num_layers"]):
        return_sequences = i < gru_config["num_layers"] - 1
        x = layers.GRU(
            units=gru_config["units"],
            return_sequences=return_sequences,
            name=f"gru_{i + 1}",
        )(x)

        if gru_config["dropout"] > 0:
            x = layers.Dropout(gru_config["dropout"], name=f"gru_dropout_{i + 1}")(x)

    rul_output = layers.Dense(1, activation="linear", name="rul_output")(x)

    model = Model(
        inputs=inputs,
        outputs=[rul_output, reconstruction_output],
        name="AE_GRU_TrueAE",
    )

    optimizer = keras.optimizers.Adam(learning_rate=gru_config["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss={
            "rul_output": "mse",
            "reconstruction_output": "mse",
        },
        loss_weights={
            "rul_output": 1.0,
            "reconstruction_output": lambda_recon,
        },
        metrics={
            "rul_output": ["mae"],
            "reconstruction_output": ["mse"],
        },
    )

    print(
        "  True AE-GRU: "
        f"{n_features} features -> {ae_config['bottleneck_dim']} latent features -> "
        f"{n_features} reconstructed features; lambda_recon={lambda_recon}"
    )

    return model

# ==========
# TRAINING & EVALUATION
# ==========

def make_targets(X, y):
    return {
        "rul_output": y,
        "reconstruction_output": X,
    }

def train_model(model, train_data, val_data):
    train_X, train_y = train_data
    val_X, val_y = val_data

    early_stop = EarlyStopping(
        monitor="val_rul_output_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        train_X,
        make_targets(train_X, train_y),
        validation_data=(val_X, make_targets(val_X, val_y)),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )

    epochs_trained = len(history.history["loss"])
    return history, epochs_trained

def predict_outputs(model, X):
    predictions = model.predict(X, verbose=0)

    if isinstance(predictions, dict):
        rul_pred = predictions["rul_output"]
        reconstruction_pred = predictions["reconstruction_output"]
    else:
        rul_pred, reconstruction_pred = predictions

    return rul_pred.flatten(), reconstruction_pred

def reconstruction_mse(X_true, X_recon):
    return float(np.mean((X_true - X_recon) ** 2))

def evaluate_model(model, train_data, val_data, test_data):
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data

    train_pred, train_recon = predict_outputs(model, train_X)
    val_pred, val_recon = predict_outputs(model, val_X)
    test_pred, test_recon = predict_outputs(model, test_X)

    train_rmse = np.sqrt(np.mean((train_y - train_pred) ** 2))
    val_rmse = np.sqrt(np.mean((val_y - val_pred) ** 2))
    test_rmse = np.sqrt(np.mean((test_y - test_pred) ** 2))

    train_mae = np.mean(np.abs(train_y - train_pred))
    val_mae = np.mean(np.abs(val_y - val_pred))
    test_mae = np.mean(np.abs(test_y - test_pred))

    train_r2 = 1 - (np.sum((train_y - train_pred) ** 2) / np.sum((train_y - np.mean(train_y)) ** 2))
    val_r2 = 1 - (np.sum((val_y - val_pred) ** 2) / np.sum((val_y - np.mean(val_y)) ** 2))
    test_r2 = 1 - (np.sum((test_y - test_pred) ** 2) / np.sum((test_y - np.mean(test_y)) ** 2))

    train_cmapss = cmapss_score(train_y, train_pred)
    val_cmapss = cmapss_score(val_y, val_pred)
    test_cmapss = cmapss_score(test_y, test_pred)

    _, train_auc = rmse_by_bins_with_auc(train_y, train_pred, RUL_BINS)
    _, val_auc = rmse_by_bins_with_auc(val_y, val_pred, RUL_BINS)
    _, test_auc = rmse_by_bins_with_auc(test_y, test_pred, RUL_BINS)

    return {
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "val_r2": val_r2,
        "test_r2": test_r2,
        "train_cmapss": train_cmapss,
        "val_cmapss": val_cmapss,
        "test_cmapss": test_cmapss,
        "train_auc_rmse": train_auc,
        "val_auc_rmse": val_auc,
        "test_auc_rmse": test_auc,
        "train_reconstruction_mse": reconstruction_mse(train_X, train_recon),
        "val_reconstruction_mse": reconstruction_mse(val_X, val_recon),
        "test_reconstruction_mse": reconstruction_mse(test_X, test_recon),
    }

# ==========
# HYPERPARAMETER SEARCH WITH OPTUNA
# ==========

def suggest_ae_config(trial):
    ae_config = {}
    for param_name, (param_type, param_values) in AE_LSTM_SEARCH_SPACE.items():
        if param_name == "activation":
            ae_config[param_name] = trial.suggest_categorical(param_name, ACTIVATION_SEARCH_SPACE)
            continue

        if param_type == "categorical":
            ae_config[param_name] = trial.suggest_categorical(param_name, param_values)
        elif param_type == "int":
            ae_config[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
        elif param_type == "float":
            ae_config[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
        elif param_type == "float_log":
            ae_config[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1], log=True)

    ae_config["lambda_recon"] = trial.suggest_categorical("lambda_recon", LAMBDA_RECON_SEARCH_SPACE)
    return ae_config

def optuna_objective(trial, train_data, val_data, gru_config, selected_features):
    ae_config = suggest_ae_config(trial)
    input_shape = (WINDOW_SIZE, len(selected_features))

    try:
        model = build_true_ae_gru_model(
            input_shape,
            ae_config,
            gru_config,
            ae_config["lambda_recon"],
        )
        train_model(model, train_data, val_data)

        val_predictions, _ = predict_outputs(model, val_data[0])
        val_rmse = np.sqrt(np.mean((val_data[1] - val_predictions) ** 2))

        del model
        keras.backend.clear_session()

        return val_rmse
    except Exception as e:
        print(f"Trial failed: {e}")
        keras.backend.clear_session()
        return float("inf")

def optimize_ae_hps(train_data, val_data, gru_config, selected_features):
    print("\n" + "="*40)
    print("TRUE AE-GRU HYPERPARAMETER OPTIMIZATION WITH OPTUNA (80 Engines)")
    print("="*40)
    print(f"\nNumber of trials: {N_TRIALS}")
    print(f"lambda_recon search space: {LAMBDA_RECON_SEARCH_SPACE}")
    print("\nSearch space (from config.py):")
    for param_name, (_, param_values) in AE_LSTM_SEARCH_SPACE.items():
        if param_name == "activation":
            param_values = ACTIVATION_SEARCH_SPACE
        print(f"{param_name}: {param_values}")
    print(f"lambda_recon: {LAMBDA_RECON_SEARCH_SPACE}")

    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name="AE_GRU_TrueAE_HP_Optimization",
    )

    study.optimize(
        lambda trial: optuna_objective(trial, train_data, val_data, gru_config, selected_features),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_trial = study.best_trial
    best_config = best_trial.params
    best_val_rmse = best_trial.value

    print(f"\n{'='*40}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*40}")
    print(f"Best Val RMSE: {best_val_rmse:.4f}")
    print("Best configuration:")
    for key, value in best_config.items():
        print(f"{key}: {value}")
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trial: #{best_trial.number}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trials_df = study.trials_dataframe()
    trials_file = OUTPUT_DIR / "optuna_trials.csv"
    trials_df.to_csv(trials_file, index=False)
    print(f"\nAll trials saved: {trials_file}")

    best_config_file = OUTPUT_DIR / "best_ae_config.json"
    with open(best_config_file, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Best config saved: {best_config_file}")

    study_file = OUTPUT_DIR / "optuna_study.pkl"
    with open(study_file, "wb") as f:
        pickle.dump(study, f)
    print(f"Study object saved: {study_file}")

    optimization_history = []
    for trial in study.trials:
        finite_values = [t.value for t in study.trials[: trial.number + 1] if t.value != float("inf")]
        optimization_history.append({
            "trial_number": trial.number,
            "value": trial.value,
            "best_value_so_far": min(finite_values) if finite_values else None,
            "duration_sec": trial.duration.total_seconds() if trial.duration else None,
            "state": trial.state.name,
            **trial.params,
        })
    opt_history_df = pd.DataFrame(optimization_history)
    opt_history_file = OUTPUT_DIR / "optimization_history.csv"
    opt_history_df.to_csv(opt_history_file, index=False)
    print(f"Optimization history saved: {opt_history_file}")

    if len(study.trials) >= 10:
        try:
            from optuna.importance import get_param_importances

            importances = get_param_importances(study)
            importance_df = pd.DataFrame([
                {"parameter": param, "importance": imp}
                for param, imp in importances.items()
            ]).sort_values("importance", ascending=False)
            importance_file = OUTPUT_DIR / "parameter_importance.csv"
            importance_df.to_csv(importance_file, index=False)
            print(f"Parameter importance saved: {importance_file}")
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")

    return best_config, study

# ==========
# EVALUATE ACROSS ENGINE COUNTS
# ==========

def save_training_history(history, epochs, n_engines):
    history_data = {
        "epoch": range(1, epochs + 1),
        "train_loss": history.history.get("loss", [None] * epochs),
        "val_loss": history.history.get("val_loss", [None] * epochs),
        "train_rul_loss": history.history.get("rul_output_loss", [None] * epochs),
        "val_rul_loss": history.history.get("val_rul_output_loss", [None] * epochs),
        "train_reconstruction_loss": history.history.get("reconstruction_output_loss", [None] * epochs),
        "val_reconstruction_loss": history.history.get("val_reconstruction_output_loss", [None] * epochs),
        "train_rul_mae": history.history.get("rul_output_mae", [None] * epochs),
        "val_rul_mae": history.history.get("val_rul_output_mae", [None] * epochs),
    }

    history_df = pd.DataFrame(history_data)
    history_dir = OUTPUT_DIR / "training_history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(history_dir / f"history_{n_engines}engines.csv", index=False)

def save_latent_outputs(model, train_data, val_data, test_data, n_engines):
    latent_model = Model(
        inputs=model.input,
        outputs=model.get_layer("latent_sequence").output,
    )

    split_data = {
        "train": train_data[0],
        "val": val_data[0],
        "test": test_data[0],
    }

    latent_dir = OUTPUT_DIR / "latent_features"
    corr_dir = OUTPUT_DIR / "latent_correlations"
    latent_dir.mkdir(parents=True, exist_ok=True)
    corr_dir.mkdir(parents=True, exist_ok=True)

    for split_name, X in split_data.items():
        latent = latent_model.predict(X, verbose=0)
        np.save(latent_dir / f"{split_name}_latent_{n_engines}engines.npy", latent)

        latent_2d = latent.reshape(-1, latent.shape[-1])
        latent_columns = [f"z{i + 1}" for i in range(latent.shape[-1])]
        latent_corr = pd.DataFrame(latent_2d, columns=latent_columns).corr()
        latent_corr.to_csv(corr_dir / f"{split_name}_latent_corr_{n_engines}engines.csv")

    print(f"Latent features saved: {latent_dir}")
    print(f"Latent correlation matrices saved: {corr_dir}")

def save_predictions(model, train_data, val_data, test_data, n_engines):
    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data
    
    # Get predictions (handle multi-output model)
    train_pred, _ = predict_outputs(model, train_X)
    val_pred, _ = predict_outputs(model, val_X)
    test_pred, _ = predict_outputs(model, test_X)
    
    # Create predictions directory
    pred_dir = OUTPUT_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV files (same format as baseline models)
    pd.DataFrame({'y_true': train_y, 'y_pred': train_pred}).to_csv(
        pred_dir / f'train_{n_engines}engines.csv', index=False)
    pd.DataFrame({'y_true': val_y, 'y_pred': val_pred}).to_csv(
        pred_dir / f'val_{n_engines}engines.csv', index=False)
    pd.DataFrame({'y_true': test_y, 'y_pred': test_pred}).to_csv(
        pred_dir / f'test_{n_engines}engines.csv', index=False)
    
    print(f"Predictions saved: {pred_dir}")

def evaluate_all_engine_counts(best_ae_config, gru_config, selected_features, engine_counts):
    print("\n" + "="*40)
    print("EVALUATING BEST TRUE AE-GRU CONFIG ACROSS ALL ENGINE COUNTS")
    print("="*40)

    all_results = []
    input_shape = (WINDOW_SIZE, len(selected_features))
    best_lambda_recon = best_ae_config.get("lambda_recon", DEFAULT_LAMBDA_RECON)

    for n_engines in engine_counts:
        print(f"\n{'='*40}")
        print(f"ENGINE COUNT: {n_engines} engines")
        print(f"{'='*40}")

        try:
            train_data, val_data, test_data = load_windowed_data(n_engines, selected_features)

            model = build_true_ae_gru_model(input_shape, best_ae_config, gru_config, best_lambda_recon)

            start_time = time.time()
            history, epochs = train_model(model, train_data, val_data)
            training_time = time.time() - start_time

            print(f"Training complete: {epochs} epochs, {training_time:.1f}s")

            save_training_history(history, epochs, n_engines)

            metrics = evaluate_model(model, train_data, val_data, test_data)
            save_latent_outputs(model, train_data, val_data, test_data, n_engines)
            
            # Save predictions for AUC-RMSE curve plotting
            save_predictions(model, train_data, val_data, test_data, n_engines)

            result = {
                "model_name": "AE_GRU_TrueAE",
                "n_engines": n_engines,
                **best_ae_config,
                "lambda_recon": best_lambda_recon,
                **metrics,
                "epochs_trained": epochs,
                "training_time_sec": training_time,
            }
            all_results.append(result)

            checkpoint_df = pd.DataFrame(all_results)
            checkpoint_file = OUTPUT_DIR / "ae_gru_trueae_all_results_checkpoint.csv"
            checkpoint_df.to_csv(checkpoint_file, index=False)

            print(f"\nResults for {n_engines} engines:")
            print(f"Val RMSE:  {metrics['val_rmse']:.4f}")
            print(f"Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"Val R2:    {metrics['val_r2']:.4f}")
            print(f"Test R2:   {metrics['test_r2']:.4f}")
            print(f"Val recon MSE: {metrics['val_reconstruction_mse']:.6f}")
            print(f"Checkpoint saved ({len(all_results)}/{len(engine_counts)} engine counts complete)")

            model_dir = OUTPUT_DIR / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f"ae_gru_trueae_{n_engines}engines.keras")

            del model
            keras.backend.clear_session()

        except Exception as e:
            print(f"Error at {n_engines} engines: {e}")
            keras.backend.clear_session()
            continue

    results_df = pd.DataFrame(all_results)
    results_file = OUTPUT_DIR / "ae_gru_trueae_all_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nAll results saved: {results_file}")

    if not results_df.empty:
        summary_stats = results_df.groupby("n_engines").agg({
            "val_rmse": ["mean", "std", "min", "max"],
            "test_rmse": ["mean", "std", "min", "max"],
            "val_r2": ["mean", "std", "min", "max"],
            "val_reconstruction_mse": ["mean", "std", "min", "max"],
            "training_time_sec": ["mean", "std", "min", "max"],
        }).round(4)
        summary_file = OUTPUT_DIR / "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved: {summary_file}")

        metrics_comparison = results_df[[
            "n_engines",
            "val_rmse",
            "test_rmse",
            "val_mae",
            "test_mae",
            "val_r2",
            "test_r2",
            "val_cmapss",
            "test_cmapss",
            "val_auc_rmse",
            "test_auc_rmse",
            "val_reconstruction_mse",
            "test_reconstruction_mse",
            "training_time_sec",
            "epochs_trained",
        ]].copy()
        metrics_file = OUTPUT_DIR / "metrics_comparison.csv"
        metrics_comparison.to_csv(metrics_file, index=False)
        print(f"Metrics comparison saved: {metrics_file}")

    return results_df

# ==========
# MAIN EXECUTION
# ==========

def main(test_mode=False):
    print("\n" + "="*40)
    print("TRUE AE-GRU FEATURE EXTRACTION EXPERIMENT")
    if test_mode:
        print("(TEST MODE: 2 trials, 80 engines, 2 epochs)")
    print("="*40)
    print(f"\nDataset: {DATASET}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Feature selection: {FS_METHOD}")
    print(f"Output directory: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    global N_TRIALS, MAX_EPOCHS
    if test_mode:
        N_TRIALS = 2
        MAX_EPOCHS = 2
        engine_counts_to_use = [80]
        print("\nTEST MODE ACTIVE:")
        print(f"N_TRIALS: {N_TRIALS}")
        print(f"MAX_EPOCHS: {MAX_EPOCHS}")
        print(f"ENGINE_COUNTS: {engine_counts_to_use}")
    else:
        engine_counts_to_use = ENGINE_COUNTS_ALL

    global GRU_FIXED_HPS
    GRU_FIXED_HPS = load_phase1_gru_hps()
    selected_features = load_selected_features()

    if selected_features is None:
        print("Cannot proceed without selected features")
        return

    print(f"\nFixed GRU config: {GRU_FIXED_HPS}")
    print(f"Selected features: {len(selected_features)} features")
    print(f"Reconstruction loss weight search space: {LAMBDA_RECON_SEARCH_SPACE}")

    # Check if we already have best config (skip Optuna if we do)
    best_config_file = OUTPUT_DIR / "best_ae_config.json"
    if best_config_file.exists():
        print("\n" + "="*40)
        print("FOUND EXISTING BEST CONFIG - SKIPPING OPTUNA")
        print("="*40)
        with open(best_config_file, 'r') as f:
            best_ae_config = json.load(f)
        print(f"\nLoaded best config: {best_ae_config}")
        study = None  # No study object when loading from file
    else:
        print("\n" + "="*40)
        print("STEP 1: HYPERPARAMETER OPTIMIZATION (80 ENGINES)")
        print("="*40)

        train_data_80, val_data_80, _ = load_windowed_data(80, selected_features)
        best_ae_config, study = optimize_ae_hps(
            train_data_80,
            val_data_80,
            GRU_FIXED_HPS,
            selected_features,
        )

    print("\n" + "="*40)
    print("STEP 2: EVALUATE ON ALL ENGINE COUNTS")
    print("="*40)

    evaluate_all_engine_counts(best_ae_config, GRU_FIXED_HPS, selected_features, engine_counts_to_use)

    metadata = {
        "experiment_name": "AE_GRU_TrueAE_Feature_Extraction",
        "dataset": DATASET,
        "window_size": WINDOW_SIZE,
        "feature_selection_method": FS_METHOD,
        "num_selected_features": len(selected_features),
        "selected_features": selected_features,
        "engine_counts": engine_counts_to_use,
        "optuna_trials": N_TRIALS,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "early_stopping_monitor": "val_rul_output_loss",
        "activation_search_space": ACTIVATION_SEARCH_SPACE,
        "lambda_recon_search_space": LAMBDA_RECON_SEARCH_SPACE,
        "best_lambda_recon": best_ae_config.get("lambda_recon", DEFAULT_LAMBDA_RECON),
        "loss_function": "total_loss = RUL_MSE + lambda_recon * reconstruction_MSE",
        "selection_objective": "validation RUL RMSE",
        "gru_fixed_config": GRU_FIXED_HPS,
        "best_ae_config": best_ae_config,
        "best_val_rmse_80engines": study.best_value if study else "loaded_from_file",
        "total_optuna_trials_completed": len(study.trials) if study else "loaded_from_file",
        "experiment_timestamp": pd.Timestamp.now().isoformat(),
    }
    metadata_file = OUTPUT_DIR / "experiment_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nExperiment metadata saved: {metadata_file}")

    print("\n" + "="*40)
    print("TRUE AE-GRU EXPERIMENT COMPLETE")
    print("="*40)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("- experiment_metadata.json")
    print("- optuna_trials.csv")
    print("- optimization_history.csv")
    print("- parameter_importance.csv")
    print("- optuna_study.pkl")
    print("- best_ae_config.json")
    print("- ae_gru_trueae_all_results.csv")
    print("- metrics_comparison.csv")
    print("- summary_statistics.csv")
    print("- training_history/")
    print("- latent_features/")
    print("- latent_correlations/")
    print("- models/")

if __name__ == "__main__":
    test_mode = "--test" in sys.argv or "-t" in sys.argv
    main(test_mode=test_mode)
