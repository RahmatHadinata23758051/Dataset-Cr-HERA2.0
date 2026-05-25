import pandas as pd
import numpy as np
import os
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, max_error

# ML Algorithms
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model across 7 distinct dimensions including metrics and inference latency.
    """
    n = len(y_test)
    p = X_test.shape[1]
    
    # 1. Benchmark Inference Latency
    # We predict the entire set row-by-row to simulate real-world IoT streaming latency
    latencies = []
    for i in range(min(n, 500)):  # Benchmark over 500 samples to get a highly stable average
        row = X_test[i:i+1]
        start = time.perf_counter()
        _ = model.predict(row)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # Convert to milliseconds
        
    avg_latency_ms = np.mean(latencies)
    
    # Predict full test set for metrics
    y_pred = model.predict(X_test)
    
    # 2. Compute Performance Metrics
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100.0  # Convert to percentage
    max_err = max_error(y_test, y_pred)
    
    return {
        "R2": r2,
        "Adj_R2": adj_r2,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE_pct": mape,
        "Max_Error": max_err,
        "Latency_ms": avg_latency_ms
    }

def main():
    print("[INFO] Starting HERA 2.0 Multi-Model Training & Benchmarking Pipeline...")
    
    # Resolve relative paths relative to this script's directory for robust execution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset", "dataset_heavy_metal_grounded.csv"))
    results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Grounded dataset '{dataset_path}' not found! Run generate_dataset.py first.")
        return
        
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Loaded grounded dataset with shape: {df.shape}")
    
    # Define physical features (inputs)
    features = ["pH", "EC_uScm", "TDS_mgL", "Suhu_Air"]
    X = df[features].values
    
    # Define target metals
    targets = {
        "Chromium": df["Chromium_ugL"].values,
        "Nickel": df["Nickel_ugL"].values
    }
    
    # Set up the 5 distinct ML algorithms to benchmark
    # Carefully regularized to prevent overfitting on the grounded dataset
    def get_models():
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0),
            "SVR (RBF Kernel)": SVR(C=20.0, epsilon=0.1, kernel="rbf"),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "XGBoost Regressor": XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
        }
        
    for metal, y in targets.items():
        print(f"\n=======================================================")
        print(f" TRAINING & EVALUATING MODELS FOR: {metal.upper()}")
        print(f"=======================================================")
        
        # Split into 80% train/validation and 20% holdout test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"[INFO] Train size: {X_train.shape[0]} | Holdout test size: {X_test.shape[0]}")
        
        # Scale features using StandardScaler (crucial for distance/linear based models SVR, Ridge, LinReg)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = get_models()
        metal_results = []
        best_r2 = -float("inf")
        best_model_name = None
        best_model = None
        
        for name, model in models.items():
            print(f"[RUN] Training {name}...")
            
            # Tree-based algorithms don't strictly require scaling, but we scale consistently for ease
            start_fit = time.perf_counter()
            model.fit(X_train_scaled, y_train)
            end_fit = time.perf_counter()
            fit_time = (end_fit - start_fit) * 1000.0
            
            # Evaluate model across the 7 dimensions
            metrics = evaluate_model(model, X_test_scaled, y_test)
            metrics["Model"] = name
            metrics["Fit_Time_ms"] = fit_time
            
            metal_results.append(metrics)
            
            print(f"      -> R2: {metrics['R2']:.4f} | RMSE: {metrics['RMSE']:.4f} ug/L | Latency: {metrics['Latency_ms']:.4f} ms")
            
            # Select best model based on R2 (and robustness check)
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_model_name = name
                best_model = model
                
        # Convert results to DataFrame and save to CSV
        df_results = pd.DataFrame(metal_results)
        # Reorder columns for beautiful presentation
        cols = ["Model", "R2", "Adj_R2", "RMSE", "MAE", "MAPE_pct", "Max_Error", "Latency_ms", "Fit_Time_ms"]
        df_results = df_results[cols].sort_values(by="R2", ascending=False)
        
        result_file = os.path.join(results_dir, f"{metal.lower()}_model_comparison.csv")
        df_results.to_csv(result_file, index=False)
        print(f"\n[OK] Saved comprehensive metrics for {metal} to: {result_file}")
        
        # Print the comparative table in stdout
        print("\n--- Comparative Model Performance Matrix ---")
        print(df_results.to_string(index=False, formatters={
            "R2": "{:,.4f}".format, "Adj_R2": "{:,.4f}".format,
            "RMSE": "{:,.2f}".format, "MAE": "{:,.2f}".format,
            "MAPE_pct": "{:,.2f}%".format, "Max_Error": "{:,.2f}".format,
            "Latency_ms": "{:,.4f}".format, "Fit_Time_ms": "{:,.1f}".format
        }))
        
        print(f"\n[BEST] Selected Best Model: {best_model_name} (R2 = {best_r2:.4f})")
        
        # Serialize the best model and its associated scaler
        # We package both the scaler and model together to ensure seamless preprocessing during inference
        export_pack = {
            "scaler": scaler,
            "model": best_model,
            "features": features,
            "metal": metal,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        export_path = os.path.join(models_dir, f"best_model_{metal.lower()}.pkl")
        with open(export_path, "wb") as f:
            pickle.dump(export_pack, f)
        print(f"[OK] Serialized best model for {metal} to: {export_path}")
        
    print("\n[OK] Model Training and Multi-Dimensional Evaluation Pipeline complete!")

if __name__ == "__main__":
    main()
