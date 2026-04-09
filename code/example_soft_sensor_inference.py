#!/usr/bin/env python3
"""
Example: Using Soft Sensor Inference for Real-time Cr Estimation
Contoh penggunaan model soft sensor untuk memprediksi konsentrasi Cr
dengan data sensor input real atau synthetic
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from ml_pipeline_cr_soft_sensor import SoftSensorInference


def resolve_default_artifacts():
    """Resolve model/scaler using metadata produced by training pipeline."""
    models_dir = Path("models")
    metadata_path = models_dir / "best_model_metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model_name = metadata.get("model_name", metadata.get("model_type", "rf"))
        scenario = metadata.get("scenario", "full")

        model_path = models_dir / f"best_model_{model_name}_{scenario}.pkl"
        if not model_path.exists():
            model_path = models_dir / f"model_{model_name}_{scenario}.pkl"

        scaler_name = metadata.get("scaler_path")
        scaler_path = models_dir / scaler_name if scaler_name else None
        if scaler_path is not None and not scaler_path.exists():
            scaler_path = None

        return model_path, scaler_path

    model_path = models_dir / "best_model_rf_full.pkl"
    legacy_scaler = models_dir / "best_model_scaler_full.pkl"
    return model_path, legacy_scaler

# ============================================================================
# EXAMPLE 1: Single Prediction from Dict
# ============================================================================

def example_single_prediction():
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Prediction from Sensor Dict")
    print("="*70)
    
    # Load model
    model_path, scaler_path = resolve_default_artifacts()
    sensor = SoftSensorInference(str(model_path), str(scaler_path) if scaler_path else None)
    
    # Sensor input (contoh data dari sensor di lapangan)
    sensor_input = {
        'EC': 1200,                    # μS/cm
        'TDS': 850,                    # mg/L
        'pH': 6.8,
        'Suhu Air (°C)': 26.5,
        'Suhu Lingkungan (°C)': 28.3,
        'Kelembapan Lingkungan (%)': 72.1,
        'Tegangan (V)': 3.95
    }
    
    # Predict
    cr_pred = sensor.predict_single(sensor_input)
    
    print("\nSensor Input:")
    for key, val in sensor_input.items():
        print(f"  {key:.<30} {val}")
    
    print(f"\n🔮 Predicted Cr: {cr_pred:.2f} μg/L")
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE 2: Batch Prediction from CSV
# ============================================================================

def example_batch_prediction():
    print("="*70)
    print("EXAMPLE 2: Batch Prediction from Sample Data")
    print("="*70)
    
    # Load model
    model_path, scaler_path = resolve_default_artifacts()
    sensor = SoftSensorInference(str(model_path), str(scaler_path) if scaler_path else None)
    
    # Create sample sensor data (simulasi multiple sensor readings)
    sample_data = pd.DataFrame({
        'EC': [1000, 1200, 1500, 800, 1100],
        'TDS': [700, 850, 1050, 560, 770],
        'pH': [6.5, 6.8, 7.2, 6.3, 7.0],
        'Suhu Air (°C)': [24.0, 26.5, 28.0, 22.5, 25.5],
        'Suhu Lingkungan (°C)': [26.0, 28.3, 30.0, 24.5, 27.5],
        'Kelembapan Lingkungan (%)': [70.0, 72.1, 75.0, 68.0, 71.0],
        'Tegangan (V)': [3.85, 3.95, 4.05, 3.75, 3.90]
    })
    
    # Batch predict
    cr_predictions = sensor.predict_batch(sample_data)
    
    # Add predictions to dataframe
    sample_data['Cr_Predicted'] = cr_predictions
    
    print("\nSample Sensor Data with Cr Predictions:")
    print(sample_data.to_string(index=False))
    
    print(f"\nPrediction Statistics:")
    print(f"  Mean Cr:  {cr_predictions.mean():.2f} μg/L")
    print(f"  Std Cr:   {cr_predictions.std():.2f} μg/L")
    print(f"  Min Cr:   {cr_predictions.min():.2f} μg/L")
    print(f"  Max Cr:   {cr_predictions.max():.2f} μg/L")
    
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE 3: Time-series Simulation
# ============================================================================

def example_timeseries_simulation():
    print("="*70)
    print("EXAMPLE 3: Time-Series Monitoring Simulation")
    print("="*70)
    
    # Load model
    model_path, scaler_path = resolve_default_artifacts()
    sensor = SoftSensorInference(str(model_path), str(scaler_path) if scaler_path else None)
    
    # Simulate hourly sensor readings (24 jam)
    np.random.seed(42)
    hours = np.arange(24)
    
    # Simulate daily variations in water quality
    ec_values = 1200 + 200 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 50, 24)
    tds_values = 850 + 150 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 30, 24)
    ph_values = 6.8 + 0.2 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.1, 24)
    temp_air = 26 + 3 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.5, 24)
    temp_env = 28 + 3 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 0.5, 24)
    humidity = 72 + 10 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 2, 24)
    voltage = 3.95 + 0.05 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 0.01, 24)
    
    # Create dataframe
    timeseries_data = pd.DataFrame({
        'Hour': hours,
        'EC': ec_values,
        'TDS': tds_values,
        'pH': ph_values,
        'Suhu Air (°C)': temp_air,
        'Suhu Lingkungan (°C)': temp_env,
        'Kelembapan Lingkungan (%)': humidity,
        'Tegangan (V)': voltage
    })
    
    # Predict
    cr_predictions = sensor.predict_batch(timeseries_data)
    timeseries_data['Cr'] = cr_predictions
    
    print("\nHourly Monitoring Data (24-hour simulation):")
    print(timeseries_data.to_string(index=False))
    
    print(f"\n24-Hour Statistics:")
    print(f"  Avg Cr:        {cr_predictions.mean():.2f} μg/L")
    print(f"  Peak Cr:       {cr_predictions.max():.2f} μg/L (Hour {cr_predictions.argmax()})")
    print(f"  Minimum Cr:    {cr_predictions.min():.2f} μg/L (Hour {cr_predictions.argmin()})")
    print(f"  Std Dev:       {cr_predictions.std():.2f} μg/L")
    print(f"  Range:         {cr_predictions.max() - cr_predictions.min():.2f} μg/L")
    
    print("="*70 + "\n")


# ============================================================================
# EXAMPLE 4: Demo Sanity Check (Not Independent Benchmark)
# ============================================================================

def example_comparison_with_real_data():
    print("="*70)
    print("EXAMPLE 4: Demo Sanity Check (Bukan Benchmark Utama)")
    print("="*70)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    # Load model
    model_path, scaler_path = resolve_default_artifacts()
    sensor = SoftSensorInference(str(model_path), str(scaler_path) if scaler_path else None)
    
    # Demo ini hanya smoke/sanity check, bukan evaluasi independen benchmark.
    dataset = pd.read_csv('Dataset/Synthetic/synthetic_cr_dataset_v2_geochemical_with_category.csv')
    
    # Take first 20 samples as example
    test_df = dataset.iloc[:20].copy()
    features = sensor.features
    X_test = test_df[features]
    y_actual = test_df['Cr']
    
    # Predict
    y_pred = sensor.predict_batch(X_test)
    
    # Calculate errors
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    
    # Results
    test_df['Cr_Predicted'] = y_pred
    test_df['Error'] = y_actual - y_pred
    test_df['Abs_Error'] = np.abs(test_df['Error'])
    test_df['Pct_Error'] = (test_df['Error'] / y_actual * 100).round(2)
    
    print("\nTest Set Comparison (Actual vs Predicted):")
    display_cols = ['pH', 'EC', 'TDS', 'Cr', 'Cr_Predicted', 'Error', 'Pct_Error']
    print(test_df[display_cols].to_string(index=False))
    
    print(f"\n📊 Demo Metrics (interpretasi terbatas):")
    print(f"  MAE (Mean Absolute Error):     {mae:.4f} μg/L")
    print(f"  RMSE (Root Mean Square Error): {rmse:.4f} μg/L")
    print(f"  Mean % Error:                  {test_df['Pct_Error'].abs().mean():.2f}%")
    print("  Catatan: gunakan file holdout/CV dari training pipeline sebagai metrik utama.")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "🔬 "*35)
    print("SOFT SENSOR INFERENCE EXAMPLES - CHROMIUM (Cr) ESTIMATION")
    print("🔬 "*35 + "\n")
    
    example_single_prediction()
    example_batch_prediction()
    example_timeseries_simulation()
    example_comparison_with_real_data()
    
    print("\n" + "✅ "*35)
    print("ALL EXAMPLES COMPLETED")
    print("✅ "*35)
    print("\nNext Steps:")
    print("  1. Modify the sensor input values untuk sesuai dengan sensor Anda")
    print("  2. Connect real sensor data dan gunakan predict_batch() untuk continuous monitoring")
    print("  3. Save predictions ke database untuk long-term analysis")
    print("  4. Set up alert thresholds jika Cr melebihi batas yang ditentukan")
    print("\n")
