#!/usr/bin/env python3
"""
Machine Learning Pipeline for Soft Sensor: Chromium (Cr) Estimation
Soft sensor yang menerima input sensor kualitas air (EC, pH, TDS, etc.)
dan menghasilkan prediksi konsentrasi Chromium
"""

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    
    # Feature scenarios
    FEATURE_SCENARIOS = {
        'full': [
            'EC', 'TDS', 'pH', 
            'Suhu Air (°C)', 'Suhu Lingkungan (°C)',
            'Kelembapan Lingkungan (%)', 'Tegangan (V)'
        ],
        'core': ['EC', 'TDS', 'pH'],
        'minimal': ['EC', 'pH']
    }
    
    TARGET = 'Cr'
    
    # Paths
    PATHS = {
        'dataset_v2': Path('Dataset/Synthetic/synthetic_cr_dataset_v2_geochemical_with_category.csv'),
        'dataset_v1': Path('Dataset/Synthetic/synthetic_cr_dataset_with_category.csv'),
        'models': Path('models'),
        'results': Path('results'),
        'plots': Path('results/plots'),
    }


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

class DataProcessor:
    def __init__(self, dataset_version='v2'):
        self.dataset_version = dataset_version
        self.scaler = StandardScaler()
        self.data = None
        
    def load_data(self):
        """Load dataset berdasarkan versi (v1 atau v2)"""
        if self.dataset_version == 'v2':
            path = Config.PATHS['dataset_v2']
        else:
            path = Config.PATHS['dataset_v1']
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
        
        self.data = pd.read_csv(path)
        print(f"✓ Dataset {self.dataset_version} loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
        return self.data
    
    def validate_features(self, features):
        """Validasi semua fitur tersedia di dataset"""
        missing = set(features) - set(self.data.columns)
        if missing:
            raise ValueError(f"Fitur tidak ditemukan di dataset: {missing}")
        return True
    
    def prepare_xy(self, features):
        """Pisahkan X (features) dan y (target)"""
        self.validate_features(features + [Config.TARGET])
        
        X = self.data[features].copy()
        y = self.data[Config.TARGET].copy()
        
        # Check missing values
        if X.isnull().any().any() or y.isnull().any():
            print(f"  ⚠ Dropping {X.isnull().sum().sum()} rows with missing values")
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
        
        return X, y


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def train_scenario(self, scenario_name, X, y):
        """Train models untuk satu feature scenario"""
        print(f"\n{'='*70}")
        print(f"Training scenario: {scenario_name.upper()}")
        print(f"Features: {X.columns.tolist()}")
        print(f"Samples: {len(X)}")
        print(f"{'='*70}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=self.random_state
        )
        
        # Standardisasi fitur
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame untuk kemudahan
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
        
        # Train models
        models_dict = self._train_models(X_train_scaled, y_train, scenario_name)
        
        # Evaluate
        results_dict = self._evaluate_models(
            models_dict, X_test_scaled, y_test, scenario_name
        )
        
        # Store
        self.models[scenario_name] = models_dict
        self.scalers[scenario_name] = scaler
        self.results[scenario_name] = {
            'train_test_split': (X_train, X_test, y_train, y_test),
            'scaled_data': (X_train_scaled, X_test_scaled),
            'metrics': results_dict
        }
        
        return models_dict, results_dict
    
    def _train_models(self, X_train, y_train, scenario):
        """Train Linear Regression & Random Forest"""
        models = {}
        
        # 1. Linear Regression (Baseline)
        print("\n  Training: Linear Regression (baseline)...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        models['lr'] = lr
        print(f"    ✓ Linear Regression trained")
        
        # 2. Random Forest (Main model)
        print("  Training: Random Forest Regressor...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        models['rf'] = rf
        print(f"    ✓ Random Forest trained (100 trees)")
        
        return models
    
    def _evaluate_models(self, models_dict, X_test, y_test, scenario):
        """Evaluate models & return metrics"""
        results = {}
        
        for model_name, model in models_dict.items():
            y_pred = model.predict(X_test)
            
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred,
                'y_test': y_test.values
            }
            
            print(f"\n  {model_name.upper()} Results (test set):")
            print(f"    RMSE: {rmse:.4f} μg/L")
            print(f"    MAE:  {mae:.4f} μg/L")
            print(f"    R²:   {r2:.4f}")
        
        return results


# ============================================================================
# RESULTS ANALYSIS & VISUALIZATION
# ============================================================================

class ResultsAnalyzer:
    def __init__(self, trainer, plots_dir):
        self.trainer = trainer
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
    def compare_scenarios(self):
        """Compare performa antar scenarios"""
        print(f"\n{'='*70}")
        print("PERBANDINGAN PERFORMA ANTAR SCENARIOS")
        print(f"{'='*70}")
        
        comparison_data = []
        for scenario, results in self.trainer.results.items():
            metrics = results['metrics']
            for model_name, metric_dict in metrics.items():
                comparison_data.append({
                    'Scenario': scenario,
                    'Model': model_name,
                    'RMSE': metric_dict['rmse'],
                    'MAE': metric_dict['mae'],
                    'R2': metric_dict['r2']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Summary table
        print("\nSummary Table:")
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_idx = comparison_df['R2'].idxmax()
        best_model_info = comparison_df.loc[best_idx]
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model_info['Model'].upper()} - {best_model_info['Scenario'].upper()}")
        print(f"{'='*70}")
        print(f"  RMSE: {best_model_info['RMSE']:.4f} μg/L")
        print(f"  MAE:  {best_model_info['MAE']:.4f} μg/L")
        print(f"  R²:   {best_model_info['R2']:.4f}")
        
        return comparison_df, best_model_info
    
    def plot_predictions(self):
        """Plot actual vs predicted untuk semua scenarios"""
        fig, axes = plt.subplots(
            len(self.trainer.results), 2,
            figsize=(14, 4 * len(self.trainer.results))
        )
        
        if len(self.trainer.results) == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, (scenario, results) in enumerate(self.trainer.results.items()):
            metrics = results['metrics']
            
            # Linear Regression
            ax = axes[row_idx, 0]
            y_test = metrics['lr']['y_test']
            y_pred = metrics['lr']['y_pred']
            ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax.set_xlabel('Actual Cr (μg/L)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Cr (μg/L)', fontsize=10, fontweight='bold')
            r2_val = metrics['lr']['r2']
            ax.set_title(f"{scenario.upper()} - Linear Regression (R²={r2_val:.4f})", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Random Forest
            ax = axes[row_idx, 1]
            y_test = metrics['rf']['y_test']
            y_pred = metrics['rf']['y_pred']
            ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='darkorange', edgecolors='black', linewidth=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax.set_xlabel('Actual Cr (μg/L)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Cr (μg/L)', fontsize=10, fontweight='bold')
            r2_val = metrics['rf']['r2']
            ax.set_title(f"{scenario.upper()} - Random Forest (R²={r2_val:.4f})", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.plots_dir / 'predictions_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: results/plots/predictions_comparison.png")
        plt.close()
    
    def plot_metrics_comparison(self, comparison_df):
        """Bar plot perbandingan metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Model Performance Metrics Comparison", fontsize=14, fontweight='bold')
        
        metrics_names = ['RMSE', 'MAE', 'R2']
        for col_idx, metric in enumerate(metrics_names):
            ax = axes[col_idx]
            
            pivot_data = comparison_df.pivot(index='Scenario', columns='Model', values=metric)
            pivot_data.plot(kind='bar', ax=ax, color=['steelblue', 'darkorange'], alpha=0.7)
            
            ax.set_xlabel('Scenario', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric, fontsize=10, fontweight='bold')
            ax.set_title(f"{metric} by Scenario & Model", fontsize=11, fontweight='bold')
            ax.legend(title='Model')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        filename = self.plots_dir / 'metrics_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: results/plots/metrics_comparison.png")
        plt.close()
    
    def save_results_report(self, comparison_df, best_info):
        """Save detailed results report"""
        best_scenario = best_info['Scenario']
        best_model = best_info['Model']
        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║          ML PIPELINE RESULTS: SOFT SENSOR FOR CHROMIUM ESTIMATION      ║
╚════════════════════════════════════════════════════════════════════════╝

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Test Size: {Config.TEST_SIZE * 100}%
Train Size: {(1 - Config.TEST_SIZE) * 100}%
Random State: {Config.RANDOM_STATE}

Feature Scenarios:
{json.dumps(Config.FEATURE_SCENARIOS, indent=2)}

Models:
  - Linear Regression (Baseline)
  - Random Forest Regressor (100 trees, max_depth=20)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. PERFORMANCE COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{comparison_df.to_string(index=False)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. BEST MODEL (SELECTED FOR PRODUCTION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Name: {best_info['Model'].upper()}
Scenario: {best_info['Scenario'].upper()}

Performance Metrics:
  - RMSE: {best_info['RMSE']:.6f} μg/L
  - MAE:  {best_info['MAE']:.6f} μg/L
  - R²:   {best_info['R2']:.6f}

Features Used:
  {', '.join(Config.FEATURE_SCENARIOS[best_info['Scenario']])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. MODEL ARTIFACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Saved Models:
  - models/best_model_{best_model}_{best_scenario}.pkl
  - models/best_model_scaler_{best_scenario}.pkl
  - models/best_model_metadata.json

Results:
  - results/ml_results_report.txt (this file)
  - results/model_comparison.csv
  - results/plots/predictions_comparison.png
  - results/plots/metrics_comparison.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. INFERENCE USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Use the SoftSensorInference class to make predictions:

    from ml_pipeline_cr_soft_sensor import SoftSensorInference
    
    sensor = SoftSensorInference('models/best_model_{best_model}_{best_scenario}.pkl', 
                                  'models/best_model_scaler_{best_scenario}.pkl')
    
    # Single prediction
    cr_pred = sensor.predict_single({{
        'EC': 1200,
        'TDS': 850,
        'pH': 6.8,
        'Suhu Air (°C)': 26.5,
        'Suhu Lingkungan (°C)': 28.3,
        'Kelembapan Lingkungan (%)': 72.1,
        'Tegangan (V)': 3.95
    }})
    print(f"Predicted Cr: {{cr_pred:.2f}} μg/L")
    
    # Batch predictions
    import pandas as pd
    sensor_data = pd.read_csv('sensor_data.csv')
    cr_predictions = sensor.predict_batch(sensor_data)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        filepath = Config.PATHS['results'] / 'ml_results_report.txt'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✓ Saved: results/ml_results_report.txt")


# ============================================================================
# MODEL SAVING & INFERENCE
# ============================================================================

class ModelManager:
    @staticmethod
    def save_best_model(trainer, best_scenario, best_model_name):
        """Simpan model terbaik & scaler"""
        best_model = trainer.models[best_scenario][best_model_name]
        best_scaler = trainer.scalers[best_scenario]
        
        model_dir = Config.PATHS['models']
        model_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_path = model_dir / f'best_model_{best_model_name}_{best_scenario}.pkl'
        joblib.dump(best_model, model_path)
        print(f"✓ Saved model: {model_path}")
        
        # Save scaler
        scaler_path = model_dir / f'best_model_scaler_{best_scenario}.pkl'
        joblib.dump(best_scaler, scaler_path)
        print(f"✓ Saved scaler: {scaler_path}")
        
        # Save metadata
        metadata = {
            'model_type': best_model_name,
            'scenario': best_scenario,
            'features': Config.FEATURE_SCENARIOS[best_scenario],
            'timestamp': datetime.now().isoformat(),
            'metrics': trainer.results[best_scenario]['metrics'][best_model_name]
        }
        
        metadata_path = model_dir / f'best_model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✓ Saved metadata: {metadata_path}")
        
        return model_path, scaler_path, metadata_path


class SoftSensorInference:
    def __init__(self, model_path, scaler_path=None):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        
        # Load metadata
        metadata_path = Path(model_path).parent / 'best_model_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.features = self.metadata['features']
        print(f"✓ Model loaded: {self.metadata['model_type']}")
        print(f"  Scenario: {self.metadata['scenario']}")
        print(f"  Features: {', '.join(self.features)}")
    
    def predict_single(self, sensor_dict):
        """Predict single sample from sensor dict"""
        X = pd.DataFrame([sensor_dict])[self.features]
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        cr_pred = self.model.predict(X)[0]
        return cr_pred
    
    def predict_batch(self, sensor_df):
        """Predict batch dari DataFrame"""
        X = sensor_df[self.features].copy()
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        cr_predictions = self.model.predict(X)
        return cr_predictions


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ML PIPELINE: SOFT SENSOR FOR CHROMIUM (Cr) ESTIMATION")
    print("="*70)
    
    # 1. Load & prepare data
    print("\n[1/5] Loading and preparing data...")
    processor = DataProcessor(dataset_version='v2')
    data = processor.load_data()
    
    # 2. Train models for all scenarios
    print("\n[2/5] Training models for all scenarios...")
    trainer = ModelTrainer(random_state=Config.RANDOM_STATE)
    
    for scenario_name, features in Config.FEATURE_SCENARIOS.items():
        X, y = processor.prepare_xy(features)
        trainer.train_scenario(scenario_name, X, y)
    
    # 3. Compare & analyze results
    print("\n[3/5] Analyzing and comparing results...")
    analyzer = ResultsAnalyzer(trainer, Config.PATHS['plots'])
    comparison_df, best_info = analyzer.compare_scenarios()
    
    # 4. Visualize results
    print("\n[4/5] Creating visualizations...")
    analyzer.plot_predictions()
    analyzer.plot_metrics_comparison(comparison_df)
    analyzer.save_results_report(comparison_df, best_info)
    
    # 5. Save best model
    print("\n[5/5] Saving best model...")
    best_scenario = best_info['Scenario']
    best_model = best_info['Model']
    ModelManager.save_best_model(trainer, best_scenario, best_model)
    
    # Save comparison results
    results_dir = Config.PATHS['results']
    results_dir.mkdir(exist_ok=True, parents=True)
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    print(f"✓ Saved comparison: results/model_comparison.csv")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nGenerated artifacts:")
    print(f"  📁 models/     - Trained models & scalers")
    print(f"  📁 results/    - Results & evaluation reports")
    print(f"  📊 images generated to: results/plots/")
    print(f"\nNext step: Use SoftSensorInference class for inference with real sensor data")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
