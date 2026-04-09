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
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5
    SMALL_DATASET_THRESHOLD = 500
    
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
        if self.data is None:
            raise RuntimeError("Dataset belum di-load. Panggil load_data() sebelum validate_features().")
        missing = set(features) - set(self.data.columns)
        if missing:
            raise ValueError(f"Fitur tidak ditemukan di dataset: {missing}")
        return True
    
    def prepare_xy(self, features):
        """Pisahkan X (features) dan y (target)"""
        if self.data is None:
            raise RuntimeError("Dataset belum di-load. Panggil load_data() sebelum prepare_xy().")
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

    def get_dataset_info(self):
        if self.dataset_version == 'v2':
            path = Config.PATHS['dataset_v2']
        else:
            path = Config.PATHS['dataset_v1']
        return {
            'dataset_version': self.dataset_version,
            'dataset_path': str(path),
            'n_rows': int(len(self.data)) if self.data is not None else 0,
            'n_columns': int(len(self.data.columns)) if self.data is not None else 0,
        }


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.model_configs = self._build_model_configs()

    def _build_model_configs(self):
        return {
            'lr': {
                'name': 'Linear Regression',
                'estimator': LinearRegression(),
                'needs_scaling': True,
            },
            'rf': {
                'name': 'Random Forest Regressor',
                'estimator': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                'needs_scaling': False,
            },
        }
        
    def train_scenario(self, scenario_name, X, y):
        """Train models untuk satu feature scenario"""
        print(f"\n{'='*70}")
        print(f"Training scenario: {scenario_name.upper()}")
        print(f"Features: {X.columns.tolist()}")
        print(f"Samples: {len(X)}")
        print(f"{'='*70}")
        
        # Holdout split for final independent test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=Config.TEST_SIZE,
            random_state=self.random_state,
        )
        cv_results = self._cross_validate_models(X_train, y_train)

        models_dict, scalers_dict = self._train_models(X_train, y_train)

        holdout_results = self._evaluate_models(
            models_dict,
            scalers_dict,
            X_test,
            y_test,
        )

        results_dict = {}
        for model_name in models_dict:
            results_dict[model_name] = {
                'cv': cv_results[model_name],
                'holdout': holdout_results[model_name],
            }
        
        # Store
        self.models[scenario_name] = models_dict
        self.scalers[scenario_name] = scalers_dict
        self.results[scenario_name] = {
            'train_test_split': (X_train, X_test, y_train, y_test),
            'metrics': results_dict
        }
        
        return models_dict, results_dict

    def _cross_validate_models(self, X_train, y_train):
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2',
        }
        kfold = KFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=self.random_state)
        results = {}

        print("\n  Cross-validation on training split...")
        for model_name, model_config in self.model_configs.items():
            print(f"    - {model_config['name']}")
            if model_config['needs_scaling']:
                estimator = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', clone(model_config['estimator'])),
                ])
            else:
                estimator = clone(model_config['estimator'])

            cv_result = cross_validate(
                estimator,
                X_train,
                y_train,
                cv=kfold,
                scoring=scoring,
                n_jobs=-1,
                return_train_score=False,
            )

            rmse_scores = -cv_result['test_rmse']
            mae_scores = -cv_result['test_mae']
            r2_scores = cv_result['test_r2']

            results[model_name] = {
                'rmse_mean': float(np.mean(rmse_scores)),
                'rmse_std': float(np.std(rmse_scores)),
                'mae_mean': float(np.mean(mae_scores)),
                'mae_std': float(np.std(mae_scores)),
                'r2_mean': float(np.mean(r2_scores)),
                'r2_std': float(np.std(r2_scores)),
                'n_folds': Config.CV_FOLDS,
            }

            print(
                f"      CV RMSE: {results[model_name]['rmse_mean']:.4f} ± {results[model_name]['rmse_std']:.4f}; "
                f"CV MAE: {results[model_name]['mae_mean']:.4f}; "
                f"CV R2: {results[model_name]['r2_mean']:.4f}"
            )

        return results

    def _train_models(self, X_train, y_train):
        """Train Linear Regression & Random Forest on training split"""
        models = {}
        scalers = {}

        print("\n  Training models on training split...")
        for model_name, model_config in self.model_configs.items():
            print(f"    - {model_config['name']}")
            estimator = clone(model_config['estimator'])
            if model_config['needs_scaling']:
                scaler = StandardScaler()
                X_train_model = scaler.fit_transform(X_train)
                scalers[model_name] = scaler
            else:
                X_train_model = X_train
                scalers[model_name] = None

            estimator.fit(X_train_model, y_train)
            models[model_name] = estimator

        return models, scalers

    def _evaluate_models(self, models_dict, scalers_dict, X_test, y_test):
        """Evaluate models on holdout and return metrics"""
        results = {}

        for model_name, model in models_dict.items():
            scaler = scalers_dict[model_name]
            X_test_model = scaler.transform(X_test) if scaler is not None else X_test
            y_pred = model.predict(X_test_model)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred.tolist(),
                'y_test': y_test.values.tolist(),
                'n_samples': int(len(y_test)),
            }

            print(f"\n  {model_name.upper()} Results (holdout test):")
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
                    'CV_RMSE_Mean': metric_dict['cv']['rmse_mean'],
                    'CV_RMSE_Std': metric_dict['cv']['rmse_std'],
                    'CV_MAE_Mean': metric_dict['cv']['mae_mean'],
                    'CV_MAE_Std': metric_dict['cv']['mae_std'],
                    'CV_R2_Mean': metric_dict['cv']['r2_mean'],
                    'CV_R2_Std': metric_dict['cv']['r2_std'],
                    'Holdout_RMSE': metric_dict['holdout']['rmse'],
                    'Holdout_MAE': metric_dict['holdout']['mae'],
                    'Holdout_R2': metric_dict['holdout']['r2'],
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Summary table
        print("\nSummary Table:")
        print(comparison_df.to_string(index=False))
        
        # Select best model by CV RMSE (lower is better)
        best_idx = comparison_df['CV_RMSE_Mean'].idxmin()
        best_model_info = comparison_df.loc[best_idx]
        
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best_model_info['Model'].upper()} - {best_model_info['Scenario'].upper()}")
        print(f"{'='*70}")
        print(f"  CV RMSE: {best_model_info['CV_RMSE_Mean']:.4f} ± {best_model_info['CV_RMSE_Std']:.4f} μg/L")
        print(f"  CV MAE:  {best_model_info['CV_MAE_Mean']:.4f} ± {best_model_info['CV_MAE_Std']:.4f} μg/L")
        print(f"  CV R²:   {best_model_info['CV_R2_Mean']:.4f} ± {best_model_info['CV_R2_Std']:.4f}")
        print(f"  Holdout RMSE: {best_model_info['Holdout_RMSE']:.4f} μg/L")
        print(f"  Holdout MAE:  {best_model_info['Holdout_MAE']:.4f} μg/L")
        print(f"  Holdout R²:   {best_model_info['Holdout_R2']:.4f}")
        
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
            y_test = np.array(metrics['lr']['holdout']['y_test'])
            y_pred = np.array(metrics['lr']['holdout']['y_pred'])
            ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax.set_xlabel('Actual Cr (μg/L)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Cr (μg/L)', fontsize=10, fontweight='bold')
            r2_val = metrics['lr']['holdout']['r2']
            ax.set_title(f"{scenario.upper()} - Linear Regression (Holdout R²={r2_val:.4f})", 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Random Forest
            ax = axes[row_idx, 1]
            y_test = np.array(metrics['rf']['holdout']['y_test'])
            y_pred = np.array(metrics['rf']['holdout']['y_pred'])
            ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='darkorange', edgecolors='black', linewidth=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
            ax.set_xlabel('Actual Cr (μg/L)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Predicted Cr (μg/L)', fontsize=10, fontweight='bold')
            r2_val = metrics['rf']['holdout']['r2']
            ax.set_title(f"{scenario.upper()} - Random Forest (Holdout R²={r2_val:.4f})", 
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
        fig.suptitle("Holdout Metrics Comparison", fontsize=14, fontweight='bold')
        
        metrics_names = ['Holdout_RMSE', 'Holdout_MAE', 'Holdout_R2']
        for col_idx, metric in enumerate(metrics_names):
            ax = axes[col_idx]
            
            pivot_data = comparison_df.pivot(index='Scenario', columns='Model', values=metric)
            pivot_data.plot(kind='bar', ax=ax, color=['steelblue', 'darkorange'], alpha=0.7)
            
            ax.set_xlabel('Scenario', fontsize=10, fontweight='bold')
            metric_label = metric.replace('Holdout_', '')
            ax.set_ylabel(metric_label, fontsize=10, fontweight='bold')
            ax.set_title(f"{metric_label} by Scenario & Model", fontsize=11, fontweight='bold')
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
        sample_split = self.trainer.results[best_scenario]['train_test_split']
        total_samples = len(sample_split[0]) + len(sample_split[1])
        holdout_samples = len(sample_split[1])
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
Cross-Validation Folds: {Config.CV_FOLDS}

Feature Scenarios:
{json.dumps(Config.FEATURE_SCENARIOS, indent=2)}

Models:
  - Linear Regression (Baseline)
  - Random Forest Regressor (100 trees, max_depth=20)

Leakage Control:
    - Holdout split dibuat sekali di awal per-scenario.
    - Benchmark CV dijalankan hanya pada training split.
    - Holdout test tidak ikut proses model selection.

Small Dataset Note:
    - Total sample pada dataset v2 saat ini relatif kecil (n={total_samples}; holdout={holdout_samples}).
    - Interpretasi metrik tetap perlu mempertimbangkan variansi antar fold.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. BENCHMARK PERFORMANCE (CV ON TRAINING SPLIT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{comparison_df[['Scenario', 'Model', 'CV_RMSE_Mean', 'CV_RMSE_Std', 'CV_MAE_Mean', 'CV_MAE_Std', 'CV_R2_Mean', 'CV_R2_Std']].to_string(index=False)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. FINAL HOLDOUT PERFORMANCE (UNSEEN TEST SPLIT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{comparison_df[['Scenario', 'Model', 'Holdout_RMSE', 'Holdout_MAE', 'Holdout_R2']].to_string(index=False)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. BEST MODEL (SELECTED FOR PRODUCTION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model Name: {best_info['Model'].upper()}
Scenario: {best_info['Scenario'].upper()}

Performance Metrics:
    CV Benchmark:
        - RMSE: {best_info['CV_RMSE_Mean']:.6f} ± {best_info['CV_RMSE_Std']:.6f} μg/L
        - MAE:  {best_info['CV_MAE_Mean']:.6f} ± {best_info['CV_MAE_Std']:.6f} μg/L
        - R²:   {best_info['CV_R2_Mean']:.6f} ± {best_info['CV_R2_Std']:.6f}
    Holdout Test:
        - RMSE: {best_info['Holdout_RMSE']:.6f} μg/L
        - MAE:  {best_info['Holdout_MAE']:.6f} μg/L
        - R²:   {best_info['Holdout_R2']:.6f}

Features Used:
  {', '.join(Config.FEATURE_SCENARIOS[best_info['Scenario']])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. MODEL ARTIFACTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Saved Models:
  - models/best_model_{best_model}_{best_scenario}.pkl
    - models/best_model_scaler_{best_model}_{best_scenario}.pkl
    - models/best_model_{best_model}_{best_scenario}.metadata.json
    - models/model_*.pkl (all model/scenario combinations)
    - models/model_*.metadata.json

Results:
  - results/ml_results_report.txt (this file)
    - results/model_comparison.csv (combined)
    - results/model_benchmark_cv.csv
    - results/model_holdout_metrics.csv
  - results/plots/predictions_comparison.png
  - results/plots/metrics_comparison.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. INFERENCE USAGE
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
    def save_models(trainer, best_scenario, best_model_name, dataset_info):
        """Save all model artifacts with per-model metadata and backward-compatible best aliases"""
        model_dir = Config.PATHS['models']
        model_dir.mkdir(exist_ok=True, parents=True)

        artifacts = []
        for scenario_name, model_dict in trainer.models.items():
            for model_name, model in model_dict.items():
                stem = f'model_{model_name}_{scenario_name}'
                model_path = model_dir / f'{stem}.pkl'
                scaler_path = model_dir / f'{stem}.scaler.pkl'
                metadata_path = model_dir / f'{stem}.metadata.json'

                joblib.dump(model, model_path)
                scaler = trainer.scalers[scenario_name][model_name]
                if scaler is not None:
                    joblib.dump(scaler, scaler_path)

                metadata = {
                    'artifact_stem': stem,
                    'model_name': model_name,
                    'model_label': trainer.model_configs[model_name]['name'],
                    'scenario': scenario_name,
                    'features': Config.FEATURE_SCENARIOS[scenario_name],
                    'dataset': dataset_info,
                    'training_timestamp': datetime.now().isoformat(),
                    'random_state': trainer.random_state,
                    'test_size': Config.TEST_SIZE,
                    'cv_folds': Config.CV_FOLDS,
                    'hyperparameters': model.get_params(),
                    'evaluation': trainer.results[scenario_name]['metrics'][model_name],
                    'scaler_path': str(scaler_path.name) if scaler is not None else None,
                }

                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, default=str)

                print(f"✓ Saved model artifact: {model_path}")
                print(f"✓ Saved metadata: {metadata_path}")
                artifacts.append((model_path, scaler_path if scaler is not None else None, metadata_path))

        best_stem = f'model_{best_model_name}_{best_scenario}'
        best_model_path = model_dir / f'{best_stem}.pkl'
        best_scaler_path = model_dir / f'{best_stem}.scaler.pkl'
        best_metadata_path = model_dir / f'{best_stem}.metadata.json'

        # Backward-compatible aliases for existing scripts
        best_model_alias = model_dir / f'best_model_{best_model_name}_{best_scenario}.pkl'
        best_scaler_alias_new = model_dir / f'best_model_scaler_{best_model_name}_{best_scenario}.pkl'
        best_metadata_alias = model_dir / f'best_model_{best_model_name}_{best_scenario}.metadata.json'
        legacy_scaler_alias = model_dir / f'best_model_scaler_{best_scenario}.pkl'
        legacy_metadata_alias = model_dir / 'best_model_metadata.json'

        best_model = joblib.load(best_model_path)
        joblib.dump(best_model, best_model_alias)
        if best_scaler_path.exists():
            best_scaler = joblib.load(best_scaler_path)
            joblib.dump(best_scaler, best_scaler_alias_new)
            joblib.dump(best_scaler, legacy_scaler_alias)
        with open(best_metadata_path, 'r', encoding='utf-8') as f:
            best_metadata = json.load(f)
        with open(best_metadata_alias, 'w', encoding='utf-8') as f:
            json.dump(best_metadata, f, indent=2, default=str)
        with open(legacy_metadata_alias, 'w', encoding='utf-8') as f:
            json.dump(best_metadata, f, indent=2, default=str)

        print(f"✓ Saved best model alias: {best_model_alias}")
        print(f"✓ Saved best scaler alias: {best_scaler_alias_new}")
        print(f"✓ Updated legacy metadata alias: {legacy_metadata_alias}")

        return artifacts


class SoftSensorInference:
    def __init__(self, model_path, scaler_path=None):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path)

        inferred_metadata_path = self.model_path.with_suffix('.metadata.json')
        legacy_metadata_path = self.model_path.parent / 'best_model_metadata.json'

        if inferred_metadata_path.exists():
            metadata_path = inferred_metadata_path
        elif legacy_metadata_path.exists():
            metadata_path = legacy_metadata_path
        else:
            raise FileNotFoundError(
                f"Metadata not found for model {self.model_path}. "
                f"Expected {inferred_metadata_path} or {legacy_metadata_path}."
            )

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        else:
            scaler_from_meta = self.metadata.get('scaler_path')
            if scaler_from_meta:
                auto_scaler_path = self.model_path.parent / scaler_from_meta
                self.scaler = joblib.load(auto_scaler_path) if auto_scaler_path.exists() else None
            else:
                self.scaler = None

        self.features = self.metadata['features']
        print(f"✓ Model loaded: {self.metadata.get('model_name', self.metadata.get('model_type', 'unknown'))}")
        print(f"  Scenario: {self.metadata['scenario']}")
        print(f"  Features: {', '.join(self.features)}")

    def _prepare_features(self, df):
        missing = [col for col in self.features if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required feature columns for inference: {missing}. "
                f"Required features: {self.features}"
            )
        return df[self.features].copy()
    
    def predict_single(self, sensor_dict):
        """Predict single sample from sensor dict"""
        X = self._prepare_features(pd.DataFrame([sensor_dict]))
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        cr_pred = self.model.predict(X)[0]
        return cr_pred
    
    def predict_batch(self, sensor_df):
        """Predict batch dari DataFrame"""
        X = self._prepare_features(sensor_df)
        
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
    if len(data) < Config.SMALL_DATASET_THRESHOLD:
        print(
            f"  [Limitasi] Dataset relatif kecil (n={len(data)}). "
            "Variansi CV bisa tinggi; interpretasi metrik harus hati-hati."
        )
    
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
    
    # 5. Save model artifacts
    print("\n[5/5] Saving best model...")
    best_scenario = best_info['Scenario']
    best_model = best_info['Model']
    dataset_info = processor.get_dataset_info()
    ModelManager.save_models(trainer, best_scenario, best_model, dataset_info)
    
    # Save comparison results
    results_dir = Config.PATHS['results']
    results_dir.mkdir(exist_ok=True, parents=True)
    comparison_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    comparison_df[['Scenario', 'Model', 'CV_RMSE_Mean', 'CV_RMSE_Std', 'CV_MAE_Mean', 'CV_MAE_Std', 'CV_R2_Mean', 'CV_R2_Std']].to_csv(
        results_dir / 'model_benchmark_cv.csv',
        index=False,
    )
    comparison_df[['Scenario', 'Model', 'Holdout_RMSE', 'Holdout_MAE', 'Holdout_R2']].to_csv(
        results_dir / 'model_holdout_metrics.csv',
        index=False,
    )
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
