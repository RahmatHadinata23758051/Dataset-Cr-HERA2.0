import pandas as pd
import numpy as np
import os
import pickle
from scipy.stats import spearmanr

def load_model_pack(pack_path):
    with open(pack_path, "rb") as f:
        pack = pickle.load(f)
    return pack

def run_scenario_test(model_pack):
    """
    Scenario Test: Checks if predictions increase monotonically as we transition from
    Clean -> Moderately Polluted -> Highly Polluted -> Extreme.
    """
    model = model_pack["model"]
    scaler = model_pack["scaler"]
    metal = model_pack["metal"]
    
    # 1. Define the 4 graded geochemical scenarios
    scenarios = pd.DataFrame([
        {"pH": 7.5, "EC_uScm": 150.0,  "TDS_mgL": 96.0,   "Suhu_Air": 24.5}, # Stage 1: Clean
        {"pH": 6.8, "EC_uScm": 500.0,  "TDS_mgL": 320.0,  "Suhu_Air": 24.5}, # Stage 2: Moderate
        {"pH": 6.0, "EC_uScm": 1200.0, "TDS_mgL": 768.0,  "Suhu_Air": 24.5}, # Stage 3: High
        {"pH": 5.2, "EC_uScm": 2200.0, "TDS_mgL": 1408.0, "Suhu_Air": 24.5}  # Stage 4: Extreme
    ])
    
    # Scale inputs
    X_scaled = scaler.transform(scenarios.values)
    
    # Predict
    predictions = model.predict(X_scaled)
    
    # Check monotonicity: Pred[0] < Pred[1] < Pred[2] < Pred[3]
    passed = True
    for i in range(3):
        if predictions[i] >= predictions[i+1]:
            passed = False
            
    status = "PASS" if passed else "FAIL"
    
    result_str = f"\n--- Scenario Monotonicity Test for {metal} ---\n"
    stages = ["Clean", "Moderately Polluted", "Highly Polluted", "Extreme Runoff"]
    for i, stage in enumerate(stages):
        result_str += f"  * Stage {i+1} ({stage}): {predictions[i]:.2f} ug/L\n"
    result_str += f"Status: {status}\n"
    
    return status, result_str

def run_sensitivity_test(model_pack):
    """
    Sensitivity Test: Computes Spearman rank correlations between perturbed features
    and output predictions to verify geochemical trends.
    Expected: pH correlation < 0 (negative), EC correlation > 0 (positive).
    """
    model = model_pack["model"]
    scaler = model_pack["scaler"]
    metal = model_pack["metal"]
    
    # 1. pH Sensitivity (varying pH, keeping others constant)
    n_steps = 100
    ph_range = np.linspace(5.0, 8.5, n_steps)
    ph_test = pd.DataFrame({
        "pH": ph_range,
        "EC_uScm": [700.0] * n_steps,
        "TDS_mgL": [448.0] * n_steps,
        "Suhu_Air": [24.5] * n_steps
    })
    X_ph = scaler.transform(ph_test.values)
    preds_ph = model.predict(X_ph)
    
    r_ph, _ = spearmanr(ph_range, preds_ph)
    ph_pass = r_ph < 0
    ph_status = "PASS" if ph_pass else "FAIL"
    
    # 2. EC Sensitivity (varying EC and TDS accordingly, keeping pH & Temp constant)
    ec_range = np.linspace(300.0, 2000.0, n_steps)
    ec_test = pd.DataFrame({
        "pH": [6.5] * n_steps,
        "EC_uScm": ec_range,
        "TDS_mgL": ec_range * 0.64,
        "Suhu_Air": [24.5] * n_steps
    })
    X_ec = scaler.transform(ec_test.values)
    preds_ec = model.predict(X_ec)
    
    r_ec, _ = spearmanr(ec_range, preds_ec)
    ec_pass = r_ec > 0
    ec_status = "PASS" if ec_pass else "FAIL"
    
    result_str = f"\n--- Geochemical Sensitivity (Spearman Rank) for {metal} ---\n"
    result_str += f"  * pH Sensitivity Correlation: {r_ph:.4f} (Expected: < 0) -> {ph_status}\n"
    result_str += f"  * EC Sensitivity Correlation: {r_ec:.4f} (Expected: > 0) -> {ec_status}\n"
    
    overall_status = "PASS" if (ph_pass and ec_pass) else "FAIL"
    result_str += f"Status: {overall_status}\n"
    
    return overall_status, result_str

def main():
    print("[INFO] Starting HERA 2.0 Physical Behavior Validation Pipeline...")
    
    # Resolve relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "phase2_training", "models"))
    results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    
    metals = ["chromium", "nickel"]
    report_content = "HERA 2.0 PHYSICAL BEHAVIOR VALIDATION REPORT\n"
    report_content += "=======================================================\n"
    
    all_passed = True
    
    for metal in metals:
        model_path = os.path.join(models_dir, f"best_model_{metal}.pkl")
        if not os.path.exists(model_path):
            print(f"[ERROR] Model pack not found for {metal} at: {model_path}")
            continue
            
        print(f"\n[VALIDATING] Loading best model pack for: {metal.upper()}...")
        pack = load_model_pack(model_path)
        
        # Run tests
        scen_status, scen_log = run_scenario_test(pack)
        sens_status, sens_log = run_sensitivity_test(pack)
        
        print(scen_log)
        print(sens_log)
        
        report_content += f"\nMETAL PARAMETER: {metal.upper()}\n"
        report_content += "-------------------------------------------------------\n"
        report_content += scen_log
        report_content += sens_log
        
        if scen_status == "FAIL" or sens_status == "FAIL":
            all_passed = False
            
    report_content += "\n=======================================================\n"
    overall_system_status = "SYSTEM PASSED" if all_passed else "SYSTEM FAILED CHECK(S)"
    report_content += f"OVERALL PHYSICAL COMPLIANCE STATUS: {overall_system_status}\n"
    
    report_path = os.path.join(results_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print(f"\n[OK] Validation pipeline complete! Detailed report saved to: {report_path}")
    print(f"[STATUS] Overall Geochemical Behavior Compliance: {overall_system_status}")

if __name__ == "__main__":
    main()
