
import pandas as pd
import numpy as np
import os
import sys

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'darts')))
from ispu_calculator import calculate_ispu, get_ispu_category, BREAKPOINTS

def analyze_discrepancy():
    print("="*60)
    print("ANALYSIS: Ground Truth vs Predictions vs Regulation")
    print("="*60)
    
    # 1. Load Ground Truth Data
    gt_path = 'dataset/data-indeks-standar-pencemar-udara-(ispu)-di-provinsi-dki-jakarta-komponen-data.csv'
    try:
        df_gt = pd.read_csv(gt_path)
        print(f"\n1. Ground Truth Data Loaded ({len(df_gt)} rows)")
        print(df_gt.head(3).T)
    except Exception as e:
        print(f"Error loading GT: {e}")
        return

    # Clean GT cols
    # It seems the GT file has 'pm25', 'pm10', 'so2', 'co', 'o3', 'no2', 'max', 'critical', 'categori'
    # We need to check if 'categori' matches our 'calculate_ispu' result for the GIVEN concentrations.
    
    # Rename cols to standard format
    col_map = {
        'pm_sepuluh': 'pm10',
        'pm_duakomalima': 'pm25',
        'sulfur_dioksida': 'so2',
        'karbon_monoksida': 'co',
        'ozon': 'o3',
        'nitrogen_dioksida': 'no2',
        'kategori': 'categori', # Standardize if mixed
        'parameter_pencemar_kritis': 'critical_parameter'
    }
    df_gt = df_gt.rename(columns=col_map)
    
    pollutants = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    
    # Filter for valid numeric rows
    # Note: 'pm10' might not exist if column rename failed or data is different, but based on output it works.
    for pol in pollutants:
        if pol in df_gt.columns:
            df_gt[pol] = pd.to_numeric(df_gt[pol], errors='coerce')
    
    # Drop rows where ALL pollutants are missing
    df_gt = df_gt.dropna(subset=[p for p in pollutants if p in df_gt.columns], how='all')
    
    print(f"\n2. Checking Consistency: Does GT Concentrations -> GT Category match Permen LHK 14/2020?")
    
    mismatch_count = 0
    total_checked = 0
    
    sample_size = min(1000, len(df_gt))
    df_sample = df_gt.sample(sample_size, random_state=42)
    
    print(f"Checking {len(df_sample)} random samples...")
    
    for idx, row in df_sample.iterrows():
        # Calculate ISPU for this row using OUR calculator
        row_ispus = {}
        for pol in pollutants:
            val = row[pol]
            if val >= 0:
                row_ispus[pol] = calculate_ispu(pol, val)
        
        if not row_ispus:
            continue
            
        my_max_ispu = max(row_ispus.values())
        my_category = get_ispu_category(my_max_ispu)
        
        gt_category = str(row['categori']).upper()
        
        # Simple mapping for comparison
        if gt_category == 'TIDAK SEHAT': gt_cat_simple = 'TIDAK SEHAT'
        elif gt_category == 'SANGAT TIDAK SEHAT': gt_cat_simple = 'SANGAT TIDAK SEHAT' # Or map to TS
        elif gt_category == 'SEDANG': gt_cat_simple = 'SEDANG'
        elif gt_category == 'BAIK': gt_cat_simple = 'BAIK'
        else: gt_cat_simple = gt_category
        
        if my_category != gt_cat_simple:
            mismatch_count += 1
            if mismatch_count <= 5: # Print first 5 mismatches
                print(f"  [MISMATCH] Row {idx}")
                print(f"    Concentrations: {row[pollutants].to_dict()}")
                print(f"    GT Label: {gt_category} (ISPU {row['max']})")
                print(f"    My Calc:  {my_category} (ISPU {my_max_ispu:.2f})")
                print(f"    Breakdown: {row_ispus}")
                
    print(f"\n  >> Mismatch Rate: {mismatch_count}/{len(df_sample)} ({mismatch_count/len(df_sample)*100:.1f}%)")
    
    if mismatch_count / len(df_sample) > 0.5:
        print("\nCONCLUSION: The Ground Truth likely uses a DIFFERENT regulation or formula.")
    else:
        print("\nCONCLUSION: The Regulation matches. The issue is likely the FORECAST values are too high.")

    # 3. Analyze Forecasted Values (from predictions_pollutants.csv if exists)
    pred_path = 'darts/predictions_pollutants.csv'
    if os.path.exists(pred_path):
        print(f"\n3. Analyzing Forecast Values ({pred_path})")
        df_pred = pd.read_csv(pred_path)
        print(df_pred[pollutants].describe())
        
        print("\n  Comparison of PM2.5:")
        print(f"    Ground Truth PM2.5 Mean: {df_gt['pm25'].mean():.2f}")
        print(f"    Forecast PM2.5 Mean:     {df_pred['pm25'].mean():.2f}")
    else:
        print("\n3. No prediction file found to analyze.")

if __name__ == "__main__":
    analyze_discrepancy()
