
import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ispu_calculator import calculate_ispu_for_dataframe, map_to_3_categories

def main():
    print("="*60)
    print("Regenerating Submission with Tuned ISPU Calculator")
    print("="*60)
    
    # 1. Load existing predictions (Raw values)
    pred_path = 'predictions_pollutants.csv'
    if not os.path.exists(pred_path):
        # Try finding it in darts folder if running from root
        pred_path = 'darts/predictions_pollutants.csv'
    
    if not os.path.exists(pred_path):
        print("Error: 'predictions_pollutants.csv' not found. Cannot regenerate.")
        return

    print(f"Loading raw predictions from: {pred_path}")
    df_preds = pd.read_csv(pred_path)
    
    # 2. Recalculate ISPU
    print("Recalculating ISPU with new breakpoints...")
    pollutants = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    
    # Apply calculation
    df_result = calculate_ispu_for_dataframe(df_preds, pollutant_cols=pollutants)
    
    # Map to 3 categories
    df_result['category_3class'] = df_result['category'].apply(map_to_3_categories)
    
    # Create submission ID (assuming standard format)
    # Ensure date format is correct (YYYY-MM-DD)
    df_result['tanggal'] = pd.to_datetime(df_result['tanggal'])
    df_result['id'] = df_result['tanggal'].dt.strftime('%Y-%m-%d') + '_' + df_result['stasiun']
    
    # 3. Analyze Distribution
    print("\nNew Category Distribution (5-Class):")
    print(df_result['category'].value_counts())
    
    print("\nNew Category Distribution (3-Class):")
    print(df_result['category_3class'].value_counts())
    
    # 4. Save
    df_result[['id', 'category']].to_csv('submission_darts_5class.csv', index=False)
    # For 3class submission, we rename the column to 'category' as per format
    df_3class = df_result[['id', 'category_3class']].rename(columns={'category_3class': 'category'})
    df_3class.to_csv('submission_darts_3class.csv', index=False)
    
    print("\nFiles updated successfully:")
    print("  - submission_darts_5class.csv")
    print("  - submission_darts_3class.csv")
    print("\nPreview:")
    print(df_result[['id', 'category', 'max_ispu', 'critical_parameter']].head())

if __name__ == "__main__":
    main()
