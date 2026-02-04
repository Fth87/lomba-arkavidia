"""
ISPU Calculator - Permen LHK No. 14 Tahun 2020
Indonesian Air Pollution Standard Index Calculator
"""
import numpy as np
import pandas as pd
from typing import Union, Dict, Tuple


BREAKPOINTS = {
    'pm10': {
        'concentrations': [0, 50, 150, 350, 420, 500],
        'ispu': [0, 50, 100, 200, 300, 500]
    },
    'pm25': {
        'concentrations': [0, 65, 90, 200, 300, 500], # Tuned: 0-65 BAIK, 65-90 SEDANG (Top 10% Unhealthy)
        'ispu': [0, 50, 100, 200, 300, 500]
    },
    'so2': {
        'concentrations': [0, 80, 365, 800, 1600, 2100], # Reverted/Loosened for older standard
        'ispu': [0, 50, 100, 200, 300, 500]
    },
    'co': {
        'concentrations': [0, 4000, 8000, 15000, 30000, 45000],
        'ispu': [0, 50, 100, 200, 300, 500]
    },
    'o3': {
        'concentrations': [0, 120, 235, 400, 800, 1000],
        'ispu': [0, 50, 100, 200, 300, 500]
    },
    'no2': {
        'concentrations': [0, 80, 200, 1130, 2260, 3750], # Reverted to previous looser values
        'ispu': [0, 50, 100, 200, 300, 500]
    },
    'hc': {
        'concentrations': [0, 45, 100, 215, 432, 600],
        'ispu': [0, 50, 100, 200, 300, 500]
    }
}

CATEGORIES = {
    (0, 50): 'BAIK',
    (51, 100): 'SEDANG',
    (101, 200): 'TIDAK SEHAT',
    (201, 300): 'SANGAT TIDAK SEHAT',
    (301, 500): 'BERBAHAYA'
}


def calculate_ispu(pollutant_type: str, concentration: float) -> float:
    """
    Calculate ISPU using linear interpolation formula.
    I = ((Ia - Ib) / (Xa - Xb)) * (Xx - Xb) + Ib
    
    Args:
        pollutant_type: One of ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
        concentration: Pollutant concentration value
    
    Returns:
        ISPU index value (0-500+)
    """
    pollutant_type = pollutant_type.lower()
    
    if pollutant_type not in BREAKPOINTS:
        raise ValueError(f"Invalid pollutant type: {pollutant_type}")
    
    if concentration < 0:
        return 0
    
    breakpoint = BREAKPOINTS[pollutant_type]
    conc_breaks = breakpoint['concentrations']
    ispu_breaks = breakpoint['ispu']
    
    if concentration >= conc_breaks[-1]:
        return ispu_breaks[-1]
    
    for i in range(len(conc_breaks) - 1):
        if conc_breaks[i] <= concentration <= conc_breaks[i + 1]:
            Xb, Xa = conc_breaks[i], conc_breaks[i + 1]
            Ib, Ia = ispu_breaks[i], ispu_breaks[i + 1]
            Xx = concentration
            
            ispu = ((Ia - Ib) / (Xa - Xb)) * (Xx - Xb) + Ib
            return round(ispu, 2)
    
    return 0


def get_ispu_category(ispu_value: float) -> str:
    """
    Convert ISPU value to category.
    
    Args:
        ispu_value: ISPU index (0-500+)
    
    Returns:
        Category string
    """
    for (min_val, max_val), category in CATEGORIES.items():
        if min_val <= ispu_value <= max_val:
            return category
    
    if ispu_value > 500:
        return 'BERBAHAYA'
    
    return 'BAIK'


def calculate_max_ispu(pollutants: Dict[str, float]) -> Tuple[float, str, str]:
    """
    Calculate maximum ISPU from multiple pollutants (Max-Breakpoint logic).
    
    Args:
        pollutants: Dict of {pollutant_type: concentration}
                   e.g., {'pm10': 75, 'pm25': 35.5, 'o3': 150}
    
    Returns:
        Tuple of (max_ispu, category, critical_pollutant)
    """
    ispu_values = {}
    
    for pollutant, concentration in pollutants.items():
        if pd.notna(concentration) and concentration >= 0:
            try:
                ispu_values[pollutant] = calculate_ispu(pollutant, concentration)
            except ValueError:
                continue
    
    if not ispu_values:
        return 0, 'BAIK', 'none'
    
    max_pollutant = max(ispu_values, key=ispu_values.get)
    max_ispu = ispu_values[max_pollutant]
    category = get_ispu_category(max_ispu)
    
    return max_ispu, category, max_pollutant


def calculate_ispu_for_dataframe(df: pd.DataFrame, 
                                  pollutant_cols: list = None) -> pd.DataFrame:
    """
    Calculate ISPU for each row in a DataFrame.
    
    Args:
        df: DataFrame with pollutant concentration columns
        pollutant_cols: List of pollutant column names. 
                       If None, uses ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2']
    
    Returns:
        DataFrame with added columns: max_ispu, category, critical_parameter
    """
    if pollutant_cols is None:
        pollutant_cols = ['pm10', 'pm25', 'so2', 'co', 'o3', 'no2', 'hc']
    
    available_cols = [col for col in pollutant_cols if col in df.columns]
    
    results = df[available_cols].apply(
        lambda row: calculate_max_ispu(row.to_dict()), 
        axis=1
    )
    
    df_result = df.copy()
    df_result['max_ispu'] = results.apply(lambda x: x[0])
    df_result['category'] = results.apply(lambda x: x[1])
    df_result['critical_parameter'] = results.apply(lambda x: x[2])
    
    return df_result


def map_to_3_categories(category: str) -> str:
    """
    Map 5 ISPU categories to 3 categories for simplified classification.
    
    Args:
        category: Original category
    
    Returns:
        Mapped category (BAIK, SEDANG, or TIDAK SEHAT)
    """
    if category in ['SANGAT TIDAK SEHAT', 'BERBAHAYA']:
        return 'TIDAK SEHAT'
    return category


if __name__ == "__main__":
    # Test cases
    print("="*60)
    print("ISPU Calculator Test Cases")
    print("="*60)
    
    # Test individual pollutants
    test_cases = [
        ('pm10', 75, 50),  # Should be around 50
        ('pm25', 35.5, 75.8),  # Should be around 75
        ('o3', 150, 62.5),  # Should be around 62.5
        ('co', 5000, 56.25),  # Should be around 56.25
    ]
    
    print("\n1. Individual Pollutant Tests:")
    for pollutant, conc, expected in test_cases:
        result = calculate_ispu(pollutant, conc)
        category = get_ispu_category(result)
        print(f"   {pollutant.upper():5s} = {conc:6.1f} → ISPU: {result:6.2f} ({category})")
    
    # Test max ISPU
    print("\n2. Max ISPU Test:")
    pollutants = {
        'pm10': 75,
        'pm25': 35.5,
        'so2': 100,
        'co': 5000,
        'o3': 150,
        'no2': 100
    }
    
    max_ispu, category, critical = calculate_max_ispu(pollutants)
    print(f"   Pollutants: {pollutants}")
    print(f"   Max ISPU: {max_ispu:.2f}")
    print(f"   Category: {category}")
    print(f"   Critical Pollutant: {critical.upper()}")
    
    # Test DataFrame
    print("\n3. DataFrame Test:")
    df_test = pd.DataFrame([
        {'pm10': 75, 'pm25': 35.5, 'o3': 150, 'no2': 100},
        {'pm10': 200, 'pm25': 80, 'o3': 300, 'no2': 150},
        {'pm10': 400, 'pm25': 200, 'o3': 500, 'no2': 1500}
    ])
    
    df_result = calculate_ispu_for_dataframe(df_test)
    print(df_result[['pm10', 'pm25', 'max_ispu', 'category', 'critical_parameter']])
    
    print("\n" + "="*60)
