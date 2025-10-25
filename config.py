"""
Configuration file for Tesla Demand Forecast Dashboard
Contains all project settings, paths, and model parameters
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
KPI_DIR = OUTPUTS_DIR / "kpis"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Data file paths (NOTE: Your file has a SPACE in the name)
SAMPLE_DATA_PATH = RAW_DATA_DIR / "Sample Data.csv"

# ============================================================================
# DATA PARAMETERS
# ============================================================================
# Validation period for forecast evaluation
VALIDATION_START_DATE = "2024-01-01"
VALIDATION_END_DATE = "2024-06-30"

# Historical training period
HISTORICAL_START_DATE = "2019-01-06"
HISTORICAL_END_DATE = "2023-12-31"

# SKU configuration
NUM_SKUS = 10
SKU_IDS = list(range(1, NUM_SKUS + 1))

# SKU Tier Classification (based on our analysis)
SKU_TIERS = {
    'Growth': [1, 2, 4, 5, 8],
    'High-Volatility': [3, 9],
    'Declining': [6, 7, 10]
}

# ============================================================================
# FORECASTING PARAMETERS
# ============================================================================
# Lag scenarios to test (in months)
LAG_SCENARIOS = [1, 3, 6, 12]

# Forecast target months (Jan-Jun 2024)
FORECAST_MONTHS = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']

# ============================================================================
# PROPHET MODEL PARAMETERS (by SKU Tier)
# ============================================================================
PROPHET_PARAMS = {
    'Growth': {
        'growth': 'linear',
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'interval_width': 0.80
    },
    'High-Volatility': {
        'growth': 'linear',
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.10,
        'seasonality_prior_scale': 10.0,
        'interval_width': 0.90
    },
    'Declining': {
        'growth': 'linear',
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.15,
        'seasonality_prior_scale': 10.0,
        'interval_width': 0.80,
        'floor': 0  # Prevent negative forecasts
    }
}

# ============================================================================
# KPI THRESHOLDS
# ============================================================================
# For business impact metrics
STOCKOUT_THRESHOLD = 0.80  # Forecast < 80% of actual
EXCESS_INVENTORY_THRESHOLD = 1.20  # Forecast > 120% of actual
SERVICE_LEVEL_TOLERANCE = 0.10  # Within ±10% of actual

# Target performance levels
TARGET_SERVICE_LEVEL = 0.75  # 75% of forecasts within tolerance
TARGET_STOCKOUT_RISK = 0.10  # <10% stockout risk
TARGET_EXCESS_RISK = 0.15  # <15% excess inventory risk

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================
DASHBOARD_TITLE = "Tesla Demand Forecast Dashboard"
DASHBOARD_SUBTITLE = "Impact of Data Recency on Forecast Accuracy"

# Color scheme (Tesla-inspired)
COLOR_SCHEME = {
    'primary': '#E82127',  # Tesla Red
    'secondary': '#393C41',  # Tesla Dark Gray
    'success': '#00B050',  # Green for good performance
    'warning': '#FFC000',  # Yellow for moderate
    'danger': '#C00000',  # Red for poor performance
    'background': '#FFFFFF',
    'text': '#000000'
}

# Chart settings
CHART_HEIGHT = 400
CHART_WIDTH = 600

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_sku_tier(sku_id):
    """Return the tier classification for a given SKU ID"""
    for tier, sku_list in SKU_TIERS.items():
        if sku_id in sku_list:
            return tier
    return 'Unknown'

def get_prophet_params(sku_id):
    """Return Prophet parameters for a given SKU based on its tier"""
    tier = get_sku_tier(sku_id)
    return PROPHET_PARAMS.get(tier, PROPHET_PARAMS['Growth'])

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

if __name__ == "__main__":
    # Test configuration
    print("="*80)
    print("TESLA DEMAND FORECAST - CONFIGURATION TEST")
    print("="*80)
    print(f"\n📁 Project Root: {PROJECT_ROOT}")
    print(f"📊 Sample Data Path: {SAMPLE_DATA_PATH}")
    print(f"✅ Data file exists: {SAMPLE_DATA_PATH.exists()}")
    print(f"\n🎯 SKUs: {SKU_IDS}")
    print(f"📅 Validation Period: {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")
    print(f"🔄 Lag Scenarios: {LAG_SCENARIOS} months")
    print("\n✅ Configuration loaded successfully!")
