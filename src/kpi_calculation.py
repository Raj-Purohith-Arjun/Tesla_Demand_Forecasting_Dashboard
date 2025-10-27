"""
KPI Calculation Module
Tesla Demand Forecast Dashboard

This module calculates:
1. Accuracy metrics (MAPE, MAE, RMSE, Bias)
2. Business impact metrics (Stockout Risk, Excess Inventory, Service Level)
3. Comparative metrics (Accuracy improvement by lag)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class KPICalculator:
    def __init__(self):
        self.forecasts = None
        self.kpi_results = None
        logger.info("KPICalculator initialized")

    def load_forecasts(self):
        forecast_path = config.FORECASTS_DIR / "all_forecasts.csv"
        try:
            df = pd.read_csv(forecast_path)
            # Outlier handling
            for sku_id in df['SKU_ID'].unique():
                sku_mask = df['SKU_ID'] == sku_id
                max_actual = df.loc[sku_mask, 'Actual_Value'].max()
                reasonable_cap = max_actual * 10
                df.loc[sku_mask & (df['Forecast_Value'] > reasonable_cap), 'Forecast_Value'] = reasonable_cap
                if max_actual < 50:
                    df.loc[sku_mask & (df['Forecast_Value'] > 500), 'Forecast_Value'] = 500
            df = df[df['Forecast_Value'] < 1e6].copy()
            self.forecasts = df
            logger.info(f"✅ Loaded {len(df)} forecasts (outliers capped)")
            return df
        except Exception as e:
            logger.error(f"Failed to load forecasts: {e}")
            raise

    # ACCURACY METRICS
    def calculate_mape(self, actual, forecast):
        mask = actual != 0
        if mask.sum() == 0:
            return np.nan
        mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
        return mape

    def calculate_mae(self, actual, forecast):
        return np.mean(np.abs(actual - forecast))

    def calculate_rmse(self, actual, forecast):
        return np.sqrt(np.mean((actual - forecast) ** 2))

    def calculate_bias(self, actual, forecast):
        mask = actual != 0
        if mask.sum() == 0:
            return np.nan
        bias = np.mean((forecast[mask] - actual[mask]) / actual[mask]) * 100
        return bias

    # BUSINESS IMPACT METRICS
    def calculate_stockout_risk(self, actual, forecast):
        threshold = config.STOCKOUT_THRESHOLD
        stockout_count = np.sum(forecast < threshold * actual)
        stockout_risk = (stockout_count / len(actual)) * 100
        return stockout_risk

    def calculate_excess_inventory_risk(self, actual, forecast):
        threshold = config.EXCESS_INVENTORY_THRESHOLD
        excess_count = np.sum(forecast > threshold * actual)
        excess_risk = (excess_count / len(actual)) * 100
        return excess_risk

    def calculate_service_level(self, actual, forecast):
        tolerance = config.SERVICE_LEVEL_TOLERANCE
        lower_bound = actual * (1 - tolerance)
        upper_bound = actual * (1 + tolerance)
        within_tolerance = np.sum((forecast >= lower_bound) & (forecast <= upper_bound))
        service_level = (within_tolerance / len(actual)) * 100
        return service_level

    def calculate_perfect_forecast_rate(self, actual, forecast):
        tolerance = 0.05
        lower_bound = actual * (1 - tolerance)
        upper_bound = actual * (1 + tolerance)
        within_tolerance = np.sum((forecast >= lower_bound) & (forecast <= upper_bound))
        perfect_rate = (within_tolerance / len(actual)) * 100
        return perfect_rate

    # GROUPED KPI CALCULATION
    def calculate_kpis_by_group(self, df, group_by=['SKU_ID', 'Lag_Months']):
        logger.info(f"Calculating KPIs grouped by {group_by}...")
        kpi_results = []
        for group_keys, group_data in df.groupby(group_by):
            actual = group_data['Actual_Value'].values
            forecast = group_data['Forecast_Value'].values
            if len(actual) < 2:
                continue
            result = {
                group_by[0]: group_keys[0] if len(group_by) > 1 else group_keys,
                group_by[1]: group_keys[1] if len(group_by) > 1 else None
            }
            if 'SKU_ID' in group_by:
                result['Tier'] = config.get_sku_tier(group_keys[0] if len(group_by) > 1 else group_keys)
            result['MAPE'] = round(self.calculate_mape(actual, forecast), 2)
            result['MAE'] = round(self.calculate_mae(actual, forecast), 2)
            result['RMSE'] = round(self.calculate_rmse(actual, forecast), 2)
            result['Bias'] = round(self.calculate_bias(actual, forecast), 2)
            result['Stockout_Risk_%'] = round(self.calculate_stockout_risk(actual, forecast), 2)
            result['Excess_Inventory_Risk_%'] = round(self.calculate_excess_inventory_risk(actual, forecast), 2)
            result['Service_Level_%'] = round(self.calculate_service_level(actual, forecast), 2)
            result['Perfect_Forecast_Rate_%'] = round(self.calculate_perfect_forecast_rate(actual, forecast), 2)
            result['Forecast_Count'] = len(actual)
            result['Avg_Actual'] = round(np.mean(actual), 2)
            result['Avg_Forecast'] = round(np.mean(forecast), 2)
            kpi_results.append(result)
        kpi_df = pd.DataFrame(kpi_results)
        logger.info(f"✅ Calculated KPIs for {len(kpi_df)} groups")
        return kpi_df

    def calculate_lag_comparison(self, kpi_df):
        logger.info("Calculating lag comparison metrics...")
        comparison_results = []
        for sku_id in kpi_df['SKU_ID'].unique():
            sku_data = kpi_df[kpi_df['SKU_ID'] == sku_id].copy()
            lag_1 = sku_data[sku_data['Lag_Months'] == 1]
            lag_12 = sku_data[sku_data['Lag_Months'] == 12]
            if len(lag_1) == 0 or len(lag_12) == 0:
                continue
            mape_1 = lag_1['MAPE'].iloc[0]
            mape_12 = lag_12['MAPE'].iloc[0]
            if not np.isnan(mape_12) and mape_12 != 0:
                improvement = ((mape_12 - mape_1) / mape_12 * 100)
            else:
                improvement = np.nan
            comparison_results.append({
                'SKU_ID': sku_id,
                'Tier': config.get_sku_tier(sku_id),
                'MAPE_1mo': mape_1,
                'MAPE_12mo': mape_12,
                'Improvement_%': round(improvement, 2) if not np.isnan(improvement) else np.nan
            })
        comparison_df = pd.DataFrame(comparison_results)
        logger.info(f"✅ Calculated lag comparisons for {len(comparison_df)} SKUs")
        return comparison_df

    def save_kpis(self, kpi_df, comparison_df):
        config.KPI_DIR.mkdir(parents=True, exist_ok=True)
        kpi_path = config.KPI_DIR / "kpi_results.csv"
        kpi_df.to_csv(kpi_path, index=False)
        logger.info(f"✅ Saved KPI results to: {kpi_path}")
        comparison_path = config.KPI_DIR / "lag_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"✅ Saved lag comparison to: {comparison_path}")
        tier_summary = kpi_df.groupby(['Tier', 'Lag_Months']).agg({
            'MAPE': 'mean',
            'MAE': 'mean',
            'RMSE': 'mean',
            'Service_Level_%': 'mean'
        }).round(2).reset_index()
        tier_path = config.KPI_DIR / "kpi_summary_by_tier.csv"
        tier_summary.to_csv(tier_path, index=False)
        logger.info(f"✅ Saved tier summary to: {tier_path}")
        print("\n" + "="*80)
        print("KPI FILES SAVED")
        print("="*80)
        print(f"📁 Main KPIs: {kpi_path}")
        print(f"📁 Lag Comparison: {comparison_path}")
        print(f"📁 Tier Summary: {tier_path}")
        print("="*80 + "\n")

def main():
    print("\n" + "="*80)
    print("TESLA DEMAND FORECAST - KPI CALCULATION")
    print("="*80 + "\n")
    calculator = KPICalculator()
    print("STEP 1: Loading forecast results...")
    forecasts = calculator.load_forecasts()
    print(f"✅ Loaded {len(forecasts)} forecasts\n")
    print("STEP 2: Calculating KPIs...")
    kpi_results = calculator.calculate_kpis_by_group(forecasts)
    print("\nSample KPI Results (first 10):")
    print(kpi_results.head(10).to_string(index=False))
    print()
    print("STEP 3: Calculating lag comparison (1-month vs 12-month)...")
    comparison = calculator.calculate_lag_comparison(kpi_results)
    print("\nLag Comparison Results:")
    print(comparison.to_string(index=False))
    print()
    print("STEP 4: Saving KPI results...")
    calculator.save_kpis(kpi_results, comparison)
    print("="*80)
    print("KPI SUMMARY BY TIER AND LAG")
    print("="*80)
    tier_summary = kpi_results.groupby(['Tier', 'Lag_Months']).agg({
        'MAPE': 'mean',
        'Service_Level_%': 'mean'
    }).round(2)
    print(tier_summary)
    print()
    print("="*80)
    print(" KPI CALCULATION COMPLETE!")
    print("="*80)
    print("\nKey Insights:")
    print(f"  • Average MAPE (all SKUs, all lags): {kpi_results['MAPE'].mean():.2f}%")
    print(f"  • Best average service level: {kpi_results['Service_Level_%'].max():.2f}%")
    print(f"  • Average improvement (1mo vs 12mo lag): {comparison['Improvement_%'].mean():.2f}%")
    print("\nNext step: Run dashboard/app.py to launch the interactive dashboard!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
