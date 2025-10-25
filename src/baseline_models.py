"""
Baseline Models Module
Tesla Demand Forecast Dashboard

Implements simple baseline models for comparison:
1. Exponential Smoothing (Holt-Winters)
2. Moving Average (Naive baseline)

These serve as benchmarks to prove the value of SARIMAX/Prophet.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineForecaster:
    """Baseline forecasting models for comparison"""
    
    def __init__(self):
        """Initialize baseline forecaster"""
        self.monthly_data = None
        self.forecasts = []
        logger.info("BaselineForecaster initialized")
    
    def load_monthly_data(self):
        """Load preprocessed monthly data"""
        data_path = config.PROCESSED_DATA_DIR / "monthly_aggregated.csv"
        
        try:
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            self.monthly_data = df
            logger.info(f"✅ Loaded {len(df)} monthly records")
            return df
        except FileNotFoundError:
            logger.error(f"❌ File not found: {data_path}")
            raise
    
    # =========================================================================
    # EXPONENTIAL SMOOTHING (Holt-Winters)
    # =========================================================================
    
    def train_exp_smoothing(self, train_data):
        """
        Train Exponential Smoothing model
        
        Uses additive trend and seasonal components
        Seasonal period = 12 months
        """
        try:
            # Need at least 2 seasonal periods
            if len(train_data) < 24:
                return None
            
            model = ExponentialSmoothing(
                train_data,
                trend='add',
                seasonal='add',
                seasonal_periods=12,
                initialization_method='estimated'
            )
            
            fitted_model = model.fit(optimized=True)
            return fitted_model
        
        except Exception as e:
            # Fall back to simple exponential smoothing
            try:
                model = ExponentialSmoothing(
                    train_data,
                    trend='add',
                    seasonal=None
                )
                fitted_model = model.fit()
                return fitted_model
            except:
                return None
    
    def forecast_exp_smoothing(self, model, steps=1):
        """Generate forecast from Exponential Smoothing model"""
        forecast = model.forecast(steps=steps)
        predicted_value = forecast.iloc[-1] if isinstance(forecast, pd.Series) else forecast[-1]
        
        # Ensure non-negative
        predicted_value = max(0, predicted_value)
        
        return predicted_value
    
    # =========================================================================
    # MOVING AVERAGE (Naive Baseline)
    # =========================================================================
    
    def forecast_moving_average(self, train_data, window=3):
        """
        Simple moving average forecast
        
        Predicts using average of last N months
        Default window = 3 months
        """
        if len(train_data) < window:
            return train_data.mean()
        
        forecast = train_data.tail(window).mean()
        
        # Ensure non-negative
        forecast = max(0, forecast)
        
        return forecast
    
    # =========================================================================
    # UNIFIED FORECASTING INTERFACE
    # =========================================================================
    
    def forecast_single_scenario(self, sku_id, forecast_month, lag_months, model_type='exp_smoothing'):
        """
        Generate forecast for a single SKU, month, and lag scenario
        
        Args:
            sku_id: SKU to forecast
            forecast_month: Month to forecast (string 'YYYY-MM')
            lag_months: Number of months lag
            model_type: 'exp_smoothing' or 'moving_average'
            
        Returns:
            dict: Forecast result
        """
        # Convert forecast month to datetime
        forecast_date = pd.to_datetime(forecast_month + '-01')
        
        # Calculate training data cutoff
        cutoff_date = forecast_date - relativedelta(months=lag_months)
        
        # Get training data up to cutoff
        train_data = self.monthly_data[
            (self.monthly_data['SKU_ID'] == sku_id) & 
            (self.monthly_data['Date'] < cutoff_date)
        ].copy()
        
        # Check if enough training data
        if len(train_data) < 12:  # Need at least 1 year
            logger.warning(f"⚠️  SKU {sku_id}, Month {forecast_month}, Lag {lag_months}: Insufficient training data ({len(train_data)} months)")
            return None
        
        # Prepare time series
        train_data = train_data.set_index('Date').sort_index()
        train_series = train_data['Monthly_Sales']
        
        # Calculate steps ahead
        steps_ahead = (forecast_date.year - cutoff_date.year) * 12 + (forecast_date.month - cutoff_date.month)
        
        # Generate forecast based on model type
        try:
            if model_type == 'exp_smoothing':
                model = self.train_exp_smoothing(train_series)
                if model is None:
                    return None
                forecast_value = self.forecast_exp_smoothing(model, steps=steps_ahead)
            
            elif model_type == 'moving_average':
                forecast_value = self.forecast_moving_average(train_series, window=3)
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        except Exception as e:
            logger.error(f"Error forecasting SKU {sku_id}, {forecast_month}, lag {lag_months}: {str(e)[:50]}")
            return None
        
        # Get actual value
        actual_data = self.monthly_data[
            (self.monthly_data['SKU_ID'] == sku_id) & 
            (self.monthly_data['Date'] == forecast_date)
        ]
        
        actual_value = actual_data['Monthly_Sales'].iloc[0] if len(actual_data) > 0 else None
        
        # Create result dictionary
        result = {
            'Model': model_type,
            'SKU_ID': sku_id,
            'Tier': config.get_sku_tier(sku_id),
            'Forecast_Month': forecast_month,
            'Lag_Months': lag_months,
            'Forecast_Value': round(forecast_value, 2),
            'Actual_Value': round(actual_value, 2) if actual_value is not None else None,
            'Training_Data_Count': len(train_data),
            'Training_Data_Cutoff': cutoff_date.strftime('%Y-%m-%d')
        }
        
        return result
    
    def run_all_forecasts(self, model_type='exp_smoothing'):
        """
        Generate forecasts for all SKUs, all months, all lag scenarios
        
        Args:
            model_type: 'exp_smoothing' or 'moving_average'
        """
        model_name = 'Exponential Smoothing' if model_type == 'exp_smoothing' else 'Moving Average'
        
        print("\n" + "="*80)
        print(f"GENERATING ALL FORECASTS ({model_name})")
        print("="*80)
        print(f"SKUs: {len(config.SKU_IDS)}")
        print(f"Forecast Months: {len(config.FORECAST_MONTHS)}")
        print(f"Lag Scenarios: {config.LAG_SCENARIOS}")
        print(f"Total Forecasts to Generate: {len(config.SKU_IDS) * len(config.FORECAST_MONTHS) * len(config.LAG_SCENARIOS)}")
        print("="*80 + "\n")
        
        forecast_results = []
        total_forecasts = len(config.SKU_IDS) * len(config.FORECAST_MONTHS) * len(config.LAG_SCENARIOS)
        completed = 0
        
        # Iterate through all combinations
        for sku_id in config.SKU_IDS:
            tier = config.get_sku_tier(sku_id)
            print(f"\n📊 Processing SKU {sku_id} ({tier})...")
            
            for forecast_month in config.FORECAST_MONTHS:
                for lag_months in config.LAG_SCENARIOS:
                    
                    # Generate forecast
                    result = self.forecast_single_scenario(sku_id, forecast_month, lag_months, model_type)
                    
                    if result is not None:
                        forecast_results.append(result)
                        completed += 1
                        
                        # Progress indicator
                        if completed % 10 == 0:
                            progress = (completed / total_forecasts) * 100
                            print(f"   Progress: {completed}/{total_forecasts} ({progress:.1f}%)")
        
        # Convert to DataFrame
        self.forecasts = pd.DataFrame(forecast_results)
        
        print("\n" + "="*80)
        print(f"✅ FORECAST GENERATION COMPLETE!")
        print(f"Total forecasts generated: {len(self.forecasts)}")
        print("="*80 + "\n")
        
        return self.forecasts
    
    def save_forecasts(self, model_type='exp_smoothing'):
        """Save forecasts to CSV"""
        if self.forecasts is None or len(self.forecasts) == 0:
            logger.error("❌ No forecasts to save.")
            return
        
        # Ensure output directory exists
        config.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save forecast file
        filename = f"{model_type}_forecasts.csv"
        output_path = config.FORECASTS_DIR / filename
        self.forecasts.to_csv(output_path, index=False)
        logger.info(f"✅ Saved forecasts to: {output_path}")
        
        print("\n" + "="*80)
        print("FORECAST FILE SAVED")
        print("="*80)
        print(f"📁 File: {output_path}")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("TESLA DEMAND FORECAST - BASELINE MODELS")
    print("="*80 + "\n")
    
    # =========================================================================
    # EXPONENTIAL SMOOTHING
    # =========================================================================
    
    print("🔄 GENERATING EXPONENTIAL SMOOTHING FORECASTS...\n")
    
    exp_engine = BaselineForecaster()
    exp_engine.load_monthly_data()
    
    exp_forecasts = exp_engine.run_all_forecasts(model_type='exp_smoothing')
    
    print("\nSample Exp Smoothing Results (first 10):")
    print(exp_forecasts.head(10).to_string(index=False))
    print()
    
    exp_engine.save_forecasts(model_type='exp_smoothing')
    
    # =========================================================================
    # MOVING AVERAGE
    # =========================================================================
    
    print("\n🔄 GENERATING MOVING AVERAGE FORECASTS...\n")
    
    ma_engine = BaselineForecaster()
    ma_engine.load_monthly_data()
    
    ma_forecasts = ma_engine.run_all_forecasts(model_type='moving_average')
    
    print("\nSample Moving Average Results (first 10):")
    print(ma_forecasts.head(10).to_string(index=False))
    print()
    
    ma_engine.save_forecasts(model_type='moving_average')
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("="*80)
    print("✅ BASELINE MODEL TRAINING COMPLETE!")
    print("="*80)
    print(f"Exponential Smoothing forecasts: {len(exp_forecasts)}")
    print(f"Moving Average forecasts: {len(ma_forecasts)}")
    print("\nNext step: Run kpi_calculation.py to evaluate all models")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
