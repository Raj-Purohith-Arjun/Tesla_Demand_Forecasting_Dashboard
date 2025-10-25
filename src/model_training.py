"""
Model Training Module - SARIMAX Version
Tesla Demand Forecast Dashboard

This module handles:
1. Training SARIMAX models for each SKU
2. Generating forecasts for different lag scenarios
3. Handling SKU-specific configurations by tier
4. Saving all forecast results
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
import logging
import sys
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForecastEngine:
    """Handle all forecasting operations using SARIMAX"""
    
    def __init__(self):
        """Initialize the forecast engine"""
        self.monthly_data = None
        self.forecasts = []
        logger.info("ForecastEngine initialized (SARIMAX)")
    
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
            logger.error("Please run data_preprocessing.py first!")
            raise
    
    def get_sarimax_params(self, sku_id):
        """
        Get SARIMAX parameters based on SKU tier
        
        Returns order (p,d,q) and seasonal_order (P,D,Q,s)
        """
        tier = config.get_sku_tier(sku_id)
        
        if tier == 'Growth':
            # Stable growth: Simple ARIMA with yearly seasonality
            return (1, 1, 1), (1, 0, 1, 12)
        elif tier == 'High-Volatility':
            # High volatility: More AR/MA terms
            return (2, 1, 2), (1, 0, 1, 12)
        else:  # Declining
            # Declining: Capture trend with minimal seasonality
            return (1, 1, 1), (0, 0, 1, 12)
    
    def train_sarimax_model(self, train_data, sku_id):
        """
        Train SARIMAX model with SKU-specific parameters
        
        Args:
            train_data: Training data (pandas Series with DatetimeIndex)
            sku_id: SKU ID to determine tier-specific parameters
            
        Returns:
            Fitted SARIMAX model
        """
        order, seasonal_order = self.get_sarimax_params(sku_id)
        
        try:
            model = SARIMAX(
                train_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False, maxiter=200)
            return fitted_model
        
        except Exception as e:
            logger.warning(f"SARIMAX failed for SKU {sku_id}, using simpler model: {str(e)[:50]}")
            # Fallback to simple model
            model = SARIMAX(
                train_data,
                order=(1, 1, 0),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=50)
            return fitted_model
    
    def generate_forecast(self, model, steps=1):
        """
        Generate forecast for specified number of steps ahead
        
        Args:
            model: Fitted SARIMAX model
            steps: Number of periods to forecast
            
        Returns:
            float: Forecasted value
        """
        forecast = model.forecast(steps=steps)
        predicted_value = forecast.iloc[-1] if isinstance(forecast, pd.Series) else forecast[-1]
        
        # Ensure non-negative forecasts
        predicted_value = max(0, predicted_value)
        
        return predicted_value
    
    def forecast_single_scenario(self, sku_id, forecast_month, lag_months):
        """
        Generate forecast for a single SKU, month, and lag scenario
        
        Args:
            sku_id: SKU to forecast
            forecast_month: Month to forecast (string 'YYYY-MM')
            lag_months: Number of months lag
            
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
        if len(train_data) < 24:  # Need at least 2 years
            logger.warning(f"⚠️  SKU {sku_id}, Month {forecast_month}, Lag {lag_months}: Insufficient training data ({len(train_data)} months)")
            return None
        
        # Prepare time series with proper datetime index
        train_data = train_data.set_index('Date').sort_index()
        train_series = train_data['Monthly_Sales']
        
        # Train model
        try:
            model = self.train_sarimax_model(train_series, sku_id)
            
            # Calculate steps ahead
            steps_ahead = (forecast_date.year - cutoff_date.year) * 12 + (forecast_date.month - cutoff_date.month)
            
            # Generate forecast
            forecast_value = self.generate_forecast(model, steps=steps_ahead)
            
        except Exception as e:
            logger.error(f"Error forecasting SKU {sku_id}, {forecast_month}, lag {lag_months}: {str(e)[:50]}")
            return None
        
        # Get actual value from validation data
        actual_data = self.monthly_data[
            (self.monthly_data['SKU_ID'] == sku_id) & 
            (self.monthly_data['Date'] == forecast_date)
        ]
        
        actual_value = actual_data['Monthly_Sales'].iloc[0] if len(actual_data) > 0 else None
        
        # Create result dictionary
        result = {
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
    
    def run_all_forecasts(self):
        """
        Generate forecasts for all SKUs, all months, all lag scenarios
        
        This creates all 240 forecasts:
        10 SKUs × 6 months × 4 lag scenarios = 240 forecasts
        """
        print("\n" + "="*80)
        print("GENERATING ALL FORECASTS (SARIMAX)")
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
                    result = self.forecast_single_scenario(sku_id, forecast_month, lag_months)
                    
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
    
    def save_forecasts(self):
        """Save all forecasts to CSV"""
        if self.forecasts is None or len(self.forecasts) == 0:
            logger.error("❌ No forecasts to save. Run run_all_forecasts() first.")
            return
        
        # Ensure output directory exists
        config.FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save main forecast file
        output_path = config.FORECASTS_DIR / "all_forecasts.csv"
        self.forecasts.to_csv(output_path, index=False)
        logger.info(f"✅ Saved forecasts to: {output_path}")
        
        # Also save summary by lag
        summary_by_lag = self.forecasts.groupby(['SKU_ID', 'Tier', 'Lag_Months']).agg({
            'Forecast_Value': 'mean',
            'Actual_Value': 'mean'
        }).round(2).reset_index()
        
        summary_path = config.FORECASTS_DIR / "forecast_summary_by_lag.csv"
        summary_by_lag.to_csv(summary_path, index=False)
        logger.info(f"✅ Saved summary to: {summary_path}")
        
        print("\n" + "="*80)
        print("FORECAST FILES SAVED")
        print("="*80)
        print(f"📁 Main file: {output_path}")
        print(f"📁 Summary file: {summary_path}")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("TESLA DEMAND FORECAST - MODEL TRAINING (SARIMAX)")
    print("="*80 + "\n")
    
    # Initialize engine
    engine = ForecastEngine()
    
    # Load data
    print("STEP 1: Loading preprocessed data...")
    engine.load_monthly_data()
    print()
    
    # Generate all forecasts
    print("STEP 2: Training SARIMAX models and generating forecasts...")
    print("⏳ This will take 2-3 minutes...\n")
    
    forecasts = engine.run_all_forecasts()
    
    # Display sample results
    print("\nSample Forecast Results (first 10):")
    print(forecasts.head(10).to_string(index=False))
    print()
    
    # Save forecasts
    print("STEP 3: Saving forecast results...")
    engine.save_forecasts()
    
    print("="*80)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*80)
    print("\nNext step: Run kpi_calculation.py to evaluate forecast accuracy")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
