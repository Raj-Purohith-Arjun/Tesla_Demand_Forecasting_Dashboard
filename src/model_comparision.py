"""
Model Comparison Module
Tesla Demand Forecast Dashboard

Compares performance of multiple forecasting models:
- SARIMAX
- Prophet (if available)
- Exponential Smoothing
- Moving Average

Identifies best model for each SKU tier and lag scenario.
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare multiple forecasting models"""
    
    def __init__(self):
        """Initialize model comparator"""
        self.models = {}
        self.comparison_results = None
        logger.info("ModelComparator initialized")
    
    def load_all_models(self):
        """Load forecast results from all available models"""
        model_files = {
            'SARIMAX': 'all_forecasts.csv',
            'Exponential_Smoothing': 'exp_smoothing_forecasts.csv',
            'Moving_Average': 'moving_average_forecasts.csv',
            'Prophet': 'prophet_forecasts.csv'  # If available
        }
        
        for model_name, filename in model_files.items():
            file_path = config.FORECASTS_DIR / filename
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    # Filter outliers
                    df = df[df['Forecast_Value'] < 1e6].copy()
                    df['Model'] = model_name
                    self.models[model_name] = df
                    logger.info(f"✅ Loaded {model_name}: {len(df)} forecasts")
                except Exception as e:
                    logger.warning(f"⚠️  Could not load {model_name}: {str(e)}")
            else:
                logger.warning(f"⚠️  {model_name} file not found: {filename}")
        
        if len(self.models) == 0:
            raise ValueError("No model forecasts found!")
        
        print("\n" + "="*80)
        print(f"LOADED {len(self.models)} MODELS FOR COMPARISON")
        print("="*80)
        for model_name in self.models.keys():
            print(f"  ✅ {model_name}")
        print("="*80 + "\n")
        
        return self.models
    
    def calculate_model_kpis(self, model_df):
        """Calculate KPIs for a single model's forecasts"""
        kpi_results = []
        
        for (sku_id, lag), group in model_df.groupby(['SKU_ID', 'Lag_Months']):
            actual = group['Actual_Value'].values
            forecast = group['Forecast_Value'].values
            
            # Skip if insufficient data
            if len(actual) < 2:
                continue
            
            # Calculate MAPE
            mask = actual != 0
            if mask.sum() == 0:
                mape = np.nan
            else:
                mape = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
            
            # Calculate MAE
            mae = np.mean(np.abs(actual - forecast))
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((actual - forecast) ** 2))
            
            # Calculate Bias
            if mask.sum() > 0:
                bias = np.mean((forecast[mask] - actual[mask]) / actual[mask]) * 100
            else:
                bias = np.nan
            
            kpi_results.append({
                'SKU_ID': sku_id,
                'Tier': config.get_sku_tier(sku_id),
                'Lag_Months': lag,
                'MAPE': round(mape, 2) if not np.isnan(mape) else np.nan,
                'MAE': round(mae, 2),
                'RMSE': round(rmse, 2),
                'Bias': round(bias, 2) if not np.isnan(bias) else np.nan,
                'Forecast_Count': len(actual)
            })
        
        return pd.DataFrame(kpi_results)
    
    def compare_all_models(self):
        """Generate comparison table for all models"""
        comparison_results = []
        
        for model_name, model_df in self.models.items():
            kpis = self.calculate_model_kpis(model_df)
            kpis['Model'] = model_name
            comparison_results.append(kpis)
        
        # Combine all results
        self.comparison_results = pd.concat(comparison_results, ignore_index=True)
        
        logger.info(f"✅ Calculated KPIs for {len(self.models)} models")
        
        return self.comparison_results
    
    def find_best_model_by_tier(self):
        """Identify best model for each tier and lag"""
        # Filter reasonable MAPEs
        valid_results = self.comparison_results[self.comparison_results['MAPE'] < 100].copy()
        
        # Group by tier and lag, find model with lowest MAPE
        best_models = []
        
        for (tier, lag), group in valid_results.groupby(['Tier', 'Lag_Months']):
            # Find model with lowest average MAPE
            avg_mape_by_model = group.groupby('Model')['MAPE'].mean()
            best_model = avg_mape_by_model.idxmin()
            best_mape = avg_mape_by_model.min()
            
            best_models.append({
                'Tier': tier,
                'Lag_Months': lag,
                'Best_Model': best_model,
                'Best_MAPE': round(best_mape, 2),
                'Model_Count': len(avg_mape_by_model)
            })
        
        best_models_df = pd.DataFrame(best_models)
        
        return best_models_df
    
    def generate_summary_table(self):
        """Generate overall model comparison summary"""
        # Filter reasonable MAPEs
        valid_results = self.comparison_results[self.comparison_results['MAPE'] < 100].copy()
        
        # Calculate average KPIs per model across all SKUs and lags
        summary = valid_results.groupby('Model').agg({
            'MAPE': 'mean',
            'MAE': 'mean',
            'RMSE': 'mean',
            'Bias': 'mean',
            'Forecast_Count': 'sum'
        }).round(2)
        
        # Sort by MAPE (best model on top)
        summary = summary.sort_values('MAPE')
        
        # Add ranking
        summary['Rank'] = range(1, len(summary) + 1)
        
        return summary
    
    def save_comparison_results(self):
        """Save all comparison results"""
        # Ensure output directory exists
        comparison_dir = config.OUTPUTS_DIR / 'model_comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed comparison
        detailed_path = comparison_dir / 'model_comparison_detailed.csv'
        self.comparison_results.to_csv(detailed_path, index=False)
        logger.info(f"✅ Saved detailed comparison to: {detailed_path}")
        
        # Save summary
        summary = self.generate_summary_table()
        summary_path = comparison_dir / 'model_comparison_summary.csv'
        summary.to_csv(summary_path)
        logger.info(f"✅ Saved summary to: {summary_path}")
        
        # Save best models by tier
        best_models = self.find_best_model_by_tier()
        best_path = comparison_dir / 'best_models_by_tier.csv'
        best_models.to_csv(best_path, index=False)
        logger.info(f"✅ Saved best models to: {best_path}")
        
        print("\n" + "="*80)
        print("MODEL COMPARISON FILES SAVED")
        print("="*80)
        print(f"📁 Detailed: {detailed_path}")
        print(f"📁 Summary: {summary_path}")
        print(f"📁 Best Models: {best_path}")
        print("="*80 + "\n")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("TESLA DEMAND FORECAST - MODEL COMPARISON")
    print("="*80 + "\n")
    
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load all available models
    print("STEP 1: Loading all model forecasts...")
    comparator.load_all_models()
    
    # Compare models
    print("\nSTEP 2: Calculating KPIs for all models...")
    comparison = comparator.compare_all_models()
    
    # Generate summary
    print("\nSTEP 3: Generating comparison summary...")
    summary = comparator.generate_summary_table()
    
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY (Ranked by MAPE)")
    print("="*80)
    print(summary.to_string())
    print("="*80 + "\n")
    
    # Find best models by tier
    print("STEP 4: Identifying best model for each tier...")
    best_models = comparator.find_best_model_by_tier()
    
    print("\n" + "="*80)
    print("BEST MODEL BY SKU TIER AND LAG")
    print("="*80)
    print(best_models.to_string(index=False))
    print("="*80 + "\n")
    
    # Save results
    print("STEP 5: Saving comparison results...")
    comparator.save_comparison_results()
    
    print("="*80)
    print("✅ MODEL COMPARISON COMPLETE!")
    print("="*80)
    print("\nKey Findings:")
    print(f"  • {len(comparator.models)} models compared")
    print(f"  • Best overall model: {summary.index[0]} (MAPE: {summary.iloc[0]['MAPE']:.2f}%)")
    print(f"  • Worst model: {summary.index[-1]} (MAPE: {summary.iloc[-1]['MAPE']:.2f}%)")
    print("\nNext step: Update dashboard to include model selector and comparison tab!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
