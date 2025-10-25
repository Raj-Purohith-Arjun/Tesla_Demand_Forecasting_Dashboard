"""
Data Preprocessing Module
Tesla Demand Forecast Dashboard

This module handles:
1. Loading raw weekly sales data
2. Data validation and quality checks
3. Weekly to monthly aggregation
4. Train/test splits for different lag scenarios
5. Saving processed datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle all data preprocessing operations"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.raw_data = None
        self.monthly_data = None
        self.validation_data = None
        logger.info("DataPreprocessor initialized")
    
    def load_raw_data(self):
        """
        Load raw weekly sales data from CSV
        
        Returns:
            pd.DataFrame: Raw data with Date, SKU_ID, Weekly_Sales columns
        """
        logger.info(f"Loading data from: {config.SAMPLE_DATA_PATH}")
        
        try:
            df = pd.read_csv(config.SAMPLE_DATA_PATH)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(['SKU_ID', 'Date']).reset_index(drop=True)
            
            self.raw_data = df
            logger.info(f"✅ Loaded {len(df)} records for {df['SKU_ID'].nunique()} SKUs")
            
            return df
        
        except FileNotFoundError:
            logger.error(f"❌ File not found: {config.SAMPLE_DATA_PATH}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df):
        """
        Perform data quality checks
        
        Args:
            df: DataFrame to validate
            
        Returns:
            dict: Validation results
        """
        logger.info("Validating data quality...")
        
        validation_results = {
            'total_records': len(df),
            'num_skus': df['SKU_ID'].nunique(),
            'date_range': (df['Date'].min(), df['Date'].max()),
            'missing_values': df.isnull().sum().to_dict(),
            'negative_sales': (df['Weekly_Sales'] < 0).sum(),
            'zero_sales': (df['Weekly_Sales'] == 0).sum(),
            'duplicates': df.duplicated(subset=['Date', 'SKU_ID']).sum()
        }
        
        # Print validation report
        print("\n" + "="*80)
        print("DATA QUALITY VALIDATION REPORT")
        print("="*80)
        print(f"✅ Total Records: {validation_results['total_records']:,}")
        print(f"✅ Number of SKUs: {validation_results['num_skus']}")
        print(f"✅ Date Range: {validation_results['date_range'][0].strftime('%Y-%m-%d')} to {validation_results['date_range'][1].strftime('%Y-%m-%d')}")
        print(f"✅ Missing Values: {sum(validation_results['missing_values'].values())}")
        print(f"✅ Negative Sales: {validation_results['negative_sales']}")
        print(f"⚠️  Zero Sales: {validation_results['zero_sales']} ({validation_results['zero_sales']/len(df)*100:.2f}%)")
        print(f"✅ Duplicates: {validation_results['duplicates']}")
        print("="*80 + "\n")
        
        # Check for critical issues
        if validation_results['duplicates'] > 0:
            logger.warning(f"⚠️  Found {validation_results['duplicates']} duplicate records")
        
        if validation_results['negative_sales'] > 0:
            logger.error(f"❌ Found {validation_results['negative_sales']} negative sales values")
            raise ValueError("Data contains negative sales values!")
        
        logger.info("✅ Data validation complete")
        return validation_results
    
    def aggregate_to_monthly(self, df):
        """
        Convert weekly sales to monthly aggregations
        
        Args:
            df: DataFrame with weekly data
            
        Returns:
            pd.DataFrame: Monthly aggregated data
        """
        logger.info("Aggregating weekly data to monthly...")
        
        # Create year-month column
        df['Year_Month'] = df['Date'].dt.to_period('M')
        
        # Aggregate to monthly
        monthly = df.groupby(['SKU_ID', 'Year_Month']).agg({
            'Weekly_Sales': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Flatten column names
        monthly.columns = ['SKU_ID', 'Year_Month', 'Monthly_Sales', 'Avg_Weekly_Sales', 'Weeks_Count']
        
        # Convert period to timestamp for easier manipulation
        monthly['Date'] = monthly['Year_Month'].dt.to_timestamp()
        
        self.monthly_data = monthly
        logger.info(f"✅ Created {len(monthly)} monthly records")
        
        return monthly
    
    def split_train_validation(self, df):
        """
        Split data into training and validation periods
        
        Args:
            df: DataFrame with monthly data
            
        Returns:
            tuple: (training_df, validation_df)
        """
        logger.info("Splitting data into train and validation sets...")
        
        train = df[df['Date'] < config.VALIDATION_START_DATE].copy()
        validation = df[df['Date'] >= config.VALIDATION_START_DATE].copy()
        
        logger.info(f"✅ Training records: {len(train)}")
        logger.info(f"✅ Validation records: {len(validation)}")
        
        self.validation_data = validation
        
        return train, validation
    
    def create_lag_dataset(self, df, forecast_month, lag_months):
        """
        Create training dataset for a specific forecast month and lag
        
        Args:
            df: Full dataset
            forecast_month: Month to forecast (as string 'YYYY-MM')
            lag_months: Number of months lag (data freshness)
            
        Returns:
            pd.DataFrame: Training data up to (forecast_month - lag_months)
        """
        forecast_date = pd.to_datetime(forecast_month)
        cutoff_date = forecast_date - relativedelta(months=lag_months)
        
        # Get training data up to cutoff
        train_data = df[df['Date'] < cutoff_date].copy()
        
        logger.debug(f"Created lag dataset: Forecast={forecast_month}, Lag={lag_months}mo, Cutoff={cutoff_date.strftime('%Y-%m-%d')}, Records={len(train_data)}")
        
        return train_data
    
    def save_processed_data(self):
        """Save all processed datasets to files"""
        logger.info("Saving processed data...")
        
        # Ensure processed directory exists
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save monthly aggregated data
        if self.monthly_data is not None:
            monthly_path = config.PROCESSED_DATA_DIR / "monthly_aggregated.csv"
            self.monthly_data.to_csv(monthly_path, index=False)
            logger.info(f"✅ Saved monthly data to: {monthly_path}")
        
        # Save validation targets
        if self.validation_data is not None:
            validation_path = config.PROCESSED_DATA_DIR / "validation_targets.csv"
            self.validation_data.to_csv(validation_path, index=False)
            logger.info(f"✅ Saved validation data to: {validation_path}")
        
        # Save SKU metadata
        sku_metadata = []
        for sku_id in config.SKU_IDS:
            tier = config.get_sku_tier(sku_id)
            sku_metadata.append({
                'SKU_ID': sku_id,
                'Tier': tier
            })
        
        metadata_df = pd.DataFrame(sku_metadata)
        metadata_path = config.PROCESSED_DATA_DIR / "sku_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"✅ Saved SKU metadata to: {metadata_path}")
        
        logger.info("✅ All processed data saved successfully")
    
    def get_summary_statistics(self, df):
        """
        Generate summary statistics by SKU
        
        Args:
            df: DataFrame with sales data
            
        Returns:
            pd.DataFrame: Summary statistics
        """
        logger.info("Generating summary statistics...")
        
        summary = df.groupby('SKU_ID')['Weekly_Sales'].agg([
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std', 'std'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('CV%', lambda x: (x.std() / x.mean() * 100))
        ]).round(2)
        
        # Add tier information
        summary['Tier'] = summary.index.map(config.get_sku_tier)
        
        return summary


def main():
    """Main execution function for testing"""
    print("\n" + "="*80)
    print("TESLA DEMAND FORECAST - DATA PREPROCESSING")
    print("="*80 + "\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Step 1: Load raw data
    print("STEP 1: Loading raw data...")
    raw_data = preprocessor.load_raw_data()
    print(f"✅ Loaded {len(raw_data)} records\n")
    
    # Step 2: Validate data
    print("STEP 2: Validating data quality...")
    validation_results = preprocessor.validate_data(raw_data)
    
    # Step 3: Aggregate to monthly
    print("STEP 3: Aggregating to monthly data...")
    monthly_data = preprocessor.aggregate_to_monthly(raw_data)
    print(f"✅ Created {len(monthly_data)} monthly records\n")
    
    # Step 4: Split train/validation
    print("STEP 4: Splitting train/validation sets...")
    train_data, validation_data = preprocessor.split_train_validation(monthly_data)
    print(f"✅ Training: {len(train_data)} records")
    print(f"✅ Validation: {len(validation_data)} records\n")
    
    # Step 5: Show example of lag dataset
    print("STEP 5: Creating example lag dataset...")
    example_lag = preprocessor.create_lag_dataset(monthly_data, '2024-06', 3)
    print(f"✅ Example: Forecasting June 2024 with 3-month lag")
    print(f"   Training data: {len(example_lag)} records up to March 2024\n")
    
    # Step 6: Summary statistics
    print("STEP 6: Generating summary statistics...")
    summary = preprocessor.get_summary_statistics(raw_data)
    print("\nSummary Statistics by SKU:")
    print(summary)
    print()
    
    # Step 7: Save processed data
    print("STEP 7: Saving processed data...")
    preprocessor.save_processed_data()
    print()
    
    print("="*80)
    print("✅ DATA PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"\nProcessed data saved to: {config.PROCESSED_DATA_DIR}")
    print("\nNext step: Run model_training.py to generate forecasts")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
