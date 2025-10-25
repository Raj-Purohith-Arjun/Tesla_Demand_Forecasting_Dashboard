"""
Utility Functions for Tesla Demand Forecast Dashboard
Provides export, styling, and helper functions
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import base64


# ==============================================================================
# STYLING & AESTHETICS
# ==============================================================================

def apply_tesla_theme():
    """Apply Tesla-inspired styling to the dashboard"""
    st.markdown("""
        <style>
        /* Tesla brand colors */
        :root {
            --tesla-red: #E82127;
            --tesla-blue: #3E6AE1;
            --tesla-dark: #171A20;
            --tesla-gray: #393C41;
            --tesla-light: #F4F4F4;
        }
        
        /* Main app styling */
        .stApp {
            background-color: #FFFFFF;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 600;
        }
        
        /* Headers */
        h1 {
            color: var(--tesla-dark);
            font-weight: 700;
            padding-bottom: 1rem;
            border-bottom: 3px solid var(--tesla-red);
        }
        
        h2 {
            color: var(--tesla-dark);
            font-weight: 600;
            margin-top: 2rem;
        }
        
        h3 {
            color: var(--tesla-gray);
            font-weight: 500;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: var(--tesla-light);
        }
        
        /* Buttons */
        .stButton>button {
            background-color: var(--tesla-red);
            color: white;
            font-weight: 600;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 2rem;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #C51D23;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid var(--tesla-blue);
        }
        
        /* Tooltips */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
            color: var(--tesla-blue);
            font-weight: 600;
        }
        
        /* Loading animation */
        .stSpinner > div {
            border-color: var(--tesla-red) transparent transparent transparent;
        }
        
        /* Tables */
        .dataframe {
            font-size: 0.9rem;
        }
        
        .dataframe th {
            background-color: var(--tesla-gray);
            color: white;
            font-weight: 600;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--tesla-light);
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)


def create_metric_card(label, value, delta=None, help_text=None):
    """Create a styled metric card with optional tooltip"""
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.metric(label=label, value=value, delta=delta)
    
    if help_text:
        with col2:
            st.markdown(f"<span class='tooltip' title='{help_text}'>ℹ️</span>", 
                       unsafe_allow_html=True)


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_dataframe_to_csv(df, filename="data.csv"):
    """Convert dataframe to CSV download"""
    csv = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}"
    )


def export_plotly_chart(fig, filename="chart.png"):
    """Export Plotly chart as PNG"""
    img_bytes = fig.to_image(format="png", width=1200, height=600)
    
    st.download_button(
        label="📥 Download Chart (PNG)",
        data=img_bytes,
        file_name=filename,
        mime="image/png",
        key=f"download_{filename}"
    )


def create_pdf_report_link():
    """Create link to generate PDF report"""
    st.markdown("""
        ### 📄 Export Full Report
        
        Click below to generate a comprehensive PDF report including:
        - Executive Summary
        - Model Comparison Analysis
        - KPI Metrics & Trends
        - Recommendations
    """)
    
    if st.button("🎯 Generate PDF Report", key="generate_pdf"):
        with st.spinner("Generating PDF report..."):
            st.info("PDF generation will be implemented in final version. For now, use browser Print → Save as PDF.")


# ==============================================================================
# DATA LOADING WITH CACHING
# ==============================================================================

@st.cache_data
def load_forecast_data(model_name='SARIMAX'):
    """Load forecast data for specified model with caching"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import config
    
    # Map model names to file names
    model_files = {
        'SARIMAX': 'all_forecasts.csv',
        'Exponential Smoothing': 'exp_smoothing_forecasts.csv',
        'Moving Average': 'moving_average_forecasts.csv'
    }
    
    file_path = config.FORECASTS_DIR / model_files.get(model_name, 'all_forecasts.csv')
    
    if file_path.exists():
        df = pd.read_csv(file_path)
        # Filter extreme outliers
        df = df[df['Forecast_Value'] < 1e6].copy()
        return df
    else:
        st.error(f"Forecast file not found: {file_path}")
        return pd.DataFrame()


@st.cache_data
def load_kpi_data():
    """Load KPI summary data with caching"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import config
    
    kpi_path = config.OUTPUTS_DIR / 'kpi_summary.csv'
    
    if kpi_path.exists():
        return pd.read_csv(kpi_path)
    else:
        st.error(f"KPI file not found: {kpi_path}")
        return pd.DataFrame()


@st.cache_data
def load_model_comparison():
    """Load model comparison data with caching"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    import config
    
    comparison_path = config.OUTPUTS_DIR / 'model_comparison' / 'model_comparison_summary.csv'
    
    if comparison_path.exists():
        return pd.read_csv(comparison_path, index_col=0)
    else:
        return pd.DataFrame()


# ==============================================================================
# TOOLTIPS & HELP TEXT
# ==============================================================================

METRIC_DEFINITIONS = {
    'MAPE': 'Mean Absolute Percentage Error - Average % deviation from actual. Lower is better. <10% is excellent.',
    'MAE': 'Mean Absolute Error - Average absolute difference from actual. Lower is better. Unit: same as forecast.',
    'RMSE': 'Root Mean Square Error - Penalizes large errors more heavily. Lower is better.',
    'Bias': 'Average tendency to over/under forecast. Negative = under-forecast, Positive = over-forecast.',
    'Service Level': '% of forecasts within ±10% of actual. Higher is better. Target: >90%.',
    'Stockout Risk': '% of forecasts that under-predicted demand. Lower is better for customer satisfaction.',
    'Excess Inventory Risk': '% of forecasts that over-predicted demand. Lower is better for cash flow.',
    'Perfect Forecast Rate': '% of forecasts within ±5% of actual. Higher is better.'
}


def show_metric_definition(metric_name):
    """Display tooltip for metric definition"""
    if metric_name in METRIC_DEFINITIONS:
        st.info(f"**{metric_name}**: {METRIC_DEFINITIONS[metric_name]}")


# ==============================================================================
# FORMATTING HELPERS
# ==============================================================================

def format_percentage(value, decimals=2):
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def format_number(value, decimals=2):
    """Format number with thousands separator"""
    return f"{value:,.{decimals}f}"


def color_code_mape(mape):
    """Return color based on MAPE value"""
    if mape < 10:
        return "🟢 Excellent"
    elif mape < 20:
        return "🟡 Good"
    elif mape < 30:
        return "🟠 Fair"
    else:
        return "🔴 Poor"
