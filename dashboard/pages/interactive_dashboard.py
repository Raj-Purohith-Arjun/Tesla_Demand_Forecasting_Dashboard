"""
Tesla Demand Forecast - Interactive Dashboard
Your original dashboard with all charts and filters
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Page configuration
st.set_page_config(
    page_title="Interactive Dashboard",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def format_mape(value):
    """Format MAPE for display - cap at 999% for readability"""
    if pd.isna(value):
        return "N/A"
    elif value > 999:
        return ">999%"
    else:
        return f"{value:.2f}%"

def calculate_tier_improvements(comparison_df):
    """Calculate improvement by tier, excluding extreme outliers"""
    tier_improvements = []
    
    for tier in ['Growth', 'High-Volatility', 'Declining']:
        tier_data = comparison_df[comparison_df['Tier'] == tier].copy()
        tier_data = tier_data[(tier_data['MAPE_1mo'] < 100) & (tier_data['MAPE_12mo'] < 1000)]
        
        if len(tier_data) > 0:
            avg_1mo = tier_data['MAPE_1mo'].mean()
            avg_12mo = tier_data['MAPE_12mo'].mean()
            avg_improvement = tier_data['Improvement_%'].mean()
            
            tier_improvements.append({
                'Tier': tier,
                'MAPE_1mo': round(avg_1mo, 2),
                'MAPE_12mo': round(avg_12mo, 2),
                'Improvement_%': round(avg_improvement, 2),
                'SKU_Count': len(tier_data)
            })
    
    return pd.DataFrame(tier_improvements)

@st.cache_data
def load_data():
    """Load all required data"""
    try:
        forecasts = pd.read_csv(config.FORECASTS_DIR / "all_forecasts.csv")
        forecasts = forecasts[forecasts['Forecast_Value'] < 1e10]
        kpis = pd.read_csv(config.KPI_DIR / "kpi_results.csv")
        comparison = pd.read_csv(config.KPI_DIR / "lag_comparison.csv")
        return forecasts, kpis, comparison
    except FileNotFoundError:
        st.error("Data files not found. Please run preprocessing and modeling scripts first.")
        st.stop()

def main():
    """Main dashboard function"""
    
    st.title("🔬 Interactive Dashboard")
    st.markdown("### Deep Dive: SKU-Level Analysis")
    st.markdown("---")
    
    forecasts, kpis, comparison = load_data()
    
    # Sidebar filters
    st.sidebar.header("📊 Filters")
    
    selected_lag = st.sidebar.selectbox(
        "Select Forecast Lag (months)",
        options=sorted(kpis['Lag_Months'].unique()),
        index=0,
        help="How old is the data used for forecasting?"
    )
    
    sku_options = ['All SKUs'] + sorted(kpis['SKU_ID'].unique().tolist())
    selected_sku = st.sidebar.selectbox(
        "Select SKU",
        options=sku_options,
        help="Choose a specific SKU or view all"
    )
    
    selected_tier = st.sidebar.radio(
        "Filter by SKU Tier",
        options=['All', 'Growth', 'High-Volatility', 'Declining'],
        help="Filter SKUs by their demand pattern"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About the Dashboard**")
    st.sidebar.info(
        "This dashboard demonstrates how forecast update frequency impacts accuracy. "
        "Shorter lags (fresher data) generally produce better forecasts."
    )
    
    # Filter data
    kpi_filtered = kpis[kpis['Lag_Months'] == selected_lag].copy()
    
    if selected_tier != 'All':
        kpi_filtered = kpi_filtered[kpi_filtered['Tier'] == selected_tier]
    
    if selected_sku != 'All SKUs':
        kpi_filtered = kpi_filtered[kpi_filtered['SKU_ID'] == selected_sku]
    
    # KEY METRICS
    st.markdown("## 📈 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        reasonable_mapes = kpi_filtered[kpi_filtered['MAPE'] < 500]['MAPE']
        avg_mape = reasonable_mapes.mean() if len(reasonable_mapes) > 0 else 0
        st.metric(
            label="Average MAPE",
            value=format_mape(avg_mape),
            delta=f"Lag: {selected_lag} month(s)",
            help="Mean Absolute Percentage Error - Lower is better"
        )
    
    with col2:
        avg_service_level = kpi_filtered['Service_Level_%'].mean()
        st.metric(
            label="Service Level",
            value=f"{avg_service_level:.2f}%",
            delta="Within ±10%",
            help="% of forecasts within acceptable range"
        )
    
    with col3:
        avg_stockout = kpi_filtered['Stockout_Risk_%'].mean()
        st.metric(
            label="Stockout Risk",
            value=f"{avg_stockout:.2f}%",
            delta="Under-forecast",
            delta_color="inverse"
        )
    
    with col4:
        avg_excess = kpi_filtered['Excess_Inventory_Risk_%'].mean()
        st.metric(
            label="Excess Inventory Risk",
            value=f"{avg_excess:.2f}%",
            delta="Over-forecast",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # LAG COMPARISON
    st.markdown("## 🔄 Lag Comparison: 1-Month vs 12-Month")
    
    col1, col2 = st.columns(2)
    
    with col1:
        comp_filtered = comparison.copy()
        if selected_tier != 'All':
            comp_filtered = comp_filtered[comp_filtered['Tier'] == selected_tier]
        if selected_sku != 'All SKUs':
            comp_filtered = comp_filtered[comp_filtered['SKU_ID'] == selected_sku]
        
        comp_display = comp_filtered.copy()
        comp_display['MAPE_1mo'] = comp_display['MAPE_1mo'].clip(upper=200)
        comp_display['MAPE_12mo'] = comp_display['MAPE_12mo'].clip(upper=500)
        
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name='1-Month Lag',
            x=comp_display['SKU_ID'],
            y=comp_display['MAPE_1mo'],
            marker_color='#00B050'
        ))
        fig_comp.add_trace(go.Bar(
            name='12-Month Lag',
            x=comp_display['SKU_ID'],
            y=comp_display['MAPE_12mo'],
            marker_color='#C00000'
        ))
        
        fig_comp.update_layout(
            title="MAPE Comparison by SKU",
            xaxis_title="SKU ID",
            yaxis_title="MAPE (%)",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig_comp, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Improvement by Tier")
        tier_improvements = calculate_tier_improvements(comp_filtered)
        
        if len(tier_improvements) > 0:
            st.dataframe(tier_improvements, use_container_width=True, hide_index=True)
            growth_tier = tier_improvements[tier_improvements['Tier'] == 'Growth']
            if len(growth_tier) > 0:
                st.success(
                    f"**Growth SKUs:** {growth_tier['Improvement_%'].values[0]:.1f}% "
                    f"MAPE reduction with 1-month lag"
                )
    
    st.markdown("---")
    
    # ACCURACY HEATMAP
    st.markdown("## 🗺️ Accuracy Heatmap: All SKUs × All Lags")
    
    heatmap_data = kpis.pivot_table(
        values='MAPE',
        index='SKU_ID',
        columns='Lag_Months',
        aggfunc='mean'
    ).clip(upper=200)
    
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Lag (Months)", y="SKU ID", color="MAPE (%)"),
        color_continuous_scale='RdYlGn_r',
        aspect="auto",
        text_auto='.1f'
    )
    
    fig_heatmap.update_layout(
        title="MAPE by SKU and Lag Scenario (capped at 200%)",
        height=500
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("---")
    
    # FORECAST VS ACTUAL
    if selected_sku != 'All SKUs':
        st.markdown(f"## 📉 Forecast vs Actual: SKU {selected_sku}")
        
        forecast_filtered = forecasts[
            (forecasts['SKU_ID'] == selected_sku) &
            (forecasts['Lag_Months'] == selected_lag)
        ].copy()
        
        fig_forecast = go.Figure()
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_filtered['Forecast_Month'],
            y=forecast_filtered['Actual_Value'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#393C41', width=3)
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_filtered['Forecast_Month'],
            y=forecast_filtered['Forecast_Value'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#E82127', width=3, dash='dash')
        ))
        
        fig_forecast.update_layout(
            title=f"Monthly Sales: Forecast vs Actual (SKU {selected_sku}, {selected_lag}-Month Lag)",
            xaxis_title="Month",
            yaxis_title="Sales Volume",
            height=400
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.markdown("### 📋 Detailed Forecast Results")
        st.dataframe(
            forecast_filtered[['Forecast_Month', 'Forecast_Value', 'Actual_Value', 'Training_Data_Count']],
            use_container_width=True
        )
    
    st.markdown("---")
    
    # RECOMMENDATIONS
    st.markdown("## 💡 Recommendations")
    
    if selected_tier == 'Growth' or selected_tier == 'All':
        st.success(
            "**Growth SKUs (1, 2, 4, 5, 8):** "
            "Use 1-month lag for forecasting. Expected MAPE: 8-11%."
        )
    
    if selected_tier == 'High-Volatility' or selected_tier == 'All':
        st.warning(
            "**High-Volatility SKUs (3, 9):** "
            "Use 1-month lag with wider prediction intervals. Expected MAPE: 12-16%."
        )
    
    if selected_tier == 'Declining' or selected_tier == 'All':
        st.error(
            "**Declining SKUs (6, 7, 10):** "
            "CRITICAL - Use 1-month lag or real-time data. "
            "Longer lags will catastrophically overforecast (200%+ MAPE)."
        )

if __name__ == "__main__":
    main()
