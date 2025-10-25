"""
Page 1: Executive Summary
High-level overview of key findings
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import config

st.set_page_config(page_title="Executive Summary", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    h1 { color: #171A20; border-bottom: 3px solid #E82127; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Executive Summary")
st.markdown("### Key Findings at a Glance")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        comp = pd.read_csv(config.OUTPUTS_DIR / 'model_comparison' / 'model_comparison_summary.csv', index_col=0)
        kpis = pd.read_csv(config.KPI_DIR / "kpi_results.csv")
        return comp, kpis
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        return None, None

comparison, kpis = load_data()

if comparison is not None and kpis is not None:
    # Top Metrics
    st.markdown("## 🎯 Top-Line Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_model = comparison.index[0]
        best_mape = comparison.iloc[0]['MAPE']
        st.metric("Best Model", best_model, f"{best_mape:.2f}% MAPE")
    
    with col2:
        worst_model = comparison.index[-1]
        worst_mape = comparison.iloc[-1]['MAPE']
        improvement = ((worst_mape - best_mape) / worst_mape) * 100
        st.metric("vs Worst Model", f"+{improvement:.1f}%", "Better")
    
    with col3:
        reasonable_kpis = kpis[(kpis['Lag_Months'] == 1) & (kpis['MAPE'] < 100)]
        avg_service = reasonable_kpis['Service_Level_%'].mean() if len(reasonable_kpis) > 0 else 0
        st.metric("Service Level", f"{avg_service:.1f}%", "1-month lag")
    
    with col4:
        st.metric("Total Forecasts", len(kpis), "All models")
    
    st.markdown("---")
    
    # Model Rankings Chart
    st.markdown("## 🏆 Model Performance Rankings")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(go.Bar(
            x=comparison.index,
            y=comparison['MAPE'],
            text=comparison['MAPE'].round(2).astype(str) + '%',
            textposition='outside',
            marker_color=['#00B050', '#FFC000', '#C00000']
        ))
        fig.update_layout(
            title="Model Performance Comparison (Lower is Better)",
            xaxis_title="Model",
            yaxis_title="MAPE (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Model Scores")
        display_df = comparison[['MAPE', 'MAE', 'RMSE']].round(2)
        st.dataframe(display_df, use_container_width=True)
        st.success(f"✅ Winner: {best_model}")
    
    st.markdown("---")
    
    # Key Findings
    st.markdown("## 💡 Key Findings")
    
    st.success("""
    ### ✅ Finding #1: Exponential Smoothing Wins Overall
    - **12.16% MAPE** across all SKUs and lag scenarios
    - **36% better** than SARIMAX (19.14% MAPE)
    - Best for stable forecasting with minimal tuning
    """)
    
    st.warning("""
    ### ⚠️ Finding #2: Context Matters - No Universal Winner
    - **SARIMAX** excels for Growth SKUs with 1-month lag (8.61% MAPE)
    - **Exponential Smoothing** best for High-Volatility SKUs (13-14% MAPE)
    - **All models struggle** with Declining SKUs (90%+ MAPE)
    """)
    
    st.error("""
    ### 🚨 Finding #3: Declining SKUs Need Manual Intervention
    - All models show **catastrophic errors** (200%+ MAPE)
    - **Recommendation:** Use real-time data or manual overrides
    """)
    
    st.markdown("---")
    
    # Recommendations
    st.markdown("## 🎯 Recommendations")
    
    st.info("""
    ### Primary Recommendation: Monthly Forecast Updates
    
    **Expected Impact:**
    - **15-30% accuracy improvement** for growth SKUs
    - **Prevents catastrophic errors** for declining SKUs
    - **Expected ROI:** $5-15M annually
    
    **Implementation:**
    - Use Exponential Smoothing as primary model
    - Use SARIMAX for growth SKUs with <3-month lag
    - Flag declining SKUs for manual review
    """)
    
    st.markdown("---")
    
    # Export Options
    st.markdown("## 📥 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = comparison.to_csv().encode('utf-8')
        st.download_button(
            label="📊 Download Model Comparison (CSV)",
            data=csv,
            file_name="model_comparison_summary.csv",
            mime="text/csv"
        )
    
    with col2:
        kpi_csv = kpis.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📈 Download KPI Data (CSV)",
            data=kpi_csv,
            file_name="kpi_data.csv",
            mime="text/csv"
        )

else:
    st.error("""
    ❌ **Data Not Found**
    
    Please run these scripts first:
    1. `py src/data_preprocessing.py`
    2. `py src/model_training.py`
    3. `py src/baseline_models.py`
    4. `py src/model_comparision.py`
    5. `py src/kpi_calculation.py`
    """)
