# Tesla Demand Forecast Dashboard

## Objective

Tesla’s mission is to accelerate the world’s transition to sustainable energy—requiring an agile, data-driven supply chain. Accurate demand forecasting is critical to:

- Optimize production, reduce costs, and prevent stockouts or write-offs
- Respond quickly to volatile EV market trends and new model launches
- Deliver outstanding customer experience with on-time order fulfillment
- Enable evidence-based decisions by supply chain and business leadership

**This system delivers a production-grade dashboard showing, in business terms, how forecast update frequency affects accuracy and financial outcomes.**

---

## Why Tesla Needs This

- **Minimize Risk**: Adapts forecasts as market and supply chain signals change, protecting against overproduction and lost sales
- **Optimized Inventory**: Reduces both excess stock and missed revenue via more timely, segment-aware forecasts
- **Agility**: Lets Tesla go-to-market faster and respond to new demand signals—for instance, Cybertruck launch, energy products, or regulatory changes
- **Operational Trust**: Models are transparent, explainable, and easily benchmarked—essential for large-scale adoption in a high-accountability environment

---

## Approach & Model Selection

### Data & Scope

- **Historical weekly sales** (2019–2024), 10 SKUs grouped by “Growth”, “High Volatility”, “Declining”
- **Forecast lags**: 1, 3, 6, 12 months—measuring business risk of stale vs. fresh data
- **All code is reproducible** (Python, Streamlit, Plotly)

### Forecast Models Chosen

- **Moving Average** (Baseline): Simple, quick to compute, easy for operations staff to trust and interpret
- **Exponential Smoothing**: Industry standard for supply chain and robust to volatility, with minimal tuning needed
- **SARIMAX**: Captures trend, seasonality, and calendar effects—state-of-the-art for production/time series analysis

**Validated all models via walk-forward simulation, benchmarking KPI improvement by segment and scenario—exactly as a Tesla data science/ops team would evaluate a vendor or new system.**

---

## Implementation Overview

1. **Data Pipeline**: Ingest, clean, and aggregate demand data to monthly summaries (business-relevant)
2. **Model Training**: Automated training and walk-forward validation (simulating real-world SKU launch/phaseout behavior)
3. **KPI Calculation**: MAPE, MAE, RMSE, Bias, Stockout/Excess Risk, Service Level
4. **Dashboard UX**: Executive-level summary, full interactivity, model “A/B” switch for scenario planning
5. **Export/Reporting**: Downloadable tables, plots, and full PDF report for supply chain/regulatory recordkeeping

---

## Dashboard Features

- **Model Selector**: Instantly compare forecasting approaches on your actual portfolio
- **Business KPIs**: Evaluate decisions using executive metrics—MAPE, service level, financial impact
- **Drilldown Analytics**: Filters by SKU, demand tier, and scenario for root cause analysis
- **Model Comparison**: See side-by-side rankings and statistical tests; recommended models by SKU type and time horizon
- **Executive Summary**: Business case, financial impact, and implementation roadmap
- **Full Methodology Tab**: Transparent documentation for team handover or audit
- **Professional Design**: Tesla-style layout using Streamlit and Plotly for enterprise-grade usability

---

## Business Value

- **Reduces forecast error by 15–30%** for Growth SKUs
- **Saves $5–15M per year** in combined stockout losses and inventory carrying costs (per scenario simulation)
- **Increases operations resilience** and customer satisfaction
- **Enables rapid, confident decision-making at all levels**

---

## How to Use

1. **Install requirements:**



2. **Run the pipeline:**
- `py src/data_preprocessing.py`
- `py src/model_training.py`
- `py src/baseline_models.py`
- `py src/model_comparision.py`
- `py src/kpi_calculation.py`
3. **Launch the dashboard:**


4. **Navigate with the sidebar:**
- Executive Summary (business view)
- Interactive Dashboard (deep dive, filtering)
- Model Comparison (quant/proof)
- Comprehensive Report (audit/hand-off)

5. **Export results** as CSV or PDF for review

---


