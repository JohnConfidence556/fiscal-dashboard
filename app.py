# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# -----------------------
# Load Cleaned Data
# -----------------------
pivot_data = pd.read_csv("africa_sovereign_debt_cleaned.csv")

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="Africa Sovereign Debt Dashboard", layout="wide")
st.title("10alytics Hackathon 2025: Africa Sovereign Debt Dashboard")

# -----------------------
# Sidebar Controls
# -----------------------
countries = pivot_data['Country'].unique()
selected_countries = st.sidebar.multiselect("Select Countries", countries, default=countries[:5])

years = sorted(pivot_data['Year'].unique())
year_range = st.sidebar.slider("Select Year Range", int(min(years)), int(max(years)),
                               value=(int(min(years)), int(max(years))))

st.sidebar.subheader("Global Scenario Analysis")
gdp_growth_factor = st.sidebar.slider("GDP Growth Shock Factor", 0.8, 1.2, 1.0, 0.01)
rev_exp_factor = st.sidebar.slider("Revenue/Expenditure Adjustment Factor", 0.8, 1.2, 1.0, 0.01)
#

# -----------------------
# Filter Data
# -----------------------
df = pivot_data[(pivot_data['Country'].isin(selected_countries)) &
                (pivot_data['Year'] >= year_range[0]) &
                (pivot_data['Year'] <= year_range[1])].copy()

# -----------------------
# Prepare Modeling Data
# -----------------------
df_model = pivot_data.copy()
df_cls_model = pivot_data.copy()

# Encode Country as numeric for modeling
for d in [df, df_model, df_cls_model]:
    d['Country_Code_Num'] = d['Country'].astype('category').cat.codes

# Feature columns
feature_cols = [
    'rev_exp_ratio', 'deficit_gdp_ratio', 'cumulative_deficit', 'deficit_volatility_3yr',
    'gdp_growth_rate', 'gdp_per_capita_calc', 'gdp_real_nominal_gap_pct',
    'trade_balance_pct_gdp', 'gdp_volatility_5yr',
    'debt_accumulation_rate', 'debt_carrying_capacity', 'fiscal_vulnerability_index',
    'interest_burden_ratio', 'Country_Code_Num'
]

# Fill missing values
df_model[feature_cols] = df_model[feature_cols].fillna(df_model[feature_cols].median())
df_cls_model[feature_cols] = df_cls_model[feature_cols].fillna(df_cls_model[feature_cols].median())
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Numeric columns for plots
numeric_plot_cols = ['population', 'revenue', 'expenditure', 'gdp_growth_rate', 'unemployment_rate',
                     'exports_gdp_ratio', 'imports_gdp_ratio', 'cumulative_deficit']
for col in numeric_plot_cols:
    df[col] = df[col].fillna(df[col].median())

# Drop rows where target is NaN
df_model = df_model.dropna(subset=['debt_to_gdp'])
df_cls_model = df_cls_model.dropna(subset=['high_debt_risk'])

# Apply global scenario adjustments
df['gdp_growth_rate'] *= gdp_growth_factor
df['rev_exp_ratio'] *= rev_exp_factor

# -----------------------
# Regression: Debt-to-GDP Forecast
# -----------------------
train_mask = df_model['Year'] <= 2020
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_reg.fit(df_model[feature_cols][train_mask], df_model['debt_to_gdp'][train_mask])
df['debt_to_gdp_forecast'] = rf_reg.predict(df[feature_cols])

# -----------------------
# Classification: High Debt Risk
# -----------------------
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_cls.fit(df_cls_model[feature_cols], df_cls_model['high_debt_risk'])
df['high_debt_risk_prob'] = rf_cls.predict_proba(df[feature_cols])[:, 1]

# Flags
df['extreme_deficit_flag'] = df['deficit_gdp_ratio'] < -5
df['high_debt_flag'] = df['debt_to_gdp_forecast'] > 60

# Remove NaNs in predictions
df_plot = df.dropna(subset=['debt_to_gdp_forecast', 'high_debt_risk_prob'])

# -----------------------
# Feature Importance: Debt-to-GDP Forecast
# -----------------------
st.subheader("📌 Feature Importance: Debt-to-GDP Forecast")
reg_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_reg.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig_reg_feat = px.bar(reg_importance, x='Importance', y='Feature', orientation='h',
                      text='Importance', color='Importance', color_continuous_scale='Viridis')
st.plotly_chart(fig_reg_feat, use_container_width=True)

# -----------------------
# Feature Importance: High Debt Risk
# -----------------------
st.subheader("📌 Feature Importance: High Debt Risk Probability")
cls_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_cls.feature_importances_
}).sort_values(by='Importance', ascending=False)

fig_cls_feat = px.bar(cls_importance, x='Importance', y='Feature', orientation='h',
                      text='Importance', color='Importance', color_continuous_scale='Cividis')
st.plotly_chart(fig_cls_feat, use_container_width=True)

# Dashboard Visualizations
# -----------------------

st.subheader("📊 Debt-to-GDP Trend vs High Debt Risk")
fig_debt = px.line(df_plot, x='Year', y='debt_to_gdp_forecast', color='Country', markers=True,
                   labels={'debt_to_gdp_forecast': 'Debt-to-GDP Forecast (%)'})
fig_debt.add_scatter(x=df_plot[df_plot['high_debt_flag']]['Year'],
                     y=df_plot[df_plot['high_debt_flag']]['debt_to_gdp_forecast'],
                     mode='markers', marker=dict(color='red', size=10),
                     name='High Debt (>60%)')
st.plotly_chart(fig_debt, use_container_width=True)
st.markdown("*Insight: Countries with forecasted debt >60% are flagged in red. Observe correlation between high debt and fiscal indicators.*")

st.subheader("📊 Fiscal Vulnerability vs Debt-to-GDP Forecast")
fig_vuln = px.scatter(df_plot, x='fiscal_vulnerability_index', y='debt_to_gdp_forecast',
                      size='population', color='high_debt_risk_prob', hover_name='Country',
                      color_continuous_scale='Reds',
                      labels={'fiscal_vulnerability_index': 'Fiscal Vulnerability Index',
                              'debt_to_gdp_forecast': 'Debt-to-GDP Forecast (%)'})
st.plotly_chart(fig_vuln, use_container_width=True)
st.markdown("*Insight: Countries with high fiscal vulnerability often have higher projected debt levels. Policy focus may be needed for these nations.*")

st.subheader("📈 Cumulative Deficit vs GDP Growth Rate")
fig_cum_gdp = px.scatter(df_plot, x='cumulative_deficit', y='gdp_growth_rate',
                         size='population', color='high_debt_risk_prob', hover_name='Country',
                         color_continuous_scale='Inferno',
                         labels={'cumulative_deficit': 'Cumulative Deficit',
                                 'gdp_growth_rate': 'GDP Growth Rate (%)'})
st.plotly_chart(fig_cum_gdp, use_container_width=True)
st.markdown("*Insight: High cumulative deficits can correlate with lower GDP growth. Countries with extreme deficits may require fiscal consolidation.*")


st.subheader("💸 Budget Deficit / Surplus Trend")
fig_deficit = px.line(df_plot, x='Year', y='budget_deficit/surplus', color='Country', markers=True)
fig_deficit.add_scatter(x=df_plot[df_plot['extreme_deficit_flag']]['Year'],
                        y=df_plot[df_plot['extreme_deficit_flag']]['budget_deficit/surplus'],
                        mode='markers', marker=dict(color='orange', size=10),
                        name='Extreme Deficit (<-5% GDP)')
st.plotly_chart(fig_deficit, use_container_width=True)
st.markdown("*Insight: Extreme deficits are highlighted in orange. These can indicate fiscal stress and debt accumulation.*")

st.subheader("💹 Revenue/Expenditure Ratio Trend")
fig_rev_exp_trend = px.line(df_plot, x='Year', y='rev_exp_ratio', color='Country', markers=True,
                            labels={'rev_exp_ratio': 'Revenue/Expenditure Ratio'})
st.plotly_chart(fig_rev_exp_trend, use_container_width=True)
st.markdown("*Insight: Ratios <1 indicate spending exceeds revenue. Monitoring this helps anticipate fiscal stress.*")


st.subheader("📈 Debt vs GDP Growth Rate")
fig_debt_gdp = px.scatter(df_plot, x='gdp_growth_rate', y='debt_to_gdp_forecast',
                          size='population', color='high_debt_risk_prob',
                          hover_name='Country', color_continuous_scale='Inferno',
                          labels={'gdp_growth_rate': 'GDP Growth Rate (%)',
                                  'debt_to_gdp_forecast': 'Debt-to-GDP Forecast (%)',
                                  'high_debt_risk_prob': 'High Debt Risk Probability'})
st.plotly_chart(fig_debt_gdp, use_container_width=True)
st.markdown("*Insight: Larger bubbles indicate higher populations; color intensity shows debt risk probability. High debt often associates with lower GDP growth.*")

st.subheader("🌍 Trade Balance vs Debt Sustainability")
df_plot['trade_balance_pct_gdp'] = df_plot['exports_gdp_ratio'] - df_plot['imports_gdp_ratio']
fig_trade = px.scatter(df_plot, x='exports_gdp_ratio', y='imports_gdp_ratio',
                       size='population', color='trade_balance_pct_gdp',
                       hover_name='Country', color_continuous_scale='Cividis',
                       labels={'exports_gdp_ratio': 'Exports % of GDP',
                               'imports_gdp_ratio': 'Imports % of GDP'})
st.plotly_chart(fig_trade, use_container_width=True)
st.markdown("*Insight: Negative trade balance (imports > exports) can contribute to higher debt pressure.*")

# ----------------



#------------

st.subheader("🏦 Fiscal KPIs Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Debt-to-GDP (%)", f"{df_plot['debt_to_gdp_forecast'].mean():.2f}")
col2.metric("High Debt Risk (%)", f"{df_plot['high_debt_risk_prob'].mean()*100:.2f}")
col3.metric("Avg Revenue/Exp Ratio", f"{df_plot['rev_exp_ratio'].mean():.2f}")
col4.metric("Cumulative Deficit", f"{df_plot['cumulative_deficit'].sum():,.0f}")

# -----------------------
# Policy Recommendations
# -----------------------
st.subheader("📝 Policy Recommendations")

def generate_dynamic_recommendations(row):
    recs = []
    if row['debt_to_gdp_forecast'] > 60 or row['high_debt_risk_prob'] > 0.6:
        recs.append("⚠️ Consider debt restructuring or fiscal consolidation to reduce debt-to-GDP.")
    if row['deficit_gdp_ratio'] < -5:
        recs.append("⚠️ Review expenditure and/or enhance revenue mobilization to control extreme deficit.")
    if 'interest_burden_ratio' in row and row['interest_burden_ratio'] < 1:
        recs.append("⚠️ Increase revenue or reduce debt servicing costs to improve interest coverage.")
    if 'cumulative_deficit' in row and row['cumulative_deficit'] > 0.3 * row.get('gdp_per_capita_calc', 1):
        recs.append("⚠️ Monitor cumulative deficit trends and adjust fiscal policies to prevent unsustainable growth.")
    if not recs:
        recs.append("✅ Fiscal conditions appear stable. Continue monitoring key indicators.")
    return recs

df_plot['recommendations'] = df_plot.apply(generate_dynamic_recommendations, axis=1)
latest_year = df_plot['Year'].max()
recommendation_df = df_plot[df_plot['Year'] == latest_year][
    ['Country', 'debt_to_gdp_forecast', 'high_debt_risk_prob', 'deficit_gdp_ratio', 'recommendations']
]

for i, row in recommendation_df.iterrows():
    st.markdown(f"**{row['Country']} (Year {latest_year}):**")
    for rec in row['recommendations']:
        st.markdown(f"- {rec}")
    st.markdown("---")

# -----------------------
# Raw Data Table
# -----------------------
st.subheader("📋 Raw Data Table")
st.dataframe(df_plot)


