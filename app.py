# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# -----------------------
# Load Cleaned Data
# -----------------------
pivot_data = pd.read_csv("data/africa_sovereign_debt_cleaned.csv")

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

# Scenario sliders (global)
st.sidebar.subheader("Global Scenario Analysis")
gdp_growth_factor = st.sidebar.slider("GDP Growth Shock Factor", 0.8, 1.2, 1.0, 0.01)
rev_exp_factor = st.sidebar.slider("Revenue/Expenditure Adjustment Factor", 0.8, 1.2, 1.0, 0.01)

# -----------------------
# Filter Data for Dashboard
# -----------------------
df = pivot_data[(pivot_data['Country'].isin(selected_countries)) &
                (pivot_data['Year'] >= year_range[0]) &
                (pivot_data['Year'] <= year_range[1])].copy()

# -----------------------
# Prepare Modeling DataFrames
# -----------------------
df_model = pivot_data.copy()  # Regression
df_cls_model = pivot_data.copy()  # Classification

# -----------------------
# Create Country_Code_Num for all
# -----------------------
for d in [df, df_model, df_cls_model]:
    d['Country_Code_Num'] = d['Country'].astype('category').cat.codes

# -----------------------
# Feature columns
# -----------------------
feature_cols = [
    'rev_exp_ratio', 'deficit_gdp_ratio', 'cumulative_deficit', 'deficit_volatility_3yr',
    'gdp_growth_rate', 'gdp_per_capita_calc', 'gdp_real_nominal_gap_pct',
    'trade_balance_pct_gdp', 'gdp_volatility_5yr',
    'debt_accumulation_rate', 'debt_carrying_capacity', 'fiscal_vulnerability_index',
    'interest_burden_ratio', 'Country_Code_Num'
]

# -----------------------
# Fill missing values in features
# -----------------------
df_model[feature_cols] = df_model[feature_cols].fillna(df_model[feature_cols].median())
df_cls_model[feature_cols] = df_cls_model[feature_cols].fillna(df_cls_model[feature_cols].median())
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Also fill numeric columns used in plots
numeric_plot_cols = [
    'population', 'revenue', 'expenditure', 'gdp_growth_rate', 'unemployment_rate',
    'exports_gdp_ratio', 'imports_gdp_ratio', 'cumulative_deficit'
]
for col in numeric_plot_cols:
    df[col] = df[col].fillna(df[col].median())

# -----------------------
# Drop rows where target is NaN
# -----------------------
df_model = df_model.dropna(subset=['debt_to_gdp'])
df_cls_model = df_cls_model.dropna(subset=['high_debt_risk'])

# -----------------------
# Apply global scenario adjustments
# -----------------------
df['gdp_growth_rate'] *= gdp_growth_factor
df['rev_exp_ratio'] *= rev_exp_factor

# -----------------------
# Regression Model: Debt-to-GDP Forecast
# -----------------------
train_mask = df_model['Year'] <= 2020
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_reg.fit(df_model[feature_cols][train_mask], df_model['debt_to_gdp'][train_mask])
df['debt_to_gdp_forecast'] = rf_reg.predict(df[feature_cols])

# -----------------------
# Classification Model: High Debt Risk
# -----------------------
rf_cls = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
rf_cls.fit(df_cls_model[feature_cols], df_cls_model['high_debt_risk'])
df['high_debt_risk_prob'] = rf_cls.predict_proba(df[feature_cols])[:, 1]

# -----------------------
# Flags for extreme cases
# -----------------------
df['extreme_deficit_flag'] = df['deficit_gdp_ratio'] < -5
df['high_debt_flag'] = df['debt_to_gdp_forecast'] > 60

# -----------------------
# Remove NaNs in predictions
# -----------------------
df_plot = df.dropna(subset=['debt_to_gdp_forecast', 'high_debt_risk_prob'])

# -----------------------
# Feature Importance Plots
# -----------------------
st.subheader("Feature Importance: Debt-to-GDP Forecast")
reg_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_reg.feature_importances_}).sort_values(
    by='Importance', ascending=False)
fig_reg_feat = px.bar(reg_importance, x='Importance', y='Feature', orientation='h')
st.plotly_chart(fig_reg_feat, use_container_width=True)

st.subheader("Feature Importance: High Debt Risk")
cls_importance = pd.DataFrame({'Feature': feature_cols, 'Importance': rf_cls.feature_importances_}).sort_values(
    by='Importance', ascending=False)
fig_cls_feat = px.bar(cls_importance, x='Importance', y='Feature', orientation='h')
st.plotly_chart(fig_cls_feat, use_container_width=True)

# -----------------------
# Dynamic Scenario Adjustments per Country
# -----------------------
st.sidebar.subheader("Per-Country Scenario Adjustments")
country_scenarios = {}
for country in selected_countries:
    gdp_factor = st.sidebar.slider(f"{country} GDP Growth Factor", 0.8, 1.2, 1.0, 0.01)
    rev_exp_factor_c = st.sidebar.slider(f"{country} Rev/Exp Factor", 0.8, 1.2, 1.0, 0.01)
    country_scenarios[country] = (gdp_factor, rev_exp_factor_c)

df_scenario = df_plot.copy()
for country, (gdp_f, rev_f) in country_scenarios.items():
    mask = df_scenario['Country'] == country
    df_scenario.loc[mask, 'gdp_growth_rate'] *= gdp_f
    df_scenario.loc[mask, 'rev_exp_ratio'] *= rev_f

df_scenario['debt_to_gdp_forecast'] = rf_reg.predict(df_scenario[feature_cols])
df_scenario['high_debt_risk_prob'] = rf_cls.predict_proba(df_scenario[feature_cols])[:, 1]

# -----------------------
# 1️⃣ Debt-to-GDP Trend
# -----------------------
st.subheader("Debt-to-GDP Trend")
fig_debt_flag = px.line(df_scenario, x='Year', y='debt_to_gdp_forecast', color='Country', markers=True)
fig_debt_flag.add_scatter(
    x=df_scenario[df_scenario['high_debt_flag']]['Year'],
    y=df_scenario[df_scenario['high_debt_flag']]['debt_to_gdp_forecast'],
    mode='markers',
    marker=dict(color='red', size=10),
    name='High Debt (>60%)'
)
st.plotly_chart(fig_debt_flag, use_container_width=True)

# -----------------------
# 2️⃣ Budget Deficit Trend
# -----------------------
st.subheader("Budget Deficit / Surplus Trend")
fig_deficit_flag = px.line(df_scenario, x='Year', y='budget_deficit/surplus', color='Country', markers=True)
fig_deficit_flag.add_scatter(
    x=df_scenario[df_scenario['extreme_deficit_flag']]['Year'],
    y=df_scenario[df_scenario['extreme_deficit_flag']]['budget_deficit/surplus'],
    mode='markers',
    marker=dict(color='orange', size=10),
    name='Extreme Deficit (<-5% GDP)'
)
st.plotly_chart(fig_deficit_flag, use_container_width=True)

# -----------------------
# 3️⃣ High Debt Risk Probability
# -----------------------
st.subheader("High Debt Risk Probability (Debt-to-GDP > 60%)")
fig_risk = px.line(df_scenario, x='Year', y='high_debt_risk_prob', color='Country', markers=True)
st.plotly_chart(fig_risk, use_container_width=True)

# -----------------------
# 4️⃣ Debt Sustainability Heatmap
# -----------------------
st.subheader("Debt-to-GDP Heatmap")
heatmap_data = df_scenario.pivot_table(index='Country', columns='Year', values='debt_to_gdp_forecast')
fig_heatmap = px.imshow(heatmap_data, text_auto=True, aspect="auto", color_continuous_scale='Reds',
                        labels=dict(x="Year", y="Country", color="Debt-to-GDP Forecast"))
st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------
# 5️⃣ Revenue vs Expenditure Scatter
# -----------------------
st.subheader("Revenue vs Expenditure")
fig_rev_exp = px.scatter(
    df_scenario, x='revenue', y='expenditure',
    size='population',  # safe now
    color='high_debt_risk_prob',
    hover_name='Country', color_continuous_scale='Viridis',
    labels={'revenue': 'Revenue', 'expenditure': 'Expenditure', 'high_debt_risk_prob': 'High Debt Risk Prob'}
)
st.plotly_chart(fig_rev_exp, use_container_width=True)

# -----------------------
# 6️⃣ GDP Growth vs Unemployment
# -----------------------
st.subheader("GDP Growth vs Unemployment")
fig_gdp_unemp = px.scatter(
    df_scenario, x='gdp_growth_rate', y='unemployment_rate',
    color='Country', size='population', hover_name='Country',
    labels={'gdp_growth_rate': 'GDP Growth Rate', 'unemployment_rate': 'Unemployment Rate'}
)
st.plotly_chart(fig_gdp_unemp, use_container_width=True)

# -----------------------
# 7️⃣ Exports & Imports vs Trade Balance
# -----------------------
st.subheader("Trade Balance Analysis")
df_scenario['trade_balance_pct_gdp'] = df_scenario['exports_gdp_ratio'] - df_scenario['imports_gdp_ratio']
fig_trade = px.scatter(
    df_scenario, x='exports_gdp_ratio', y='imports_gdp_ratio',
    size='population',  # safe now
    color='trade_balance_pct_gdp',
    hover_name='Country', color_continuous_scale='Cividis',
    labels={'exports_gdp_ratio': 'Exports % of GDP', 'imports_gdp_ratio': 'Imports % of GDP'}
)
st.plotly_chart(fig_trade, use_container_width=True)

# -----------------------
# 8️⃣ Cumulative Deficit Trend
# -----------------------
st.subheader("Cumulative Deficit Over Time")
fig_cum_deficit = px.line(df_scenario, x='Year', y='cumulative_deficit', color='Country', markers=True)
st.plotly_chart(fig_cum_deficit, use_container_width=True)

# -----------------------
# 9️⃣ Summary KPIs
# -----------------------
st.subheader("Summary KPIs")
total_countries = df_scenario['Country'].nunique()
avg_debt = df_scenario['debt_to_gdp_forecast'].mean()
high_risk_pct = df_scenario['high_debt_risk_prob'].mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Countries Selected", total_countries)
col2.metric("Average Forecast Debt-to-GDP (%)", f"{avg_debt:.2f}")
col3.metric("Average High Debt Risk (%)", f"{high_risk_pct:.2f}")

# -----------------------
# 10️⃣ Dynamic Policy Recommendation Panel
# -----------------------
st.subheader("Policy Recommendations")


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


df_scenario['recommendations'] = df_scenario.apply(generate_dynamic_recommendations, axis=1)

# Show recommendations for the latest year
latest_year = df_scenario['Year'].max()
recommendation_df = df_scenario[df_scenario['Year'] == latest_year][
    ['Country', 'debt_to_gdp_forecast', 'high_debt_risk_prob', 'deficit_gdp_ratio', 'recommendations']
]

for i, row in recommendation_df.iterrows():
    st.markdown(f"**{row['Country']} (Year {latest_year}):**")
    for rec in row['recommendations']:
        st.markdown(f"- {rec}")
    st.markdown("---")

# -----------------------
# 11️⃣ Raw Data Table
# -----------------------
st.subheader("Raw Data Table")
st.dataframe(df_scenario)
