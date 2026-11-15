import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Weather Data EDA Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("weather_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Province/State'] = df['Province/State'].fillna('Unknown')
    numeric_cols = ['temp', 'min', 'max', 'rh', 'wdsp', 'prcp', 'dewp', 'stp']
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df

df = load_data()

# Title & Description
st.title("Historical Weather Data Analysis Dashboard")
st.write("Explore temperature, humidity, wind speed, and rainfall trends across countries and dates.")

# Sidebar Filters
st.sidebar.header("Filter Weather Data")
countries = ["All"] + sorted(df['Country/Region'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Select Country/Region", countries)

min_date = df['Date'].min().date()
max_date = df['Date'].max().date()
date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Apply Filters
filtered_df = df.copy()
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['Country/Region'] == selected_country]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

# 4 Key Numbers on Top
col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Temperature", f"{filtered_df['temp'].mean():.1f}")
col2.metric("Highest Temperature", f"{filtered_df['max'].max():.1f}")
col3.metric("Average Humidity (%)", f"{filtered_df['rh'].mean():.1f}%")
col4.metric("Total Records / Days", f"{len(filtered_df):,}")

st.divider()

# Row 1: Univariate & Line Chart Over Time
c1, c2 = st.columns(2)

with c1:
    st.subheader("1. Temperature Distribution (Univariate)")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(filtered_df['temp'], bins=25, kde=True, color='steelblue', ax=ax1)
    ax1.set_title("Distribution of Temperature")
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

with c2:
    st.subheader("2. Temperature Trend Over Time")
    daily_temp = filtered_df.groupby('Date')['temp'].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(daily_temp['Date'], daily_temp['temp'], color='darkorange')
    ax2.set_title("Average Temperature Over Time")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Temperature")
    plt.xticks(rotation=30)
    st.pyplot(fig2)

# Row 2: Bivariate Scatter & Correlation Heatmap
c3, c4 = st.columns(2)

with c3:
    st.subheader("3. Temperature vs Humidity (Bivariate)")
    sample_df = filtered_df.sample(min(2000, len(filtered_df)), random_state=42) if len(filtered_df) > 0 else filtered_df
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=sample_df, x='temp', y='rh', alpha=0.4, color='teal', ax=ax3)
    ax3.set_title("Temperature vs Relative Humidity")
    ax3.set_xlabel("Temperature")
    ax3.set_ylabel("Relative Humidity (%)")
    st.pyplot(fig3)

with c4:
    st.subheader("4. Weather Correlation Heatmap")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    corr = filtered_df[['temp', 'min', 'max', 'rh', 'wdsp', 'prcp', 'dewp', 'stp']].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    ax4.set_title("Correlation Heatmap")
    st.pyplot(fig4)

# Row 3: 5th Graph (Wind Speed Distribution / Top Countries)
st.subheader("5. Wind Speed Distribution Across Selected Data")
fig5, ax5 = plt.subplots(figsize=(10, 3.5))
sns.boxplot(x=filtered_df['wdsp'], color='mediumpurple', ax=ax5)
ax5.set_title("Wind Speed Spread and Storm Outliers")
ax5.set_xlabel("Wind Speed (wdsp)")
st.pyplot(fig5)

# Raw Data Checkbox
if st.checkbox("Show Raw Weather Data Table"):
    st.dataframe(filtered_df.head(100))

# Main Findings Section
st.subheader("Main Findings")
st.markdown("""
- **Temperature & Humidity Relationship:** Higher temperatures generally correspond with lower relative humidity (negative correlation).
- **Seasonal Warming Trend:** Average temperatures show a steady upward shift across the recorded timeline.
- **Extreme Weather Outliers:** Wind speed and precipitation are heavily right-skewed due to occasional storms and heavy rainfall days.
""")
