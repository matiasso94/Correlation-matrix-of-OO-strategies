import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

# Streamlit Page Configuration
st.set_page_config(page_title="Trade Log Correlation Analysis", layout="wide")
st.title("Trade Log Correlation Analysis")

# Upload multiple trade logs
st.sidebar.header("Upload Trade Logs")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files", accept_multiple_files=True, type=["csv"])

@st.cache_data
def load_trade_log(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Ensure Date is in datetime format
    df['P/L %'] = df['P/L %'].fillna(0)  # Handle missing values in P/L %
    return df[['Date', 'P/L %']]

# Store data from uploaded files
trade_logs = {}
if uploaded_files:
    for file in uploaded_files:
        file_name = file.name  # Use exact file name
        trade_logs[file_name] = load_trade_log(file)

@st.cache_data
def fetch_spx_data(start_date, end_date, timeframe):
    interval = '1d' if timeframe == 'Daily' else '1wk' if timeframe == 'Weekly' else '1mo'
    spx = yf.download("^GSPC", start=start_date, end=end_date, interval=interval)
    spx = spx[['Adj Close']]
    spx['SPX P/L %'] = spx['Adj Close'].pct_change() * 100
    spx.reset_index(inplace=True)
    spx['SPX Net Liquidity'] = 100000 * (1 + spx['SPX P/L %'] / 100).cumprod()
    return spx[['Date', 'SPX P/L %', 'SPX Net Liquidity']]

# Timeframe Selection
timeframe = st.sidebar.radio("Select Timeframe", ['Daily', 'Weekly', 'Monthly'], index=0)

def get_period_cutoffs(df, timeframe):
    if timeframe == 'Weekly':
        df['Period'] = df['Date'].dt.to_period('W').apply(lambda x: x.start_time)
    elif timeframe == 'Monthly':
        df['Period'] = df['Date'].dt.to_period('M').apply(lambda x: x.start_time)
    return df

def process_data(df, timeframe):
    if timeframe in ['Weekly', 'Monthly']:
        df = get_period_cutoffs(df, timeframe)
        df = df.sort_values(by='Date')
        df = df.groupby('Period', as_index=False).agg({'P/L %': 'sum'})
        df.rename(columns={'Period': 'Date'}, inplace=True)
    return df.reset_index(drop=True)

# Process and merge trade logs
merged_df = pd.DataFrame()
processed_trade_logs = {}
for file_name, df in trade_logs.items():
    processed_trade_logs[file_name] = process_data(df, timeframe)
    processed_trade_logs[file_name].rename(columns={'P/L %': f"{file_name}_P/L"}, inplace=True)

for file_name, df in processed_trade_logs.items():
    df = df.loc[:, ~df.columns.duplicated()]
    if not merged_df.empty:
        merged_df = merged_df.merge(df, on='Date', how='left')
    else:
        merged_df = df.copy()

# Calculate Combined Portfolio Net Liquidity
if not merged_df.empty:
    portfolio_columns = [col for col in merged_df.columns if "_P/L" in col]
    merged_df['Combined Portfolio P/L %'] = merged_df[portfolio_columns].sum(axis=1)
    merged_df['Net Liquidity'] = (1 + merged_df['Combined Portfolio P/L %'] / 100).cumprod() * 100000

# Determine valid date range
valid_dates = [df['Date'].dropna().min() for df in processed_trade_logs.values() if not df.empty and 'Date' in df]
min_date = min(valid_dates) if valid_dates else None
valid_dates = [df['Date'].dropna().max() for df in processed_trade_logs.values() if not df.empty and 'Date' in df]
max_date = max(valid_dates) if valid_dates else None

if min_date and max_date:
    spx_data = fetch_spx_data(min_date, max_date, timeframe)
    merged_df = merged_df.merge(spx_data, on='Date', how='left').sort_values(by='Date')
    merged_df = merged_df.dropna(subset=['Net Liquidity', 'SPX Net Liquidity'])

    st.subheader("Processed Trade Log Data")
    st.dataframe(merged_df.reset_index(drop=True))

    correlation_matrix = merged_df.drop(columns=['Date', 'Net Liquidity', 'SPX Net Liquidity'], errors='ignore').corr()

    if correlation_matrix.shape[0] > 1:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig, clear_figure=True)
    else:
        st.warning("Not enough data for correlation matrix.")

    st.subheader("P/L % Over Time")
    columns_to_plot = [col for col in merged_df.columns if col not in ['Date', 'Net Liquidity', 'SPX Net Liquidity']]
    fig = px.line(merged_df, x='Date', y=columns_to_plot, title="Strategy Performance Over Time")
    st.plotly_chart(fig)

    # Plot Combined Portfolio vs SPX
    st.subheader("Combined Portfolio vs SPX Performance")
    fig = px.line(merged_df, x='Date', y=['Net Liquidity', 'SPX Net Liquidity'], title="Net Liquidity vs SPX",
                  labels={"Net Liquidity": "Portfolio", "SPX Net Liquidity": "SPX"},
                  color_discrete_map={"Net Liquidity": "gold", "SPX Net Liquidity": "cyan"})
    st.plotly_chart(fig)