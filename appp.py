import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta
import openai
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.utils.data as torch_data
import os
from scipy.stats import zscore
from agents import formulate_questions, web_research_agent, institutional_knowledge_agent, consolidate_reports, management_discussion, buy_sell_decision

st.cache_data = True

# Set Streamlit Page Configurations
st.set_page_config(page_title="Trading Learning through Anomalies", layout="wide")

st.title("📈 Anomaly detection Algorithm")

# Define stock tickers
stocks = ["AMZN", "AAPL", "WMT", "MSFT", "TSLA", "WFC", "NVDA"]
game_stocks = ["AXP", "T", "BAC", "BA"]

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="5y")
    return df.reset_index()

def analyze_stock(df, selected_ticker):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = df['Close'].astype(float)
    lookback = 5
    timeseries = df[['Close']].values.astype('float32')
    X, y = [], []
    for i in range(len(timeseries)-lookback):
        X.append(timeseries[i:i+lookback])
        y.append(timeseries[i+1:i+lookback+1])
    X, y = torch.tensor(X), torch.tensor(y)

    class StockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
            self.linear = nn.Linear(50, 1)
        def forward(self, x):
            x, _ = self.lstm(x)
            return self.linear(x)

    model = StockModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = torch_data.DataLoader(torch_data.TensorDataset(X, y), shuffle=True, batch_size=8)
    for epoch in range(50):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        pred_series = torch.full_like(torch.tensor(timeseries), float('nan'))
        pred_series[lookback:] = model(X)[:, -1, :]
    error = abs(df['Close'].values - pred_series.numpy().flatten())
    anomalies = df.loc[error > 6, ['Date', 'Close']]
    return anomalies

def detect_anomalies(data, threshold=2):
    data['Returns'] = data['Close'].pct_change()
    data['Z-Score'] = zscore(data['Returns'], nan_policy='omit')
    return data[abs(data['Z-Score']) > threshold][['Date', 'Close']].reset_index(drop=True)

# Layout with sidebar selection
col1, col2 = st.columns([1, 2])
with col1:
    selected_ticker = st.selectbox("**Select Stock**", stocks)
    algos = ["Standard Scalar - Z-Score", "Deep Learning - LSTM"]
    selected_algo = st.selectbox("**Select Anomaly Detection Algorithm**", algos)
    if st.button("Analyze Stock"):
        df = fetch_stock_data(selected_ticker)
        st.session_state['df'] = df
        st.session_state['anomalies'] = detect_anomalies(df) if selected_algo == "Z-Score" else analyze_stock(df, selected_ticker)

    if 'anomalies' in st.session_state and not st.session_state['anomalies'].empty:
        st.session_state['selected_anomaly'] = st.selectbox("**Select an anomaly date**", st.session_state['anomalies']['Date'].astype(str))
        if st.button("Analyze Anomaly"):
            st.session_state['question'] = formulate_questions(st.session_state['selected_anomaly'], selected_ticker)
            st.session_state['web_response'] = web_research_agent(st.session_state['question'])
            st.session_state['institutional_response'] = institutional_knowledge_agent(st.session_state['question'])
            st.session_state['consolidated_report'] = consolidate_reports(st.session_state['web_response'], st.session_state['institutional_response'])
            st.session_state['management_response'] = management_discussion(st.session_state['consolidated_report'])

with col2:
    if 'df' in st.session_state and 'anomalies' in st.session_state:
        st.subheader("📊 Stock Price Trend & Anomalies")
        trace1 = go.Scatter(x=st.session_state['df']['Date'], y=st.session_state['df']['Close'], mode='lines', name='Closing Price')
        trace2 = go.Scatter(x=st.session_state['anomalies']['Date'], y=st.session_state['anomalies']['Close'], mode='markers', name='Anomaly', marker=dict(color='red', size=8))
        fig = go.Figure(data=[trace1, trace2])
        fig.update_layout(title=f"Stock Price Trend for {selected_ticker}", xaxis_title="Date", yaxis_title="Price", hovermode='closest')
        st.plotly_chart(fig)

    if 'management_response' in st.session_state:
        st.markdown("---")
        st.subheader("🧐 AI-Powered Anomaly Analysis")
        st.markdown("### 🔍 Question to Investigate the Anomaly")
        st.info(st.session_state['question'])
        st.markdown("### 🌍 Web Research Findings")
        st.success(st.session_state['web_response'])
        st.markdown("### 🏛️ Institutional Knowledge")
        st.success(st.session_state['institutional_response'])
        st.markdown("### 📜 Consolidated Report")
        st.warning(st.session_state['consolidated_report'])
        st.markdown("### 🏢 Management Discussion")
        st.error(st.session_state['management_response'])

# 🎮 Trading Game Section
st.markdown("---")
st.subheader("🎮GameZone!")

if 'game_started' not in st.session_state:
    st.session_state['game_started'] = False
if 'decision_made' not in st.session_state:
    st.session_state['decision_made'] = False

if st.button("Start Game"):
    st.session_state['game_started'] = True
    st.session_state['decision_made'] = False
    game_ticker = random.choice(game_stocks)
    st.session_state['game_ticker'] = game_ticker
    game_df = fetch_stock_data(game_ticker)
    st.session_state['game_df1'] = game_df
    game_anomalies = detect_anomalies(game_df)
    
    if not game_anomalies.empty:
        selected_anomaly = game_anomalies.sample(1)
        anomaly_date = selected_anomaly['Date'].values[0]
        st.session_state['game_df'] = game_df[game_df['Date'].dt.tz_localize(None) <= anomaly_date]
        st.session_state['anomaly_date'] = anomaly_date
        question = f"On {anomaly_date}, for the {game_ticker} stock price there was an unusual fluctuation. Can you investigate possible reasons for this anomaly?"

        st.session_state['web_response'] = web_research_agent(question)
        st.session_state['game_ticker'] = game_ticker
        
if st.session_state['game_started']:
    st.success(f"Random Stock Selected: {st.session_state['game_ticker']}. Anomaly Detected on {st.session_state['anomaly_date']}!")
    st.plotly_chart(go.Figure(data=[go.Scatter(x=st.session_state['game_df']['Date'], y=st.session_state['game_df']['Close'], mode='lines')]))
    st.success(st.session_state['web_response'])
    if not st.session_state['decision_made']:
        decision = st.radio("What would you do?", ["Sell", "Buy"], key='decision')
        if st.button("Submit Decision"):
            st.session_state['decision_made'] = True
            # Get the full AI response, then extract the first word as the correct decision.
            full_decision = buy_sell_decision(st.session_state['web_response'])
            correct_decision = full_decision.split()[0].strip().upper()
            st.session_state['correct_decision'] = correct_decision
            st.session_state['user_decision'] = decision.strip().upper()
            game_df = fetch_stock_data(st.session_state['game_ticker'])
            # subset the game_df, add 5 days to the anomaly date
            anomaly_date = st.session_state['anomaly_date']
            game_df_ana = game_df[game_df['Date'].dt.tz_localize(None) <= anomaly_date]
            extra_points = min(30, len(game_df_ana))  # Ensure it doesn't exceed available data
            game_df_extended = game_df.iloc[:((len(game_df_ana) + extra_points))]
            # Create the figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=game_df_extended['Date'], 
                y=game_df_extended['Close'], 
                mode='lines',
                name="Stock Price"
            ))
            if len(game_df_ana) < len(game_df_extended):  # Ensure index is within range
                vline_date = game_df_extended['Date'].iloc[len(game_df_ana)]
                fig.add_vline(x=vline_date, line=dict(color="red", width=2, dash="dot"), name="Anomaly Marker")
            fig.update_layout(
                title="Stock Price with Anomaly Marker",
                xaxis_title="Date",
                yaxis_title="Close Price"
            )
            st.plotly_chart(fig)

    if st.session_state['decision_made']:
        if st.session_state['user_decision'] == st.session_state['correct_decision']:
            st.success(f"❌ Incorrect Decision! 😞\n")
        else:
            st.error(f"✅ Correct Decision! 🎉\n")
    #st.session_state['game_df'] = game_df[game_df['Date'].dt.tz_localize(None) <= anomaly_date + timedelta(days=7)]
