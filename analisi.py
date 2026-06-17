
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Forex Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f6fa;
}

div[data-testid="metric-container"] {
    background-color: white;
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

h1 {
    font-size: 40px !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.title("📈 Forex Prediction Dashboard (LSTM & ARIMA)")

col1, col2 = st.columns(2)

with col1:
    pair = st.selectbox(
        "Currency Pair",
        ["EUR/USD"]
    )

with col2:
    timeframe = st.selectbox(
        "Timeframe",
        ["Monthly"]
    )

# =====================================
# UPLOAD FILE
# =====================================

uploaded_file = st.file_uploader(
    "Upload Historical EUR/USD CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # =====================================
    # LOAD DATA
    # =====================================

    df = pd.read_csv(uploaded_file)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    df.rename(
        columns={
            "price": "close",
            "higt": "high",
            "vol": "volume",
            "change %": "change_percent"
        },
        inplace=True
    )

    # =====================================
    # DATE
    # =====================================

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    # =====================================
    # CLOSE PRICE
    # =====================================

    df["close"] = (
        df["close"]
        .astype(str)
        .str.replace(",", "")
        .astype(float)
    )

    # =====================================
    # DUMMY MODEL
    # (sementara sebelum LSTM & ARIMA asli)
    # =====================================

    df["lstm_prediction"] = (
        df["close"]
        .rolling(window=3)
        .mean()
        .fillna(df["close"])
    )

    df["arima_prediction"] = (
        df["close"]
        .rolling(window=5)
        .mean()
        .fillna(df["close"])
    )

    # =====================================
    # METRICS LSTM
    # =====================================

    rmse_lstm = np.sqrt(
        mean_squared_error(
            df["close"],
            df["lstm_prediction"]
        )
    )

    mae_lstm = mean_absolute_error(
        df["close"],
        df["lstm_prediction"]
    )

    mape_lstm = (
        np.mean(
            np.abs(
                (df["close"] -
                 df["lstm_prediction"])
                / df["close"]
            )
        ) * 100
    )

    # =====================================
    # METRICS ARIMA
    # =====================================

    rmse_arima = np.sqrt(
        mean_squared_error(
            df["close"],
            df["arima_prediction"]
        )
    )

    mae_arima = mean_absolute_error(
        df["close"],
        df["arima_prediction"]
    )

    mape_arima = (
        np.mean(
            np.abs(
                (df["close"] -
                 df["arima_prediction"])
                / df["close"]
            )
        ) * 100
    )

    # =====================================
    # KPI CARDS
    # =====================================

    last_price = df["close"].iloc[-1]

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Last Price",
            f"{last_price:.4f}"
        )

    with k2:
        st.metric(
            "RMSE",
            f"{rmse_lstm:.5f}"
        )

    with k3:
        st.metric(
            "MAPE",
            f"{mape_lstm:.2f}%"
        )

    with k4:
        st.metric(
            "Latency",
            "120 ms"
        )

    # =====================================
    # GRAFIK
    # =====================================

    st.subheader("Actual Price vs Prediction")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["close"],
            mode="lines",
            name="Actual Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["lstm_prediction"],
            mode="lines",
            name="LSTM Prediction"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["arima_prediction"],
            mode="lines",
            name="ARIMA Prediction"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=550,
        autosize=True,
        legend=dict(
            orientation="h"
        )
    )

    st.plotly_chart(fig)

    # =====================================
    # EVALUASI LSTM
    # =====================================

    st.subheader("Model Evaluation - LSTM")

    l1, l2, l3 = st.columns(3)

    l1.metric(
        "RMSE (LSTM)",
        f"{rmse_lstm:.5f}"
    )

    l2.metric(
        "MAE (LSTM)",
        f"{mae_lstm:.5f}"
    )

    l3.metric(
        "MAPE (LSTM)",
        f"{mape_lstm:.2f}%"
    )

    # =====================================
    # EVALUASI ARIMA
    # =====================================

    st.subheader("Model Evaluation - ARIMA")

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "RMSE (ARIMA)",
        f"{rmse_arima:.5f}"
    )

    a2.metric(
        "MAE (ARIMA)",
        f"{mae_arima:.5f}"
    )

    a3.metric(
        "MAPE (ARIMA)",
        f"{mape_arima:.2f}%"
    )

    # =====================================
    # PERBANDINGAN MODEL
    # =====================================

    st.subheader("Perbandingan Model")

    comparison = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "MAPE"],
        "LSTM": [
            round(rmse_lstm, 5),
            round(mae_lstm, 5),
            round(mape_lstm, 2)
        ],
        "ARIMA": [
            round(rmse_arima, 5),
            round(mae_arima, 5),
            round(mape_arima, 2)
        ]
    })

    st.dataframe(comparison)

    # =====================================
    # DATASET
    # =====================================

    st.subheader("Dataset Preview")
    st.dataframe(df)

else:

    st.info(
        "Silakan upload file CSV EUR/USD terlebih dahulu."
    )
