import numpy as np
import pandas as pd
import streamlit as st
import warnings

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Red Tide Forecasting - EDEN AI")

st.title("🌊 Red Tide Prediction (1–2 Months Ahead)")
st.markdown("This model uses historical Karenia Brevis data to predict red tide likelihood using Facebook Prophet.")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_red_tide_forecast():
    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset with Karenia Brevis data on the Home page first.")
        return

    df = st.session_state.df.copy()

    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            st.error(f"Failed to convert dataset index to datetime: {e}")
            return

    # Find Karenia Brevis column (case insensitive)
    kb_cols = [col for col in df.columns if "karenia" in col.lower()]
    if not kb_cols:
        st.warning("⚠️ Your uploaded dataset does not contain any Karenia Brevis related columns required for red tide forecasting.")
        return

    kb_col = kb_cols[0]
    df_kb = df[[kb_col]].copy()  # Only keep Karenia Brevis column as DataFrame

    st.write(f"### Using column: `{kb_col}` for red tide forecasting")
    st.dataframe(df_kb.tail())

    split_date = '2022-01-01'
    df_train = df_kb.loc[df_kb.index <= split_date].copy()
    df_test = df_kb.loc[df_kb.index > split_date].copy()

    # Prepare data for Prophet with dynamic renaming
    date_col_train = df_train.index.name if df_train.index.name else df_train.reset_index().columns[0]
    date_col_test = df_test.index.name if df_test.index.name else df_test.reset_index().columns[0]

    df_train_prophet = df_train.reset_index().rename(columns={date_col_train: 'ds', kb_col: 'y'})

    model = Prophet()
    model.fit(df_train_prophet)

    if df_test.empty:
        st.info(f"⚠️ No test data available after split date ({split_date}). Generating 60 days future forecast.")
        future = model.make_future_dataframe(periods=60)
        df_forecast = model.predict(future)
        plot_df = df_forecast
        actual_df = df_train_prophet  # Only training data actuals available
    else:
        df_test_prophet = df_test.reset_index().rename(columns={date_col_test: 'ds', kb_col: 'y'})
        df_forecast = model.predict(df_test_prophet)
        plot_df = df_forecast
        actual_df = df_test_prophet

    # Interactive Forecast vs Actual Plot
    st.write("### Red Tide Forecast vs Actual (Interactive)")
    fig = go.Figure()

    # Plot actual data points
    fig.add_trace(go.Scatter(
        x=actual_df['ds'],
        y=actual_df['y'],
        mode='markers',
        name='Actual',
        marker=dict(color='red')
    ))

    # Plot forecast line
    fig.add_trace(go.Scatter(
        x=plot_df['ds'],
        y=plot_df['yhat'],
        mode='lines',
        name='Forecast'
    ))

    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=plot_df['ds'],
        y=plot_df['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=plot_df['ds'],
        y=plot_df['yhat_lower'],
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    fig.update_layout(
        title='Red Tide Forecast vs Actual',
        xaxis_title='Date',
        yaxis_title='Karenia Brevis Cells per Liter',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Components
    st.write("### Forecast Components (Trend, Weekly, Yearly Patterns)")
    fig_trend = px.line(plot_df, x='ds', y='trend', title='Trend Component')
    st.plotly_chart(fig_trend, use_container_width=True)

    if 'weekly' in plot_df.columns:
        fig_weekly = px.line(plot_df, x='ds', y='weekly', title='Weekly Seasonality')
        st.plotly_chart(fig_weekly, use_container_width=True)

    if 'yearly' in plot_df.columns:
        fig_yearly = px.line(plot_df, x='ds', y='yearly', title='Yearly Seasonality')
        st.plotly_chart(fig_yearly, use_container_width=True)

    # Evaluation metrics only if test data is available
    if not df_test.empty:
        y_true = actual_df['y'].values
        y_pred = plot_df['yhat'].values[:len(y_true)]

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        st.write("### Model Evaluation Metrics")
        st.metric("Mean Absolute Error", f"{mae:,.2f}")
        st.metric("Mean Squared Error", f"{mse:,.2f}")
        st.metric("MAPE (%)", f"{mape:.2f}%")
    else:
        st.info("ℹ️ Evaluation metrics are not available because there is no test data after the split date.")

