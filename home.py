import streamlit as st
import pandas as pd
import numpy as np
import time

import trends_page
import predictions_page
import reports_page
import red_tide_forecast_page  # ✅ NEW IMPORT

# Page setup
st.set_page_config(
    page_title="EDEN AI by ALAREX",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Loading screen
if "loaded" not in st.session_state:
    st.session_state.loaded = False

    with st.spinner("Initializing EDEN AI for aquatic analysis..."):
        st.markdown("""
            <style>
                body, .stApp {
                    background-color: #0b0f1a;
                    color: #00ffcc;
                    font-family: 'Courier New', monospace;
                }
                .centered {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    overflow: hidden;
                }
                .title {
                    font-size: 72px;
                    font-weight: bold;
                    margin-bottom: 0;
                }
                .subtitle {
                    font-size: 20px;
                    margin-top: 0;
                    margin-bottom: 40px;
                    letter-spacing: 1.5px;
                }
                .wave {
                    position: absolute;
                    bottom: 0;
                    width: 100%;
                    height: 120px;
                    background: url("https://i.ibb.co/vvyRw7F/wave-light-teal2.gif") repeat-x;
                    background-size: cover;
                }
                @keyframes flip {
                    0% { transform: rotateY(0deg); }
                    50% { transform: rotateY(180deg); }
                    100% { transform: rotateY(360deg); }
                }
            </style>

            <div class="centered">
                <div class="title">EDEN AI</div>
                <div class="subtitle">AQUATIC INTELLIGENCE SYSTEM - BOOTING...</div>
            </div>
            <div class="wave"></div>
        """, unsafe_allow_html=True)
        time.sleep(5)

    st.session_state.loaded = True
    st.rerun()

# Main Title
st.title("🌊 EDEN AI by ALAREX")

def show_home():
    st.header("📁 Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="home_csv_uploader")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="latin1")

        rename_cols = {}
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["date", "year", "month", "day"]):
                rename_cols[col] = "date"
        df = df.rename(columns=rename_cols)

        for col in df.columns:
            if col == "date":
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

        df = df.select_dtypes(include=["number", "datetime", "bool"])

        st.session_state.df = df
        st.success("✅ File uploaded successfully!")

        st.markdown("#### 📊 Data Preview")
        st.dataframe(df, use_container_width=True, height=400)

        st.markdown("#### 🧮 Data Shape")
        st.write(f"{df.shape[0]} rows × {df.shape[1]} columns")

        st.markdown("#### ❓ Null Values")
        st.write(df.isnull().sum())

        st.markdown("#### 📈 Basic Statistics")
        st.write(df.describe())

        st.markdown("#### 🔬 Scatter Plot of Numerical Columns")
        st.scatter_chart(df.select_dtypes(include=[np.number]))

        if st.button("🧹 Clean Data"):
            df_cleaned = df.dropna().drop_duplicates()
            st.session_state.df = df_cleaned
            st.success("✅ Data cleaned successfully!")
            st.dataframe(df_cleaned, use_container_width=True, height=400)
            st.write(f"Cleaned Data Shape: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns")
            st.write("Cleaned Data Statistics:")
            st.write(df_cleaned.describe())

    else:
        st.warning("⚠️ Please upload a CSV file to proceed with the analysis.")

# Sidebar
st.sidebar.header("🧭 Navigation")
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "home"
if st.sidebar.button("📉 Trends Analysis"):
    st.session_state.page = "trends"
if st.sidebar.button("📊 Predictions"):
    st.session_state.page = "predictions"
if st.sidebar.button("📝 Reports"):
    st.session_state.page = "reports"
if st.sidebar.button("🌊 Red Tide Forecast"):  # ✅ NEW BUTTON
    st.session_state.page = "red_tide_forecast"

# Page Router
if st.session_state.page == "home":
    show_home()
elif st.session_state.page == "trends":
    if "df" in st.session_state:
        trends_page.show_trends(st.session_state.df)
    else:
        st.warning("⚠️ Please upload data on the Home page first.")
elif st.session_state.page == "predictions":
    if "df" in st.session_state:
        predictions_page.show_predictions(st.session_state.df)
    else:
        st.warning("⚠️ Please upload data on the Home page first.")
elif st.session_state.page == "reports":
    if "df" in st.session_state:
        reports_page.show_reports(st.session_state.df)
    else:
        st.warning("⚠️ Please upload data on the Home page first.")
elif st.session_state.page == "red_tide_forecast":  # ✅ NEW PAGE HANDLER
    red_tide_forecast_page.run_red_tide_forecast()
