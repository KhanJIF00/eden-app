import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def show_trends(df):
    st.markdown(
        "<h2 style='color:#00eaff; margin-bottom:0.5rem;'>📈 Trends Analysis</h2>"
        "<div style='color:#eaf6fb; font-size:1rem; margin-bottom:1rem;'>"
        "Explore trends and metrics in your water quality data. Choose between time-based and non-time-based analytics."
        "</div>",
        unsafe_allow_html=True
    )

    if df.empty:
        st.warning("No data available for trends analysis.")
        return

    # Detect date column
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    st.markdown("---")

    # --- Analysis Type Selection ---
    st.subheader("Choose Type of Analysis")
    analysis_type = st.radio("Select:", ["Non-time-based analytics", "Time-based analytics"])

    st.markdown("---")

    # --- Time-based Analysis ---
    if analysis_type == "Time-based analytics" and date_col:
        st.success(f"📅 Date column detected: `{date_col}`")
        features = [col for col in df.select_dtypes(include=["number", "bool"]).columns if col != date_col]
        if features:
            st.subheader("Select a Feature to Analyze Over Time")
            feature = st.selectbox("Choose Feature:", features)
            st.line_chart(df.set_index(date_col)[feature])
        else:
            st.warning("No numeric or boolean features available for time-based analysis.")

    else:
        if date_col:
            df = df.drop(columns=[date_col])
            st.info("🕒 Date column dropped for non-time-based analytics.")

        # --- Trend Analysis ---
        st.subheader("Select Trend Metrics")
        trend_options = ["Mean", "Median", "Standard Deviation", "Variance"]
        selected_trends = st.multiselect("Metrics:", trend_options)
        if not selected_trends:
            st.warning("Please select at least one trend analysis option.")
            return
        st.markdown("#### Summary Statistics:")
        for trend in selected_trends:
            if trend == "Mean":
                st.write("**Mean** of numeric columns:")
                st.dataframe(df.mean(numeric_only=True))
            elif trend == "Median":
                st.write("**Median** of numeric columns:")
                st.dataframe(df.median(numeric_only=True))
            elif trend == "Standard Deviation":
                st.write("**Standard Deviation** of numeric columns:")
                st.dataframe(df.std(numeric_only=True))
            elif trend == "Variance":
                st.write("**Variance** of numeric columns:")
                st.dataframe(df.var(numeric_only=True))

        st.markdown("---")

        # --- Visualizations ---
        st.subheader("Additional Visualizations")
        visualization_options = ["Histogram", "Correlation Matrix", "Scatter Plot"]
        selected_visualizations = st.multiselect("Choose visualizations to display:", visualization_options)
        for viz in selected_visualizations:
            if viz == "Histogram":
                st.write("Histogram of numeric columns:")
                for col in df.select_dtypes(include=[np.number]).columns:
                    st.subheader(f"Histogram of {col}")
                    st.bar_chart(df[col].value_counts())
            elif viz == "Correlation Matrix":
                st.write("Correlation Matrix Heatmap of numeric columns:")
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    corr = numeric_df.corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("No numeric columns available for correlation matrix.")
            elif viz == "Scatter Plot":
                st.write("Scatter Plot of numeric columns:")
                if len(df.select_dtypes(include=[np.number]).columns) >= 2:
                    st.scatter_chart(df.select_dtypes(include=[np.number]))
                else:
                    st.warning("Not enough numeric columns for scatter plot.")