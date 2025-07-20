import streamlit as st
from fpdf import FPDF
import tempfile
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def show_reports(df):
    st.markdown(
        "<h2 style='color:#00eaff; margin-bottom:0.5rem;'>📑 Reports</h2>"
        "<div style='color:#eaf6fb; font-size:1rem; margin-bottom:1rem;'>"
        "Review and export your analysis results, graphs, and key data insights."
        "</div>",
        unsafe_allow_html=True
    )

    st.subheader("1. Choose Report Type")
    report_type = st.radio("Select the type of report you want to generate:", ["Visual Report", "Summary Report of Trends", "Full Report"])

    st.subheader("2. Dataset Overview")
    st.markdown(f"**🧮 Shape:** `{df.shape[0]} rows × {df.shape[1]} columns`")

    st.markdown("**📌 Column Information:**")
    col_info = pd.DataFrame({
        "Type": df.dtypes,
        "Nulls": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2),
        "Unique": df.nunique()
    })
    st.dataframe(col_info)

    st.markdown("**📊 Descriptive Statistics:**")
    st.dataframe(df.describe(include="all").transpose())

    st.subheader("3. Select Graphs to Display")
    available_graphs = {
        "Scatter Plot": "scatter",
        "Histogram": "histogram",
        "Correlation Matrix": "correlation",
        "Line Chart": "line"
    }
    selected_graphs = st.multiselect("Choose graphs to include in your report:", list(available_graphs.keys()))

    graph_images = []
    st.subheader("4. Graph Visualizations")
    for graph in selected_graphs:
        st.markdown(f"**{graph}**")
        if available_graphs[graph] == "scatter":
            if len(df.select_dtypes(include=["number"]).columns) >= 2:
                chart = df.select_dtypes(include=["number"])
                st.scatter_chart(chart)
                fig = pd.plotting.scatter_matrix(chart, figsize=(8, 6))
                for ax in fig.ravel():
                    ax.set_xlabel(ax.get_xlabel(), fontsize=8, rotation=45)
                    ax.set_ylabel(ax.get_ylabel(), fontsize=8)
                tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig[0][0].get_figure().savefig(tmpfile.name)
                graph_images.append(tmpfile.name)
                plt.close('all')
        elif available_graphs[graph] == "histogram":
            for col in df.select_dtypes(include=["number"]).columns:
                fig, ax = plt.subplots()
                df[col].hist(ax=ax, bins=20)
                ax.set_title(f"Histogram of {col}")
                st.pyplot(fig)
                tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(tmpfile.name)
                graph_images.append(tmpfile.name)
                plt.close(fig)
        elif available_graphs[graph] == "correlation":
            numeric_df = df.select_dtypes(include=["number"])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fig.savefig(tmpfile.name)
                graph_images.append(tmpfile.name)
                plt.close(fig)
        elif available_graphs[graph] == "line":
            if "date" in df.columns:
                features = [col for col in df.select_dtypes(include=["number", "bool"]).columns if col != "date"]
                if features:
                    feature = st.selectbox("Select feature for line chart:", features, key="report_line_feature")
                    st.line_chart(df.set_index("date")[feature])
            else:
                st.line_chart(df.select_dtypes(include=["number"]))

    st.markdown("---")

    # --- Generate PDF ---
    if st.button("Generate PDF Report"):
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.set_text_color(0, 234, 255)
        pdf.cell(0, 10, "EDEN AI Data Report", ln=True, align="C")
        pdf.ln(10)

        # Section: Dataset Shape
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 10, f"Dataset Shape:\n{df.shape[0]} rows × {df.shape[1]} columns\n")

        # Section: Column Info
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 10, "Column Information", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(0, 0, 0)
        col_info_text = col_info.astype(str).to_string()
        pdf.multi_cell(0, 6, col_info_text)
        pdf.ln(5)

        # Section: Stats
        if report_type in ["Summary Report of Trends", "Full Report"]:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 10, "Descriptive Statistics", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.set_text_color(0, 0, 0)
            desc_stats = df.describe().round(2).astype(str).to_string()
            pdf.multi_cell(0, 6, desc_stats)
            pdf.ln(5)

        # Section: Graphs
        if report_type in ["Visual Report", "Full Report"] and graph_images:
            pdf.set_font("Arial", 'B', 12)
            pdf.set_text_color(0, 102, 204)
            pdf.cell(0, 10, "Graphs", ln=True)
            for img_path in graph_images:
                pdf.image(img_path, w=170)
                pdf.ln(5)

        # Save and Provide Download Link
        pdf_output = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.output(pdf_output.name)

        with open(pdf_output.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="EDEN_AI_Report.pdf">📥 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.info("You can also export this page as PDF using your browser's print option (Ctrl+P or Cmd+P).")
