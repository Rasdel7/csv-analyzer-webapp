import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CSV Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Data Analyzer")
st.markdown("Upload any CSV file and get instant insights, "
            "charts and statistics.")
st.markdown("---")


uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    
    df = pd.read_csv(uploaded_file)

    st.success(f"File loaded! {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.markdown("---")

    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Overview", "📈 Visualize", "🔗 Correlations", "🧹 Data Quality"
    ])

 
    with tab1:
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows",        f"{df.shape[0]:,}")
        col2.metric("Columns",     f"{df.shape[1]}")
        col3.metric("Missing",     f"{df.isnull().sum().sum():,}")
        col4.metric("Duplicates",  f"{df.duplicated().sum():,}")

        st.markdown("### Column Information")
        col_info = pd.DataFrame({
            'Column':   df.columns,
            'Type':     df.dtypes.values,
            'Non-Null': df.count().values,
            'Missing':  df.isnull().sum().values,
            'Unique':   df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)

        st.markdown("### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        st.markdown("### Create Your Own Chart")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_cols     = df.columns.tolist()

        chart_type = st.selectbox(
            "Chart type:",
            ["Histogram", "Bar Chart", "Scatter Plot",
             "Line Chart", "Box Plot"]
        )

        if chart_type == "Histogram":
            col = st.selectbox("Select column:", numeric_cols)
            bins = st.slider("Number of bins:", 10, 100, 30)
            if col:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(df[col].dropna(), bins=bins,
                        color='#3498db', edgecolor='black')
                ax.set_title(f'Distribution of {col}', fontsize=14)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.tight_layout()
                st.pyplot(fig)

        elif chart_type == "Bar Chart":
            col = st.selectbox("Select column:", all_cols)
            top_n = st.slider("Show top N values:", 5, 30, 10)
            if col:
                fig, ax = plt.subplots(figsize=(10, 5))
                val_counts = df[col].value_counts().head(top_n)
                colors = sns.color_palette('Blues_r', len(val_counts))
                ax.bar(val_counts.index.astype(str),
                       val_counts.values, color=colors,
                       edgecolor='black')
                ax.set_title(f'Top {top_n} values in {col}', fontsize=14)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)

        elif chart_type == "Scatter Plot":
            col_x = st.selectbox("X axis:", numeric_cols, index=0)
            col_y = st.selectbox("Y axis:", numeric_cols,
                                  index=min(1, len(numeric_cols)-1))
            if col_x and col_y:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df[col_x], df[col_y],
                           alpha=0.4, color='#9b59b6', s=15)
                ax.set_title(f'{col_x} vs {col_y}', fontsize=14)
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

        elif chart_type == "Line Chart":
            col = st.selectbox("Select column:", numeric_cols)
            if col:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df[col].values, color='#2ecc71', linewidth=1.5)
                ax.set_title(f'{col} over index', fontsize=14)
                ax.set_xlabel('Index')
                ax.set_ylabel(col)
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

        elif chart_type == "Box Plot":
            col = st.selectbox("Select column:", numeric_cols)
            if col:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.boxplot(df[col].dropna(), patch_artist=True,
                           boxprops=dict(facecolor='#3498db', alpha=0.7))
                ax.set_title(f'Box Plot — {col}', fontsize=14)
                ax.set_ylabel(col)
                plt.tight_layout()
                st.pyplot(fig)

   
    with tab3:
        st.markdown("### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation.")
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            corr = numeric_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True,
                        fmt='.2f', cmap='coolwarm',
                        center=0, square=True,
                        linewidths=0.5, ax=ax)
            ax.set_title('Feature Correlation Heatmap', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### Top Correlated Pairs")
            corr_pairs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    corr_pairs.append({
                        'Feature 1': corr.columns[i],
                        'Feature 2': corr.columns[j],
                        'Correlation': round(corr.iloc[i, j], 3)
                    })
            corr_df = pd.DataFrame(corr_pairs).sort_values(
                'Correlation', key=abs, ascending=False).head(10)
            st.dataframe(corr_df, use_container_width=True)

    with tab4:
        st.markdown("### Missing Values Analysis")

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            st.success("✅ No missing values found in this dataset!")
        else:
            missing_df = pd.DataFrame({
                'Column':  missing.index,
                'Missing': missing.values,
                'Percent': (missing.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(missing_df['Column'],
                   missing_df['Percent'],
                   color='#e74c3c', edgecolor='black')
            ax.set_title('Missing Values by Column (%)', fontsize=13)
            ax.set_ylabel('Missing %')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("### Duplicate Rows")
        dupes = df.duplicated().sum()
        if dupes == 0:
            st.success("✅ No duplicate rows found!")
        else:
            st.warning(f"⚠️ {dupes} duplicate rows found.")
            if st.button("Show duplicates"):
                st.dataframe(df[df.duplicated()],
                             use_container_width=True)

        st.markdown("### Outlier Detection (Numeric Columns)")
        numeric_df = df.select_dtypes(include=[np.number])
        outlier_summary = []
        for col in numeric_df.columns:
            Q1  = numeric_df[col].quantile(0.25)
            Q3  = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = numeric_df[
                (numeric_df[col] < Q1 - 1.5 * IQR) |
                (numeric_df[col] > Q3 + 1.5 * IQR)
            ]
            outlier_summary.append({
                'Column':   col,
                'Outliers': len(outliers),
                'Percent':  round(len(outliers) / len(df) * 100, 2)
            })
        outlier_df = pd.DataFrame(outlier_summary).sort_values(
            'Outliers', ascending=False)
        st.dataframe(outlier_df, use_container_width=True)

else:
    st.info("👆 Upload a CSV file above to get started.")
    st.markdown("### What this app does:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 🔍 **Overview** — preview data, column types, stats
        - 📈 **Visualize** — build charts with any column
        """)
    with col2:
        st.markdown("""
        - 🔗 **Correlations** — heatmap and top pairs
        - 🧹 **Data Quality** — missing values, duplicates, outliers
        """)

st.markdown("---")
st.markdown("Built by **Jyotiraditya** | Upload any CSV and explore instantly")