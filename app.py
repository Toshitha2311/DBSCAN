
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="DBSCAN Clustering Dashboard",
    layout="wide"
)

st.title("🔍 DBSCAN Clustering Dashboard")

# --------------------------------------------------
# CREATE FOLDERS
# --------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# DEFAULT DATASET
# --------------------------------------------------

@st.cache_data
def load_default_data():

    file_path = "data/sample_customer_data.csv"

    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    np.random.seed(42)

    customers = 300

    age = np.random.normal(40, 12, customers).astype(int)
    income = np.random.normal(70, 25, customers).astype(int)
    spending = np.random.normal(50, 20, customers).astype(int)

    age = np.clip(age, 18, 70)
    income = np.clip(income, 15, 150)
    spending = np.clip(spending, 1, 100)

    df = pd.DataFrame({
        "Age": age,
        "Annual Income": income,
        "Spending Score": spending
    })

    df.to_csv(file_path, index=False)

    return df

# --------------------------------------------------
# SIDEBAR - DATASET
# --------------------------------------------------

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    save_path = os.path.join(
        "data",
        uploaded_file.name
    )

    df.to_csv(save_path, index=False)

    st.sidebar.success("Uploaded dataset loaded.")

else:

    df = load_default_data()

    st.sidebar.info("Using built-in sample dataset.")

# --------------------------------------------------
# DOWNLOAD SAMPLE DATA
# --------------------------------------------------

sample_csv = load_default_data().to_csv(index=False)

st.sidebar.download_button(
    "Download Sample Dataset",
    sample_csv,
    "sample_customer_data.csv",
    "text/csv"
)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("📁 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

st.dataframe(df.head())

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

st.sidebar.header("🎯 Feature Selection")

numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_cols) < 2:
    st.error(
        "Dataset must contain at least 2 numeric columns."
    )
    st.stop()

selected_features = st.sidebar.multiselect(
    "Select Features",
    numeric_cols,
    default=numeric_cols[:min(3, len(numeric_cols))]
)

if len(selected_features) < 2:
    st.warning(
        "Please select at least 2 features."
    )
    st.stop()

X = df[selected_features]

# --------------------------------------------------
# FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# DBSCAN HYPERPARAMETERS
# --------------------------------------------------

st.sidebar.header("⚙ DBSCAN Hyperparameters")

eps = st.sidebar.slider(
    "Epsilon (eps)",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1
)

min_samples = st.sidebar.slider(
    "Min Samples",
    min_value=2,
    max_value=20,
    value=5
)

metric = st.sidebar.selectbox(
    "Distance Metric",
    ["euclidean", "manhattan", "cosine"]
)

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["auto", "ball_tree", "kd_tree", "brute"]
)

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

model = DBSCAN(
    eps=eps,
    min_samples=min_samples,
    metric=metric,
    algorithm=algorithm
)

clusters = model.fit_predict(X_scaled)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    {
        "model": model,
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "algorithm": algorithm,
        "features": selected_features
    },
    "models/dbscan_model.pkl"
)

# --------------------------------------------------
# ADD CLUSTERS
# --------------------------------------------------

df["Cluster"] = clusters

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

df.to_csv(
    "outputs/clustered_dataset.csv",
    index=False
)

# --------------------------------------------------
# CLUSTER STATISTICS
# --------------------------------------------------

noise_points = (clusters == -1).sum()

n_clusters = len(set(clusters))
if -1 in clusters:
    n_clusters -= 1

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Clusters Found",
        n_clusters
    )

with col2:
    st.metric(
        "Noise Points",
        noise_points
    )

# --------------------------------------------------
# SILHOUETTE SCORE
# --------------------------------------------------

valid_clusters = len(set(clusters))

if valid_clusters > 1:

    try:

        score = silhouette_score(
            X_scaled,
            clusters
        )

        st.success(
            f"Silhouette Score: {score:.3f}"
        )

    except:
        st.warning(
            "Unable to calculate Silhouette Score."
        )

else:

    st.warning(
        "Silhouette Score requires at least 2 clusters."
    )

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Cluster Distribution")

cluster_count = (
    df["Cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_count.columns = [
    "Cluster",
    "Count"
]

fig_bar = px.bar(
    cluster_count,
    x="Cluster",
    y="Count",
    color="Cluster",
    title="Cluster Distribution"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# --------------------------------------------------
# CLUSTER VISUALIZATION
# --------------------------------------------------

st.subheader("🎯 Cluster Visualization")

col1, col2 = st.columns(2)

with col1:

    x_axis = st.selectbox(
        "X Axis",
        selected_features,
        index=0
    )

with col2:

    y_axis = st.selectbox(
        "Y Axis",
        selected_features,
        index=min(
            1,
            len(selected_features) - 1
        )
    )

fig_scatter = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=df["Cluster"].astype(str),
    hover_data=selected_features,
    title="DBSCAN Clustering Visualization"
)

st.plotly_chart(
    fig_scatter,
    use_container_width=True
)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

corr = df[selected_features].corr()

fig_heat = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Feature Correlation"
)

st.plotly_chart(
    fig_heat,
    use_container_width=True
)

# --------------------------------------------------
# CLUSTER INSIGHTS
# --------------------------------------------------

st.subheader("🧠 Cluster Insights")

filtered_df = df[df["Cluster"] != -1]

if len(filtered_df) > 0:

    summary = (
        filtered_df
        .groupby("Cluster")[selected_features]
        .agg(["mean", "min", "max"])
        .round(2)
    )

    st.dataframe(summary)

    for cluster in sorted(
        filtered_df["Cluster"].unique()
    ):

        st.markdown(
            f"### Cluster {cluster}"
        )

        subset = filtered_df[
            filtered_df["Cluster"] == cluster
        ]

        for feature in selected_features:

            avg = subset[
                feature
            ].mean()

            st.write(
                f"Average {feature}: {avg:.2f}"
            )

else:

    st.warning(
        "All points were classified as noise."
    )

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("⬇ Download Clustered Dataset")

csv = df.to_csv(index=False)

st.download_button(
    "Download Results",
    csv,
    "clustered_dataset.csv",
    "text/csv"
)

# --------------------------------------------------
# STATUS
# --------------------------------------------------

st.success(
    "Model saved in models/dbscan_model.pkl"
)

st.success(
    "Results saved in outputs/clustered_dataset.csv"
)

