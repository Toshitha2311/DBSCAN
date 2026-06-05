import streamlit as st
import pandas as pd
import numpy as np
import os

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

    n = 300

    df = pd.DataFrame({
        "Age": np.clip(np.random.normal(40, 12, n).astype(int), 18, 70),
        "Annual Income": np.clip(np.random.normal(70, 25, n).astype(int), 15, 150),
        "Spending Score": np.clip(np.random.normal(50, 20, n).astype(int), 1, 100)
    })

    df.to_csv(file_path, index=False)
    return df

# --------------------------------------------------
# SIDEBAR - DATA INPUT
# --------------------------------------------------

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    save_path = os.path.join("data", uploaded_file.name)
    df.to_csv(save_path, index=False)

    st.sidebar.success("Uploaded dataset loaded!")

else:
    df = load_default_data()
    st.sidebar.info("Using default dataset")

# --------------------------------------------------
# DOWNLOAD SAMPLE DATA
# --------------------------------------------------

sample_csv = load_default_data().to_csv(index=False)

st.sidebar.download_button(
    "Download Sample Dataset",
    sample_csv,
    "sample_data.csv",
    "text/csv"
)

# --------------------------------------------------
# DATA OVERVIEW
# --------------------------------------------------

st.subheader("📁 Dataset Overview")

col1, col2 = st.columns(2)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])

st.dataframe(df.head())

# --------------------------------------------------
# HANDLE MISSING VALUES
# --------------------------------------------------

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# --------------------------------------------------
# FEATURE SELECTION
# --------------------------------------------------

st.sidebar.header("🎯 Feature Selection")

num_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(num_cols) < 2:
    st.error("Need at least 2 numeric columns")
    st.stop()

selected_features = st.sidebar.multiselect(
    "Select Features",
    num_cols,
    default=num_cols[:min(3, len(num_cols))]
)

if len(selected_features) < 2:
    st.warning("Select at least 2 features")
    st.stop()

X = df[selected_features]

# --------------------------------------------------
# SCALING
# --------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# DBSCAN HYPERPARAMETERS
# --------------------------------------------------

st.sidebar.header("⚙ DBSCAN Parameters")

eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1)
min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)

metric = st.sidebar.selectbox(
    "Metric",
    ["euclidean", "manhattan", "cosine"]
)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

model = DBSCAN(
    eps=eps,
    min_samples=min_samples,
    metric=metric
)

clusters = model.fit_predict(X_scaled)

df["Cluster"] = clusters

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------

output_path = "outputs/clustered_dataset.csv"
df.to_csv(output_path, index=False)

# --------------------------------------------------
# CLUSTER STATS
# --------------------------------------------------

noise = (clusters == -1).sum()

n_clusters = len(set(clusters))
if -1 in clusters:
    n_clusters -= 1

col1, col2 = st.columns(2)
col1.metric("Clusters Found", n_clusters)
col2.metric("Noise Points", noise)

# --------------------------------------------------
# SILHOUETTE SCORE
# --------------------------------------------------

valid = len(set(clusters)) > 1

if valid:
    try:
        score = silhouette_score(X_scaled, clusters)
        st.success(f"Silhouette Score: {score:.3f}")
    except:
        st.warning("Silhouette score not available")
else:
    st.warning("Need at least 2 clusters for silhouette score")

# --------------------------------------------------
# CLUSTER DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().sort_index().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

fig = px.bar(
    cluster_counts,
    x="Cluster",
    y="Count",
    color="Cluster",
    title="Cluster Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

st.subheader("🎯 Cluster Visualization")

col1, col2 = st.columns(2)

with col1:
    x_axis = st.selectbox("X Axis", selected_features, 0)

with col2:
    y_axis = st.selectbox("Y Axis", selected_features, 1)

fig2 = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=df["Cluster"].astype(str),
    hover_data=selected_features,
    title="DBSCAN Clusters"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# HEATMAP
# --------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

corr = df[selected_features].corr()

fig3 = px.imshow(
    corr,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(fig3, use_container_width=True)

# --------------------------------------------------
# INSIGHTS
# --------------------------------------------------

st.subheader("🧠 Cluster Insights")

clean_df = df[df["Cluster"] != -1]

if len(clean_df) > 0:
    summary = clean_df.groupby("Cluster")[selected_features].mean().round(2)
    st.dataframe(summary)

    for c in sorted(clean_df["Cluster"].unique()):
        st.markdown(f"### Cluster {c}")
        for f in selected_features:
            st.write(f"{f}: {clean_df[clean_df['Cluster']==c][f].mean():.2f}")
else:
    st.warning("All points are noise!")

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.subheader("⬇ Download Results")

csv = df.to_csv(index=False)

st.download_button(
    "Download Clustered Data",
    csv,
    "clustered_dataset.csv",
    "text/csv"
)

st.success("Clustering completed successfully 🚀")