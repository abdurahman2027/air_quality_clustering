import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# -------------------------
# Config & load data
# -------------------------
st.set_page_config(page_title="Air Quality K-Means", layout="wide")

# Title & sub-header
# -------------------------
st.title("🏕️ Air Quality Clustering Model")
st.subheader("K-Means clustering demo — Developed by Abdu Rahman")


# Sidebar: profile, help link, inputs (sliders + exact boxes)
# -------------------------
PROFILE_IMG = "profile.jpg"
with st.sidebar:
    if os.path.exists(PROFILE_IMG):
        st.image(PROFILE_IMG, width=140)
    else:
        st.info("profile.jpg")
    st.markdown("**Abdu Rahman**  \n**Student ID:** PIUS20230015")
    st.markdown("*An Aspiring Data Scientist*")

    # helpful external link (EPA)
    st.subheader("ℹ️ Learn about features")
    st.markdown(
        "If you're unfamiliar with the pollutant names and sensors, this short official guide is helpful:"
    )
    st.markdown(
        "[Meaning of Air Quality Variables](https://archive.ics.uci.edu/dataset/360/air+quality)"
    )

CSV = "AirQuality_clean.csv"  
FEATURES = ["CO_GT","NO2_GT","Nox_GT","C6H6_GT","NMHC_GT","PT08_S5_O3","T"]

# -------------------------
# Cluster interpretation 
# -------------------------
CLUSTER_DESCRIPTIONS = {
    0: "Medium pollutant concentration level",
    1: "Higher pollutant concentration level",
    2: "Lower pollutant concentration level"
}

# Load cleaned data
df = pd.read_csv(CSV)
X = df[FEATURES].copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Parameter Interface")
feature_list = FEATURES.copy()
x_feature = st.sidebar.selectbox("Select X-axis feature", feature_list, index=0)
y_feature = st.sidebar.selectbox("Select Y-axis feature", feature_list, index=1)

k = st.sidebar.slider("Number of clusters (k)", 2, 8, 3)
cluster_choice = st.sidebar.selectbox("Select cluster to view", list(range(k)))

# -------------------------
# Fit KMeans (on scaled data)
# -------------------------
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
df["cluster"] = labels

# -------------------------
# Layout: table + visualization
# -------------------------
st.header("Cluster Data & Visualization")
col_table, col_chart = st.columns(2)

with col_table:
    st.subheader("Cluster Data Table")
    st.write(df[df["cluster"] == cluster_choice][[*FEATURES, "cluster"]])

    if st.checkbox("Show full dataset"):
        st.write(df[[*FEATURES, "cluster"]])

with col_chart:
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(6,5))
    colors = plt.cm.get_cmap("tab10", k)
    for c in range(k):
        c_idx = df["cluster"] == c
        ax.scatter(
            df.loc[c_idx, x_feature],
            df.loc[c_idx, y_feature],
            s=20,
            alpha=0.7,
            label=f"Cluster {c}",
            color=colors(c)
        )

    # plot centroids (inverse transform centroids from scaled space to original units)
    centroids_scaled = kmeans.cluster_centers_
    xi = FEATURES.index(x_feature)
    yi = FEATURES.index(y_feature)
    centroids_orig = scaler.inverse_transform(centroids_scaled)
    ax.scatter(
        centroids_orig[:, xi], centroids_orig[:, yi],
        marker="X", s=200, color="black", label="Centroids"
    )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

# -------------------------
# Predict cluster for new instance
# -------------------------
st.header("Predict Cluster for New Input")

col1, col2, col3 = st.columns(3)
with col1:
    co = st.number_input("CO_GT", value=float(X["CO_GT"].median()))
    no2 = st.number_input("NO2_GT", value=float(X["NO2_GT"].median()))
with col2:
    nox = st.number_input("Nox_GT", value=float(X["Nox_GT"].median()))
    c6h6 = st.number_input("C6H6_GT", value=float(X["C6H6_GT"].median()))
with col3:
    nmhc = st.number_input("NMHC_GT", value=float(X["NMHC_GT"].median()))
    pt08 = st.number_input("PT08_S5_O3", value=float(X["PT08_S5_O3"].median()))
    t = st.number_input("T", value=float(X["T"].median()))

if st.button("Predict Cluster"):
    new_point = np.array([[co, no2, nox, c6h6, nmhc, pt08, t]])
    new_scaled = scaler.transform(new_point)
    pred = kmeans.predict(new_scaled)[0]
    st.success(f"This input belongs to cluster: {pred}")

    # -------------------------
    # Show cluster interpretation
    # -------------------------
    st.info(f"Cluster interpretation: **{CLUSTER_DESCRIPTIONS.get(pred, 'N/A')}**")

st.markdown("---")
st.caption("This is Air Quality K-Means Clustering — for educational use only, not for a real-world air quality prediction system.")
