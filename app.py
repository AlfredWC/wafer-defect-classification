# app.py — Streamlit dashboard for wafer defect classification
"""
This interactive dashboard visualizes wafer defect classification results using:
- Traditional SVMs on handcrafted features (original + augmented data)
- CNN trained on autoencoder-augmented data
- Deployment analysis on unlabeled wafers using high-confidence predictions

Sections:
1. Dataset Overview
2. Model Comparisons
3. Model Deployment: High-confidence Predictions & Grad-CAM visualizations
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from keras.models import load_model

# Configure the Streamlit page
st.set_page_config(layout="wide")
st.title("Yield Engineering: Wafer Defect Classification Dashboard")

# Sidebar navigation menu
section = st.sidebar.selectbox(
    "Go to",
    ["Section 1: Overview", "Section 2: Model Comparisons", "Section 3: Model Deployment: Unlabeled Data"]
)

# Cache data loading for performance
@st.cache_data
def load_data():
    # Original raw dataset
    df = pd.read_pickle('./data/LSWMD.pkl')

    # Augmented dataset: CNN training data
    npz = np.load('./data/augmented_dataset.npz', allow_pickle=True)
    aug_X = npz['X']
    aug_Y = npz['Y']
    npz.close()

    # High-confidence predictions on unlabeled wafers
    df_conf = pd.read_pickle('./data/high_confidence_unlabeled_preds_95.pkl')

    return df, aug_X, aug_Y, df_conf

# Load all required data at the start
df, aug_X, aug_Y, df_conf = load_data()

# Full list of defect class names (0–8)
defect_classes = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]

# --- Section 1: Dataset Overview ---
if section == "Section 1: Overview":
    st.header("Section 1: Dataset Overview")

    # 1-1: General dataset stats
    st.subheader("1-1: Data Summary")
    st.markdown(
        """
- **Data Source**: [Kaggle WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm-811k-wafer-map)
- **Attributes**: waferMap, failureType, lot, waferIndex, dieSize, etc.
- **Wafer Sizes**: Range 6×21 to 300×202, 632 unique dimensions.
"""
    )
    df['dim'] = df['waferMap'].apply(lambda x: x.shape)
    total = len(df)
    labeled = df[df['failureType'].apply(lambda x: len(x) > 0)]
    unlabeled = df[df['failureType'].apply(lambda x: len(x) == 0)]

    # Display high-level metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Total wafers", f"{total:,}")
    c2.metric("Labeled wafers", f"{len(labeled):,}")
    c3.metric("Unlabeled wafers", f"{len(unlabeled):,}")

    # 1-2: Bar chart of labeled defect class distribution
    st.subheader("1-2: Defect Distribution (Labeled)")
    dist = labeled['failureType'].apply(lambda x: x[0][0]).value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=dist.index, y=dist.values, palette='viridis', ax=ax)
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    st.pyplot(fig)

    # 1-3: Distribution table of augmented dataset
    st.subheader("1-3: Augmented Dataset Distribution")
    aug_counts = np.argmax(aug_Y, axis=1)
    aug_dist = pd.Series(aug_counts).value_counts().sort_index()
    df_aug = pd.DataFrame({
        'Defect Type': [defect_classes[i] for i in aug_dist.index],
        'Count': aug_dist.values,
        'Percentage (%)': (aug_dist.values / len(aug_counts) * 100).round(2)
    })
    st.dataframe(df_aug)

# --- Section 2: Model Comparisons ---
elif section == "Section 2: Model Comparisons":
    st.header("Section 2: Model Comparisons")

    # 2-1: SVM on original data
    st.subheader("2-1: SVM w/ Original Data")
    st.markdown(
        """
- Handcrafted features (Radon, geometry, symmetry)
- One-vs-One Linear SVM
- Source: Hyundoil's blog
"""
    )
    c1, c2 = st.columns(2)
    with c1:
        st.image("./images/wu_conf_matrix.png", width=300)
        st.text("Accuracy: ~83%")
    with c2:
        st.image("./images/wu_conf_matrix_norm.png", width=300)

    # 2-2: SVM with augmented data
    st.subheader("2-2: SVM w/ Augmented Data")
    st.markdown("Alleviating class imbalance with autoencoder augmentation.")
    c3, c4 = st.columns(2)
    with c3:
        st.image("./images/wu_conf_matrix(augmented_data).png", width=300)
    with c4:
        st.image("./images/wu_conf_matrix_norm(augmented_data).png", width=300)

    # 2-3: CNN trained on augmented data
    st.subheader("2-3: CNN w/ Augmented Data")
    st.markdown("Supervised CNN trained on augmented data for improved deep representation.")
    c5, c6 = st.columns(2)
    with c5:
        st.image("./images/cnn_conf_matrix.png", width=300)
    with c6:
        st.image("./images/cnn_conf_matrix_norm.png", width=300)

    # Summary of model performance
    st.markdown("### Metrics Summary")
    st.table(pd.DataFrame({
        'Model': ['SVM (orig)', 'SVM (aug)', 'CNN (aug)'],
        'Accuracy': ['83.0%', '84.2%', '91.7%'],
        'F1 Score': ['0.83', '0.84', '0.92']
    }))

# --- Section 3: Deployment Results on Unlabeled Data ---
else:
    st.header("Section 3: Model Deployment: Unlabeled Data Analysis")
    st.markdown(
        "Filtered 26×26 unlabeled wafers, kept high-confidence (>95%) predictions, and applied Grad-CAM to low-confidence samples."
    )

    # 3-1: Table of high-confidence predictions
    st.subheader("3-1: High-Confidence Predictions")
    if 'predictedLabel' in df_conf.columns:
        hc = df_conf['predictedLabel'].value_counts()
        df_hc = pd.DataFrame({
            'Defect Type': hc.index,
            'Count': hc.values,
            'Percentage (%)': (hc.values / len(df_conf) * 100).round(2)
        })
        st.dataframe(df_hc)
    else:
        st.warning("No PredictedLabel column in high-confidence data.")

    # 3-2: Grad-CAM visualizations for each class tab
    st.subheader("3-2: Grad-CAM Visualizations (Low Confidence)")
    tabs = st.tabs(defect_classes)
    for tab, defect in zip(tabs, defect_classes):
        with tab:
            img_path = f"./images/gradcam_{defect}.png"
            if os.path.exists(img_path):
                st.image(img_path, caption=defect, width=300)
            else:
                st.warning(f"Missing: {img_path}")
