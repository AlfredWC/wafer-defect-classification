# train_wu_model.py
"""
This script trains a Wu-style SVM classifier using handcrafted features
extracted from labeled wafer maps (failureType 0–7 only). It includes:

- Feature extraction using spatial, geometric, and Radon-based methods.
- One-vs-One Linear SVM classification with scikit-learn.
- Evaluation through classification reports and confusion matrices.
- Saves the trained model as a `.pkl` file.
- Optionally generates and saves raw and normalized confusion matrix images.

Run this to benchmark the SVM baseline using traditional ML approaches.
"""

import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

from wu_features import extract_features  # Custom feature extraction logic
from utils.wu_viz import plot_confusion_matrix  # Custom visualization function


def load_images_and_labels(df):
    """Convert failure types to numeric labels and extract features from wafer maps."""
    df['failureType'] = df['failureType'].apply(lambda x: str(np.squeeze(x)))
    mapping_type = {
        'Center': 0, 'Donut': 1, 'Edge-Loc': 2, 'Edge-Ring': 3,
        'Loc': 4, 'Random': 5, 'Scratch': 6, 'Near-full': 7, 'none': 8
    }
    df['failureNum'] = df['failureType'].map(mapping_type)
    df = df[df['failureNum'] < 8].reset_index(drop=True)  # Only keep pattern defects
    images = df['waferMap'].values
    labels = df['failureNum'].values
    features = np.array([extract_features(img) for img in images])
    return features, labels


def train_and_evaluate(df, save_model_path='wu_svm_model.pkl'):
    """Train and evaluate One-vs-One SVM classifier."""
    X, y = load_images_and_labels(df)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Show class distribution
    print('Training target stats:', Counter(y_train))
    print('Testing target stats:', Counter(y_test))

    # Train SVM model
    model = OneVsOneClassifier(LinearSVC(random_state=42))
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    print(f'Training Accuracy: {accuracy_score(y_train, y_train_pred) * 100:.2f}%')
    print(f'Testing Accuracy: {accuracy_score(y_test, y_test_pred) * 100:.2f}%')

    # Reports
    print("\nTrain Classification Report:\n", classification_report(y_train, y_train_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

    # Save model to disk
    with open(save_model_path, 'wb') as f:
        pickle.dump(model, f)

    # Confusion matrices
    class_names = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
    cm = confusion_matrix(y_test, y_test_pred)

    # Visualize and save
    os.makedirs('./images', exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,6))
    plot_confusion_matrix(cm, class_names, normalize=False, title='Wu SVM Confusion Matrix (Original Data)')
    fig.savefig('./images/wu_conf_matrix(original_data).png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,6))
    plot_confusion_matrix(cm, class_names, normalize=True, title='Wu SVM Norm Conf Matrix (Original Data)')
    fig.savefig('./images/wu_conf_matrix_norm(original_data).png', dpi=300)
    plt.close(fig)

    return model, (X_test, y_test, y_test_pred)

# Optional usage:
# if __name__ == '__main__':
#     df = pd.read_pickle('./data/LSWMD.pkl')
#     model, results = train_and_evaluate(df)
