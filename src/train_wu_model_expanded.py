# train_wu_model_expanded.py
"""
This script trains a Wu-style SVM classifier on the autoencoder-augmented wafer map dataset.

Differences from the original:
- Loads balanced class samples from `augmented_dataset.npz`.
- Extracts handcrafted features using `wu_features.py`.
- Trains a One-vs-One Linear SVM on the extracted features.
- Evaluates model performance with accuracy, classification reports, and confusion matrices.
- Saves confusion matrix plots and the trained model to disk.

This is the ML baseline for defect classification using augmented data.
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

from wu_features import extract_features
from utils.wu_viz import plot_confusion_matrix

# Step 1: Load 1000 samples per defect class (0–7) from .npz dataset
def load_augmented_subset_npz(path='./data/augmented_dataset.npz', samples_per_class=1000):
    data = np.load(path, allow_pickle=True)
    X = data['X']  # shape (N, 26, 26, 3), one-hot wafer maps
    Y = data['Y']  # shape (N, 9), one-hot encoded labels

    y_idx = np.argmax(Y, axis=1)                # convert one-hot to class index
    maps = np.argmax(X, axis=3)                 # convert one-hot wafer maps to grayscale (0,1,2)

    # Map numeric label to original defect name
    inv_map = {
        0: 'Center', 1: 'Donut', 2: 'Edge-Loc', 3: 'Edge-Ring',
        4: 'Loc', 5: 'Random', 6: 'Scratch', 7: 'Near-full', 8: 'none'
    }
    failure_labels = [inv_map[i] for i in y_idx]

    df = pd.DataFrame({
        'waferMap': list(maps),
        'failureType': failure_labels
    })

    # Only select samples_per_class from defect types 0–7 (exclude 'none')
    selected = []
    for cls in range(8):
        class_df = df[y_idx == cls].sample(n=samples_per_class, random_state=42)
        class_df = class_df.copy()
        class_df['failureNum'] = cls
        selected.append(class_df)

    sub_df = pd.concat(selected).reset_index(drop=True)
    print("Loaded", len(sub_df), "samples from .npz data")
    return sub_df

# Step 2: Extract handcrafted features from wafer maps
def load_images_and_labels(df):
    images = df['waferMap'].values
    labels = df['failureNum'].values
    features = np.array([extract_features(img) for img in images])
    return features, labels

# Step 3: Train One-vs-One SVM and evaluate
def train_and_evaluate(df, save_model_path='wu_svm_model_expanded.pkl'):
    X, y = load_images_and_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print('Training target stats:', Counter(y_train))
    print('Testing target stats:', Counter(y_test))

    model = OneVsOneClassifier(LinearSVC(random_state=42))
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f'Training Accuracy: {train_acc * 100:.2f}%')
    print(f'Testing Accuracy: {test_acc * 100:.2f}%')

    print("\nTrain Classification Report:\n", classification_report(y_train, y_train_pred))
    print("Test Classification Report:\n", classification_report(y_test, y_test_pred))

    # Save the trained model to disk
    with open(save_model_path, 'wb') as f:
        pickle.dump(model, f)

    # Create and save confusion matrices
    class_names = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full']
    cm = confusion_matrix(y_test, y_test_pred)

    os.makedirs('./images', exist_ok=True)

    fig, ax = plt.subplots(figsize=(6,6))
    plot_confusion_matrix(cm, class_names, normalize=False, title='Wu SVM Confusion Matrix (Augmented Data)')
    fig.savefig('./images/wu_conf_matrix(augmented_data).png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,6))
    plot_confusion_matrix(cm, class_names, normalize=True, title='Wu SVM Norm Conf Matrix (Augmented Data)')
    fig.savefig('./images/wu_conf_matrix_norm(augmented_data).png', dpi=300)
    plt.close(fig)

    return model, (X_test, y_test, y_test_pred)

# Optional entry point
if __name__ == '__main__':
    df_subset = load_augmented_subset_npz(samples_per_class=1000)
    model, results = train_and_evaluate(df_subset)
