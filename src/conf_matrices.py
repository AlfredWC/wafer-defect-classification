# conf_matrices.py
"""
This script loads the trained CNN model and the augmented dataset to
generate and save both the raw and normalized confusion matrices
as PNG images under the `./images/` directory.

This is useful for visual inspection and analysis without needing to rerun the training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from utils.wu_viz import plot_confusion_matrix  # Custom function to make prettier plots

# 1) Load preprocessed augmented dataset
data = np.load('./data/augmented_dataset.npz', allow_pickle=True)
X = data['X']        # Shape: (N, 26, 26, 3) — wafer maps
Y = data['Y']        # Shape: (N, 9) — one-hot encoded labels

# Split into train/test using same seed and stratification for consistency
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=2019, stratify=np.argmax(Y, axis=1)
)

# 2) Load the trained CNN model (generated from train_cnn_model.py)
model = load_model('./models/cnn_model.h5')

# 3) Predict class probabilities on the test set
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)   # Convert softmax output to class indices
y_true = np.argmax(y_test, axis=1)         # Convert one-hot labels to class indices

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# 4) Save confusion matrices as images (raw and normalized)
os.makedirs('./images', exist_ok=True)

class_names = [
    'Center', 'Donut', 'Edge-Loc', 'Edge-Ring',
    'Loc', 'Near-full', 'Random', 'Scratch', 'none'
]

# Save both raw and normalized versions
for norm in [False, True]:
    fig, ax = plt.subplots(figsize=(6, 6))
    title = f"CNN {'Normalized ' if norm else ''}Confusion Matrix"
    plot_confusion_matrix(cm, class_names, normalize=norm, title=title)

    # Set output filename
    fname = f"./images/cnn_conf_matrix{'_norm' if norm else ''}.png"
    fig.savefig(fname, dpi=300)
    plt.close(fig)

print("✅ Saved both raw and normalized confusion matrices to ./images/")
