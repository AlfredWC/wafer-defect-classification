# generate_gradcam_comparisons.py
"""
This script generates side-by-side comparison figures of raw wafer maps and their Grad-CAM overlays
for both high-confidence and low-confidence predictions per defect class using a trained CNN model.

It performs the following steps:
1. Loads the trained CNN model and filters the WM-811K dataset to 26×26 unlabeled wafer maps.
2. One-hot encodes the wafer maps and generates predictions with softmax scores.
3. Splits the predictions into high- and low-confidence groups based on a threshold (95%).
4. Computes Grad-CAM heatmaps for each class (high and low confidence).
5. Saves a visual comparison (raw + Grad-CAM) for each defect class to the ./images folder.

Used in the Streamlit dashboard to visually explain CNN predictions on unlabeled data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model, Model

# Step 1: Load trained CNN model and dataset
model = load_model('./models/cnn_model.h5')
df = pd.read_pickle('./data/LSWMD.pkl')
df['waferMapDim'] = df['waferMap'].apply(lambda x: x.shape)
df = df[df['waferMapDim'] == (26, 26)]

# Step 2: Prepare unlabeled wafer maps
unlabeled_df = df[df['failureType'].apply(lambda x: len(x) == 0)].copy()
raw_maps = np.stack(unlabeled_df['waferMap'].values)
x_unlabeled = np.zeros((len(raw_maps), 26, 26, 3))
for w in range(len(raw_maps)):
    for i in range(26):
        for j in range(26):
            val = raw_maps[w, i, j]
            if val in [0, 1, 2]:
                x_unlabeled[w, i, j, int(val)] = 1

# Step 3: Make predictions and split by confidence
softmax_preds = model.predict(x_unlabeled)
max_probs = softmax_preds.max(axis=1)
pred_labels = softmax_preds.argmax(axis=1)

# Label mapping
inv_map = {
    0: 'Center', 1: 'Donut', 2: 'Edge-Loc', 3: 'Edge-Ring',
    4: 'Loc', 5: 'Near-full', 6: 'Random', 7: 'Scratch', 8: 'none'
}

# High- and low-confidence separation
conf_thresh = 0.95
high_idx = np.where(max_probs > conf_thresh)[0]
low_idx = np.where(max_probs <= conf_thresh)[0]
confident_maps = raw_maps[high_idx]
confident_preds = pred_labels[high_idx]
confident_probs = max_probs[high_idx]
labels = [inv_map[p] for p in confident_preds]

# Save summary DataFrame of high-confidence predictions
defect_classes = list(inv_map.values())
os.makedirs('./images', exist_ok=True)

df_conf = pd.DataFrame({
    'waferMap': list(confident_maps),
    'predictedLabel': labels,
    'confidence': confident_probs
})

df_conf.to_pickle('./data/high_confidence_unlabeled_preds.pkl')
print(f"Saved {len(df_conf)} high-confidence predictions.")
print('There are', len(unlabeled_df), 'unlabeled wafer maps with 26×26 dimensions')
print(f"Out of {len(unlabeled_df)} unlabeled wafers, {len(df_conf)} were confidently labeled (>{conf_thresh*100:.0f}% confidence)")
print(f"Percentage labeled: {len(df_conf)/len(unlabeled_df)*100:.2f}%")

# Step 4: Grad-CAM computation function
def compute_gradcam(model, img, class_index):
    conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    last_conv = conv_layers[-1].name
    grad_model = Model(inputs=model.input, outputs=[model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-6
    heatmap = tf.image.resize(heatmap[..., np.newaxis], (26, 26)).numpy().squeeze()
    return heatmap

# Step 5: Generate and save comparison figures
def save_gradcam_comparison(img, heatmap, label, conf_level, confidence_score):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(np.argmax(img, axis=-1))
    ax[0].set_title("Raw Wafer Map")
    ax[1].imshow(np.argmax(img, axis=-1), cmap='gray')
    ax[1].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[1].set_title(f"{label} ({confidence_score*100:.1f}%)")
    for a in ax:
        a.axis('off')
    fname = f"./images/gradcam_{label}_{conf_level}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close(fig)

# For each class, select one example from high and low confidence
for cls_idx, cls_name in inv_map.items():
    # High-confidence
    idx_h = high_idx[pred_labels[high_idx] == cls_idx]
    if len(idx_h) > 0:
        i = idx_h[0]
        img = x_unlabeled[i]
        conf_score = max_probs[i]
        heatmap = compute_gradcam(model, img, cls_idx)
        save_gradcam_comparison(img, heatmap, cls_name, 'high', conf_score)

    # Low-confidence
    idx_l = low_idx[pred_labels[low_idx] == cls_idx]
    if len(idx_l) > 0:
        i = idx_l[0]
        img = x_unlabeled[i]
        conf_score = max_probs[i]
        heatmap = compute_gradcam(model, img, cls_idx)
        save_gradcam_comparison(img, heatmap, cls_name, 'low', conf_score)

print("Saved 18 Grad-CAM comparison figures to ./images/")
