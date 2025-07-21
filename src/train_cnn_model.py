# train_cnn_model.py
"""
This script defines, trains, and evaluates a convolutional neural network
for wafer defect classification using the autoencoder-augmented dataset.

It includes:
- CNN model definition using Keras functional API.
- 3-fold cross-validation to validate generalization.
- Final training and test accuracy reporting.
- Plots of training loss and accuracy curves.
- Confusion matrix analysis and visualization.
- Saves the trained model as `cnn_model.h5` for reuse.

This script provides the deep learning benchmark for comparison
against feature-based SVM approaches.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix
from scikeras.wrappers import KerasClassifier
from keras import layers, Input, models
from utils.wu_viz import plot_confusion_matrix

# 1) Load the augmented wafer map dataset
dataset_path = './data/augmented_dataset.npz'
if not os.path.exists(dataset_path):
    print('→ Augmented dataset not found. Generating via data_augmenter.py...')
    import data_augmenter_this  # Will generate and save the augmented data

data = np.load(dataset_path, allow_pickle=True)
X = data['X']  # Shape: (N, 26, 26, 3) — 3-channel one-hot wafer map
Y = data['Y']  # Shape: (N, 9) — one-hot encoded labels
print(f'Loaded augmented dataset: X shape {X.shape}, Y shape {Y.shape}')

# 2) Split the dataset into training and testing sets (stratified by class)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.33, random_state=2019, stratify=np.argmax(Y, axis=1)
)
print(f'Train: {x_train.shape}, {y_train.shape}   Test: {x_test.shape}, {y_test.shape}')

# 3) Define CNN architecture for wafer defect classification
def create_model():
    inp = Input(shape=(26,26,3))  # Input wafer image (26x26x3)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    flat = layers.Flatten()(x)
    d1 = layers.Dense(512, activation='relu')(flat)
    d2 = layers.Dense(128, activation='relu')(d1)
    out = layers.Dense(Y.shape[1], activation='softmax')(d2)  # 9-class softmax output
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4) Perform 3-fold cross-validation for initial evaluation
cnn = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1024, verbose=1)
kfold = KFold(n_splits=3, shuffle=True, random_state=2019)
scores = cross_val_score(cnn, x_train, y_train, cv=kfold, n_jobs=-1)
print(f'3-Fold CV Accuracy: {np.mean(scores):.4f}')

# 5) Train final CNN model using the full training set
history = cnn.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=512
)
print(f'Final Testing Accuracy: {cnn.score(x_test, y_test):.4f}')

# 6) Visualize training performance
hist = history.history_

plt.figure(figsize=(6,4))
plt.plot(hist['loss'], label='train_loss')
plt.plot(hist['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss'); plt.show()

plt.figure(figsize=(6,4))
plt.plot(hist['accuracy'], label='train_acc')
plt.plot(hist['val_accuracy'], label='val_acc')
plt.legend(); plt.title('Accuracy'); plt.show()

# 7) Evaluate model with confusion matrix on test data
y_pred_probs = cnn.predict(x_test)
if y_pred_probs.ndim > 1:
    y_pred = np.argmax(y_pred_probs, axis=1)
else:
    y_pred = y_pred_probs

y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred)
class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'none']

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix')

plt.subplot(1,2,2)
plot_confusion_matrix(cm, class_names, normalize=True, title='Normalized Confusion Matrix')
plt.show()

# 8) Save the trained CNN model to disk
os.makedirs('./models', exist_ok=True)
cnn.model_.save('./models/cnn_model.h5')  # .model_ used because of scikeras wrapper
print("→ Saved trained CNN model to './models/cnn_model.h5'")
