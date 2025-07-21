# data_augmenter.py
"""
This script filters labeled wafer maps to 26×26 size and trains
a convolutional autoencoder to generate synthetic (augmented) samples
for underrepresented defect classes.

Steps include:
- Extracting all labeled 26×26 wafer maps.
- One-hot encoding pixel values into 3 channels.
- Training a shallow encoder-decoder autoencoder.
- Generating new samples by perturbing the latent space.
- Balancing class distribution by upsampling minority classes
  and slightly downsampling the majority 'none' class.

The final augmented dataset is saved as `augmented_dataset.npz`
for use in both traditional ML and DL training pipelines.
"""

import os
import numpy as np
import pandas as pd
from keras import layers, Input, models
from keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore")

# 1) Load dataset and filter only 26×26 wafer maps
df = pd.read_pickle('./data/LSWMD.pkl')
df['waferMapDim'] = df['waferMap'].apply(lambda x: x.shape)
sub_df = df[df['waferMapDim'] == (26, 26)].reset_index(drop=True)

# 2) Extract wafer maps and labels
sw = np.ones((1, 26, 26))  # dummy starter array
labels = []
for i in range(len(sub_df)):
    ft = sub_df.loc[i, 'failureType']
    if len(ft) == 0:
        continue
    sw = np.concatenate((sw, sub_df.loc[i, 'waferMap'].reshape(1, 26, 26)), axis=0)
    labels.append(ft[0][0])

x = sw[1:]  # remove dummy
y = np.array(labels).reshape(-1, 1)
x = x.reshape(-1, 26, 26, 1)

# 3) Convert to 3-channel (binary one-hot: 0, 1, 2)
new_x = np.zeros((len(x), 26, 26, 3), dtype=np.uint8)
for w in range(len(x)):
    for i in range(26):
        for j in range(26):
            new_x[w, i, j, int(x[w, i, j])] = 1

# 4) Build autoencoder (Conv2D + MaxPool → Conv2DTranspose + Upsampling)
input_tensor = Input((26, 26, 3))
encode = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_tensor)
latent_vector = layers.MaxPool2D()(encode)

decode_layer_1 = layers.Conv2DTranspose(64, (3,3), padding='same', activation='relu')
decode_layer_2 = layers.UpSampling2D()
output_tensor = layers.Conv2DTranspose(3, (3,3), padding='same', activation='sigmoid')

# Build full autoencoder
dec = decode_layer_1(latent_vector)
dec = decode_layer_2(dec)
ae = models.Model(input_tensor, output_tensor(dec))
ae.compile(optimizer='Adam', loss='mse')
ae.fit(new_x, new_x, batch_size=512, epochs=15, verbose=1)

# 5) Split encoder and decoder for latent sampling
encoder = models.Model(input_tensor, latent_vector)
dec_input = Input((13, 13, 64))
dec = decode_layer_1(dec_input)
dec = decode_layer_2(dec)
decoder = models.Model(dec_input, output_tensor(dec))

# 6) Define augmentation function for each defect type
def gen_data(wafer, label):
    z = encoder.predict(wafer)
    gen_x = np.zeros((1, 26, 26, 3))
    reps = (2000 // len(wafer)) + 1
    for _ in range(reps):
        z2 = z + np.random.normal(0, 0.1, z.shape)
        g = decoder.predict(z2)
        gen_x = np.concatenate((gen_x, g), axis=0)
    gen_y = np.full((len(gen_x), 1), label)
    return gen_x[1:], gen_y[1:]

# 7) Augment minority defect classes (excluding 'none')
faulty_case = np.unique(y)
for cls in faulty_case:
    if cls == 'none':
        continue
    idx = np.where(y == cls)[0]
    gx, gy = gen_data(new_x[idx], cls)
    new_x = np.concatenate((new_x, gx), axis=0)
    y = np.concatenate((y, gy), axis=0)

# 8) Downsample 'none' class to reduce imbalance
none_idx = np.where(y == 'none')[0]
rm = np.random.choice(none_idx, size=83, replace=False)
new_x = np.delete(new_x, rm, axis=0)
y = np.delete(y, rm, axis=0)

# 9) Label encode defect types → one-hot
faulty_case = np.unique(y)
new_y = y.copy().reshape(-1)
for i, l in enumerate(faulty_case):
    new_y[new_y == l] = i
new_y = new_y.astype(int)
new_y = to_categorical(new_y)

# 10) Save to .npz for reuse in other training scripts
os.makedirs('./data', exist_ok=True)
np.savez_compressed('./data/augmented_dataset.npz', X=new_x, Y=new_y)
print(f"Saved augmented dataset with {new_x.shape[0]} samples (X) and labels Y of shape {new_y.shape}.")
