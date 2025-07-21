# wu_features.py
"""
This module defines functions for extracting handcrafted features
from wafer maps for use in Wu et al.-style SVM classification.

It includes:
- Local density calculations in 13 fixed wafer regions.
- Radon transform-based statistical descriptors.
- Geometric properties from connected components in the map.

All feature vectors are concatenated into a fixed-length vector
representing a single wafer.

Used by `train_wu_model.py` for feature-based SVM classification.
"""

import numpy as np
from skimage.transform import radon
from skimage import measure
from scipy import interpolate, stats

# Calculates defect density in a region as % of total pixels
# (Assumes defects are encoded with value 2)
def cal_den(x):
    return 100 * (np.sum(x == 2) / np.size(x))

# Splits image into 13 regions (center, edges, and inner blocks) and
# computes density feature for each region
def find_regions(x):
    rows, cols = x.shape
    ind1 = np.arange(0, rows, rows // 5)
    ind2 = np.arange(0, cols, cols // 5)
    regions = [
        x[ind1[0]:ind1[1], :],            # Top
        x[:, ind2[4]:],                   # Right
        x[ind1[4]:, :],                   # Bottom
        x[:, ind2[0]:ind2[1]],            # Left
        x[ind1[1]:ind1[2], ind2[1]:ind2[2]], x[ind1[1]:ind1[2], ind2[2]:ind2[3]], x[ind1[1]:ind1[2], ind2[3]:ind2[4]],
        x[ind1[2]:ind1[3], ind2[1]:ind2[2]], x[ind1[2]:ind1[3], ind2[2]:ind2[3]], x[ind1[2]:ind1[3], ind2[3]:ind2[4]],
        x[ind1[3]:ind1[4], ind2[1]:ind2[2]], x[ind1[3]:ind1[4], ind2[2]:ind2[3]], x[ind1[3]:ind1[4], ind2[3]:ind2[4]]
    ]
    return [cal_den(r) for r in regions]

# Computes interpolated mean of radon transform at 20 evenly spaced points
def cubic_inter_mean(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    x_mean = np.mean(sinogram, axis=1)
    f = interpolate.interp1d(np.linspace(1, len(x_mean), len(x_mean)), x_mean, kind='cubic')
    return f(np.linspace(1, len(x_mean), 20)) / 100

# Computes interpolated std deviation of radon transform at 20 evenly spaced points
def cubic_inter_std(img):
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)
    x_std = np.std(sinogram, axis=1)
    f = interpolate.interp1d(np.linspace(1, len(x_std), len(x_std)), x_std, kind='cubic')
    return f(np.linspace(1, len(x_std), 20)) / 100

# Computes distance of (x, y) to center of image
def cal_dist(img, x, y):
    return np.sqrt((x - img.shape[0] / 2)**2 + (y - img.shape[1] / 2)**2)

# Extracts geometric features from the largest labeled region
# Includes area, perimeter, centroid distance, axis lengths, eccentricity, solidity
def fea_geom(img):
    norm_area = img.shape[0] * img.shape[1]
    norm_perimeter = np.hypot(img.shape[0], img.shape[1])
    labels = measure.label(img, connectivity=1, background=0)
    labels_flat = labels[labels > 0]
    if labels_flat.size == 0:
        return [0] * 6
    mode_result = stats.mode(labels_flat, axis=None, keepdims=True)
    region_lbl = int(np.atleast_1d(mode_result.mode)[0])
    props = measure.regionprops(labels)
    region = props[min(region_lbl - 1, len(props) - 1)]
    return [
        region.area / norm_area,
        region.perimeter / norm_perimeter,
        cal_dist(img, *region.local_centroid),
        region.major_axis_length / norm_perimeter,
        region.minor_axis_length / norm_perimeter,
        region.eccentricity,
        region.solidity
    ]

# Combines all feature extractors into a single feature vector for a wafer map
def extract_features(img):
    return np.concatenate([
        find_regions(img),             # 13 region-based density features
        cubic_inter_mean(img),        # 20 interpolated mean radon features
        cubic_inter_std(img),         # 20 interpolated std radon features
        fea_geom(img)                 # 7 shape-based features
    ])
