# wu_viz.py
"""
This utility module provides a reusable function for plotting confusion matrices.
Used across multiple training scripts to visualize classification performance.

Function:
- plot_confusion_matrix(): accepts a confusion matrix and class labels, 
  then renders a matplotlib plot with optional normalization.
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np

def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots a confusion matrix using Matplotlib.
    
    Parameters:
    - cm: (np.array) Confusion matrix (square array)
    - class_names: (list) Labels for classes in the matrix
    - normalize: (bool) If True, normalize each row to sum to 1
    - title: (str) Title for the plot
    - cmap: (matplotlib colormap) Color scheme for visualization
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'  # format numbers
    thresh = np.max(cm) / 2.0  # color threshold for better contrast

    # Loop through matrix cells to annotate
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Axis labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.yticks(ticks=range(len(class_names)), labels=class_names)
    plt.tight_layout()
