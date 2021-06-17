"""Supervised neighbour retrieval by label"""
from collections.abc import Sequence
import numpy as np


class LabeledNeighbourMap(Sequence):
    """Retrieves neighbour indices according to class labels provided in constructor.
    Rows with the same label will be regarded as neighbours."""
    def __init__(self, labels):
        """Constructs a LabeledNeighbourMap instance from a list of labels.
        :param labels list: List of labels for each data-point. One label per data-point."""
        self.class_indicies = {label: np.argwhere(labels == label).ravel()
                               for label in np.unique(labels)}
        self.labels = labels
    def __len__(self):
        """Returns the number of rows in the data"""
        return len(self.labels)
    def __getitem__(self, idx):
        """Retrieves the neighbours for the row index provided"""
        return self.class_indicies[self.labels[idx]]
