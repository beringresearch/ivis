"""Custom datasets that load images from disk."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf

from .sequence import IndexableDataset


class ImageDataset(IndexableDataset):
    """When indexed, loads images from disk, resizes to consistent size, then returns image.
    Since the returned images will consist of 3 dimensions, the model ivis uses must be capable
    of dealing with this dimensionality of data (for example, a Convolutional Neural Network).
    Such a model can be constructed externally and then passed to ivis as the argument for 'model'.

    :param list filepath_list: All image filepaths in dataset.
    :param tuple img_shape: A tuple (height, width) containing desired dimensions to resize the
        images to.
    :param color_mode str: Either "rgb", "rgba" or "grayscale". Determines how many channels present
        in images that are read in - 3, 4, or 1 respectively.
    :param resize_method str: Interpolation method to use when resizing image. Must be one of:
        "area", "bicubic", "bilinear", "gaussian", "lanczos3", "lanczos5", "mitchellcubic", "nearest".
    :param preserve_aspect_ratio boolean: Whether to preserve the aspect ratio when resizing images.
        If True, will maintain aspect ratio by padding the image.
    :param dtype tf.dtypes.DType: The dtype to read the image into. One of tf.uint8 or tf.uint16.
    :param preprocessing_function Callable: A function to apply to every image. Will be called
        at the end of the pipeline, after image reading and resizing.
        If None (default), no function will be applied."""

    shape = None
    def __init__(self, filepath_list, img_shape, color_mode="rgb",
                 resize_method="bilinear", preserve_aspect_ratio=False,
                 dtype=tf.uint8, preprocessing_function=None, n_jobs=-1):
        self.filepath_list = np.array(filepath_list)
        self.img_shape = img_shape
        if color_mode == "rgb":
            self.channels = 3
        elif color_mode == "rgba":
            self.channels = 4
        elif color_mode == "grayscale":
            self.channels = 1
        else:
            raise ValueError("color_mode arg '{}' not recognized."
                             " Must be one of 'rgb', 'rgba', 'grayscale.".format(color_mode))
        self.resize_method = resize_method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.dtype = dtype
        self.preprocessing_function = preprocessing_function
        self.shape = (len(filepath_list), *img_shape, self.channels)
        self.n_jobs = n_jobs
        self.workers = None
        if n_jobs and os.cpu_count():
            # Negative worker counts wrap around to cpu core count, where -1 is one worker/core
            self.workers = ThreadPoolExecutor(
                max_workers=n_jobs if n_jobs > 0 else os.cpu_count() + n_jobs + 1)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return self.get_batch(range(start, stop, step))
        if idx < 0:
            idx += len(self)
        img = self.read_image(self.filepath_list[idx])
        img = self.resize_image(img)
        if self.preprocessing_function:
            img = self.preprocessing_function(img)
        return np.asarray(img)

    def __len__(self):
        return len(self.filepath_list)

    def read_image(self, filepath):
        """Reads an image from disk into a numpy array"""
        img_bytes = tf.io.read_file(filepath)
        return tf.image.decode_png(img_bytes, channels=self.channels, dtype=self.dtype)

    def resize_image(self, img):
        """Resizes an numpy array image to desired dimensions"""
        if not self.preserve_aspect_ratio:
            img = tf.image.resize(img, self.img_shape, method=self.resize_method)
        else:
            img = tf.image.resize_with_pad(img, *self.img_shape, method=self.resize_method)
        return tf.cast(img, self.dtype).numpy()

    def get_batch(self, idx_seq):
        """Returns a batch of data points based on the index sequence provided."""
        if self.workers is not None:
            return list(self.workers.map(self.__getitem__, idx_seq))
        return super().get_batch(idx_seq)

class FlattenedImageDataset(ImageDataset):
    """Returns flattened versions of images read in from disk. This dataset can be
    used with the default neighbour retrieval method used by ivis (Annoy KNN index)
    since it is 2D."""

    def __init__(self, filepath_list, img_shape, color_mode="rgb",
                 resize_method="bilinear", preserve_aspect_ratio=False,
                 dtype=tf.uint8, preprocessing_function=None, n_jobs=None):
        super().__init__(filepath_list, img_shape, color_mode, resize_method,
                         preserve_aspect_ratio, dtype, preprocessing_function,
                         n_jobs=n_jobs)
        self.shape = (self.shape[0], np.prod(self.shape[1:]))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return np.asarray(super().__getitem__(idx))
        return np.asarray(super().__getitem__(idx)).flatten()
