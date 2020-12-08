"""A collection of callbacks that can be passed to ivis to be called during training.
These provide utilities such as saving checkpoints during training (allowing for
resuming if interrupted), as well as periodic logging of plots and model embeddings.
With this information, you may decide to terminate a training session early
due to a lack of improvements to the visualizations, for example.

To use a callback during training, simply pass a list of callback objects
to the Ivis object when creating it using the callbacks keyword argument.
The ivis.nn.callbacks module contains a set of callbacks provided for use with ivis models,
but any `tf.keras.callbacks.Callbacks` object can be passed and will be used during training:
for example, `tf.keras.callbacks.TensorBoard`. This means it's also possible to write your own
callbacks for ivis to use.
"""

import io
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler

# Matplotlib and seaborn are optional dependencies
try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None
try:
    import seaborn as sns
except ImportError:
    sns = None


class ModelCheckpoint(Callback):
    """Periodically saves the model during training. By default, it saves the
    model every epoch; increasing the ``epoch_interval`` will make
    checkpointing less frequent.

    If the given filename contains the ``{}`` string, the epoch number will be
    subtituted in, resulting in multiple checkpoint folders with different
    names. If a filename such as 'ivis-checkpoint' is provided, only the
    latest checkpoint will be kept.

    :param str log_dir: Folder to save resulting embeddings.
    :param str filename: Filename to save each file as. `{}` in string
        will be substituted with the epoch number.

    Example usage:
    ::

        from ivis.nn.callbacks import ModelCheckpoint
        from ivis import Ivis

        # Save only the latest checkpoint to current directory every 10 epochs
        checkpoint_callback = ModelCheckpoint(log_dir='.',
                                            filename='latest-checkpoint.ivis',
                                            epoch_interval=10)

        model = Ivis(callbacks=[checkpoint_callback])
    """

    def __init__(self, log_dir='./model_checkpoints',
                 filename='model-checkpoint_{}.ivis', epoch_interval=1):
        super(ModelCheckpoint, self).__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.epochs_since_last_save = 0
        self.ivis_model = None

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epoch_interval:
            self.epochs_since_last_save = 0
            filename = self.filename.format(epoch + 1)
            self.ivis_model.save_model(
                os.path.join(self.log_dir, filename), overwrite=True)

    def register_ivis_model(self, ivis_model):
        self.ivis_model = ivis_model


class EmbeddingsLogging(Callback):
    """Periodically saves embeddings of the data provided to ``data``
    using the latest state of the ``Ivis`` model.
    By default, saves embeddings every epoch; increasing the
    ``epoch_interval`` will save the embeddings less frequently.

    :param list[float] data: Data to embed with the latest Ivis object
    :param str log_dir: Folder to save resulting embeddings.
    :param str filename: Filename to save each file as. `{}` in string
        will be substituted with the epoch number.

    Example usage:
    ::

        from ivis.nn.callbacks import EmbeddingsLogging
        from ivis import Ivis
        from tensorflow.keras.datasets import mnsit

        (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()

        # Save embeddings of test set every epoch
        embeddings_callback = EmbeddingsLogging(X_test,
                                                log_dir='test-embeddings',
                                                filename='{}_test_embeddings.npy',
                                                epoch_interval=1)

        model = Ivis(callbacks=[embeddings_callback])

        # Train on training set
        model.fit(X_train)
    """

    def __init__(self, data, log_dir='./embeddings_logs',
                 filename='{}_embeddings.npy', epoch_interval=1):
        super(EmbeddingsLogging, self).__init__()
        self.data = data
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.epochs_since_last_save = 0
        self.embeddings = None

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epoch_interval:
            self.epochs_since_last_save = 0
            self.embeddings = self.model.layers[3].predict(self.data)
            filename = self.filename.format(epoch + 1)
            np.save(os.path.join(self.log_dir, filename), self.embeddings)


class EmbeddingsImage(Callback):
    """Periodically generates and plots 2D embeddings of the data
    provided to ``data`` using the latest state of the ``Ivis`` model.
    By default, saves plots of the embeddings every epoch; increasing the
    ``epoch_interval`` will save the plots less frequently.

    :param list[float] data: Data to embed and plot with the latest Ivis model
    :param list[int] labels: Labels with which to colour plotted
        embeddings. If `None` all points will have the same color.
    :param str log_dir: Folder to save resulting embeddings.
    :param str filename: Filename to save each file as. `{}` in string
        will be substituted with the epoch number.

    Example usage:
    ::

        from ivis.nn.callbacks import EmbeddingsImage
        from ivis import Ivis
        from tensorflow.keras.datasets import mnsit

        (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()

        # Plot embeddings of test set every epoch colored by labels
        embeddings_callback = EmbeddingsImage(X_test, Y_test,
                                                log_dir='test-embeddings',
                                                filename='{}_test_embeddings.npy',
                                                epoch_interval=1)

        model = Ivis(callbacks=[embeddings_callback])

        # Train on training set
        model.fit(X_train)
    """

    def __init__(self, data, labels=None, log_dir='./logs',
                 filename='{}_embeddings.png', epoch_interval=1):
        super(EmbeddingsImage, self).__init__()
        _check_visualization_libraries()

        self.data = data
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.zeros(len(data))
        self.n_classes = len(np.unique(self.labels, axis=0))
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.epochs_since_last_save = 0
        self.embeddings = None

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epoch_interval:
            self.epochs_since_last_save = 0
            self.embeddings = self.model.layers[3].predict(self.data)
            filename = self.filename.format(epoch + 1)
            self.plot_embeddings(filename)

    def plot_embeddings(self, filename):
        embeddings = MinMaxScaler((0, 1)).fit_transform(self.embeddings)

        fig = plt.figure()
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], s=1,
                        hue=self.labels,
                        palette=sns.color_palette("hls", self.n_classes),
                        linewidth=0)

        plt.savefig(os.path.join(self.log_dir, filename), dpi=300)
        plt.close(fig)


class TensorBoardEmbeddingsImage(Callback):
    """Periodically generates and plots 2D embeddings of the data
    provided to ``data`` using the latest state of the ``Ivis`` model.
    The plots are designed to be viewed in Tensorboard, which will provide
    an image that shows the history of embeddings plots through training.
    By default, saves plots of the embeddings every epoch; increasing the
    ``epoch_interval`` will save the plots less frequently.

    :param list[float] data: Data to embed and plot with the latest Ivis
    :param list[int] labels: Labels with which to colour plotted
        embeddings. If `None` all points will have the same color.
    :param str log_dir: Folder to save resulting embeddings.
    :param str filename: Filename to save each file as. `{}` in string
        will be substituted with the epoch number.

    Example usage:
    ::

        from ivis.nn.callbacks import TensorBoardEmbeddingsImage
        from ivis import Ivis
        from tensorflow.keras.datasets import mnsit

        (X_train, Y_train), (X_test, Y_test)  = mnist.load_data()

        # Plot embeddings of test set every epoch colored by labels
        embeddings_callback = TensorBoardEmbeddingsImage(X_test, Y_test,
                                                log_dir='test-embeddings',
                                                filename='{}_test_embeddings.npy',
                                                epoch_interval=1)

        model = Ivis(callbacks=[embeddings_callback])

        # Train on training set
        model.fit(X_train)
    """

    def __init__(self, data, labels=None,
                 log_dir='./logs', epoch_interval=1):
        super(TensorBoardEmbeddingsImage, self).__init__()
        _check_visualization_libraries()

        self.data = data
        self.log_dir = log_dir
        if labels is not None:
            self.labels = labels
        else:
            self.labels = np.zeros(len(data))
        self.n_classes = len(np.unique(self.labels, axis=0))
        self.epochs_since_last_save = 0
        self.epoch_interval = epoch_interval
        self.embeddings = None
        self.file_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'embeddings')
        )

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epoch_interval:
            self.embeddings = self.model.layers[3].predict(self.data)
            image = self.plot_embeddings(self.embeddings)
            with self.file_writer.as_default():
                tf.summary.image("Embeddings", image, step=epoch)

    def plot_embeddings(self, embeddings):
        embeddings = MinMaxScaler((0, 1)).fit_transform(self.embeddings)

        fig = plt.figure()
        buf = io.BytesIO()
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], s=1,
                        hue=self.labels,
                        palette=sns.color_palette("hls", self.n_classes),
                        linewidth=0)

        plt.savefig(buf, format='png', dpi=300)
        plt.close(fig)
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image


def _check_visualization_libraries():
    if plt is None:
        raise ImportError(
            'Failed to import `matplotlib`.'
            'To use visualization callbacks install `matplotlib`.'
            'For example: pip install matplotlib'
        )
    if sns is None:
        raise ImportError(
            'Failed to import `seaborn`.'
            'To use visualization callbacks install `seaborn`.'
            'For example: pip install seaborn'
        )
