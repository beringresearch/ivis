from keras.callbacks import Callback
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import io
import tensorflow as tf
import os


class ModelCheckpoint(Callback):
    def __init__(self, log_dir='./model_checkpoints',
                 filename='model-checkpoint_{}.ivis', epoch_interval=1):
        super(ModelCheckpoint, self).__init__()
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.epochs_since_last_save = 0

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
    def __init__(self, batch, log_dir='./embeddings_logs',
                 filename='{}_embeddings.npy', epoch_interval=1):
        super(EmbeddingsLogging, self).__init__()
        self.batch = batch
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epoch_interval:
            self.epochs_since_last_save = 0
            self.embeddings = self.model.layers[3].predict(self.batch)
            filename = self.filename.format(epoch + 1)
            np.save(os.path.join(self.log_dir, filename), self.embeddings)


class EmbeddingsImage(Callback):
    def __init__(self, batch, batch_labels=None, log_dir='./logs',
                 filename='{}_embeddings.png', epoch_interval=1):
        super(EmbeddingsImage, self).__init__()
        self.batch = batch
        if batch_labels is not None:
            self.batch_labels = batch_labels
        else:
            self.batch_labels = np.zeros(len(batch))
        self.n_classes = len(np.unique(self.batch_labels, axis=0))
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.filename = filename
        self.epoch_interval = epoch_interval
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.epoch_interval:
            self.epochs_since_last_save = 0
            self.embeddings = self.model.layers[3].predict(self.batch)
            filename = self.filename.format(epoch + 1)
            self.plot_embeddings(filename)

    def plot_embeddings(self, filename):
        embeddings = MinMaxScaler((0, 1)).fit_transform(self.embeddings)

        fig = plt.figure()
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], s=1,
                        hue=self.batch_labels,
                        palette=sns.color_palette("hls", self.n_classes),
                        linewidth=0)

        plt.savefig(os.path.join(self.log_dir, filename), dpi=300)
        plt.close(fig)
