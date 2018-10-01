from __future__ import print_function, division

import numpy as np

from keras.datasets import mnist, cifar10, cifar100, imdb
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.core import Dropout, SpatialDropout1D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import sequence
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from resnet import cifar10_resnet
from loss import (crossentropy, robust, unhinged, sigmoid, ramp, savage,
                  boot_soft)


# losses that need sigmoid on top of last layer
yes_softmax = ['crossentropy', 'forward', 'est_forward', 'backward',
               'est_backward', 'boot_soft', 'savage']
# unhinged needs bounded models or it diverges
yes_bound = ['unhinged', 'ramp', 'sigmoid']


class KerasModel():

    def get_data(self):

        (X_train, y_train), (X_test, y_test) = self.load_data()

        idx_perm = np.random.RandomState(101).permutation(X_train.shape[0])
        X_train, y_train = X_train[idx_perm], y_train[idx_perm]

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        return X_train, X_test, y_train, y_test

    # custom losses for the CNN
    def make_loss(self, loss, P=None):

        if loss == 'crossentropy':
            return crossentropy
        elif loss in ['forward', 'backward']:
            return robust(loss, P)
        elif loss == 'unhinged':
            return unhinged
        elif loss == 'sigmoid':
            return sigmoid
        elif loss == 'ramp':
            return ramp
        elif loss == 'savage':
            return savage
        elif loss == 'boot_soft':
            return boot_soft
        else:
            ValueError("Loss unknown.")

    def compile(self, model, loss, P=None):

        if self.optimizer is None:
            ValueError()

        metrics = ['accuracy']

        model.compile(loss=self.make_loss(loss, P),
                      optimizer=self.optimizer, metrics=metrics)

        model.summary()
        self.model = model

    def load_model(self, file):
        self.model.load_weights(file)
        print('Loaded model from %s' % file)

    def fit_model(self, model_file, X_train, Y_train, validation_split=None,
                  validation_data=None):

        # cannot do both
        if validation_data is not None and validation_split is not None:
            return ValueError()

        callbacks = []
        monitor = 'val_loss'
        # monitor = 'val_acc'

        mc_callback = ModelCheckpoint(model_file, monitor=monitor,
                                      verbose=1, save_best_only=True)
        callbacks.append(mc_callback)

        if hasattr(self, 'scheduler'):
            callbacks.append(self.scheduler)

        # use data augmentation
        if hasattr(self, 'data_generator'):

            # hack for using validation with data augmentation
            idx_val = np.round(validation_split * X_train.shape[0]).astype(int)
            X_val, Y_val = X_train[:idx_val], Y_train[:idx_val]
            X_train_local, Y_train_local = X_train[idx_val:], Y_train[idx_val:]

            self.data_generator.fit(X_train_local)

            history = \
                self.model.fit_generator(
                    self.data_generator.flow(X_train_local, Y_train_local,
                                             batch_size=self.num_batch),
                    steps_per_epoch=X_train.shape[0] // self.num_batch,
                    epochs=self.epochs, max_q_size=100,
                    validation_data=(X_val, Y_val),
                    verbose=1, callbacks=callbacks)

        else:

            history = self.model.fit(
                        X_train, Y_train, batch_size=self.num_batch,
                        epochs=self.epochs,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        verbose=1, callbacks=callbacks)

        # use the model that reached the lowest loss at training time
        self.load_model(model_file)

        return history.history

    def evaluate_model(self, X, Y):
        score = self.model.evaluate(X, Y, batch_size=self.num_batch, verbose=1)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score[1]

    def predict_proba(self, X):
        pred = self.model.predict(X, batch_size=self.num_batch, verbose=1)
        return pred


class MNISTModel(KerasModel):

    def __init__(self, num_batch=32):
        self.num_batch = num_batch
        self.classes = 10
        self.epochs = 40
        self.normalize = True
        self.optimizer = None

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)

        if self.normalize:
            X_train = X_train / 255.
            X_test = X_test / 255.

        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss, P=None):

        input = Input(shape=(784,))

        x = Dense(128, kernel_initializer='he_normal')(input)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(10, kernel_initializer='he_normal')(x)

        if loss in yes_bound:
            output = BatchNormalization(axis=1)(output)

        if loss in yes_softmax:
            output = Activation('softmax')(output)

        model = Model(inputs=input, outputs=output)
        self.compile(model, loss, P)


class CIFAR10Model(KerasModel):

    def __init__(self, num_batch=32, type='deep'):
        self.num_batch = num_batch
        self.classes = 10
        self.img_channels = 3
        self.img_rows = 32
        self.img_cols = 32
        self.filters = 32
        self.num_pool = 2
        self.num_conv = 3
        self.type = type

        self.epochs = 120
        self.augmentation = True
        self.optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0)
        self.lr_scheduler()
        self.decay = 0.0001

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        X_train = X_train.reshape(X_train.shape[0], self.img_rows,
                                  self.img_cols, self.img_channels)
        X_test = X_test.reshape(X_test.shape[0], self.img_rows, self.img_cols,
                                self.img_channels)

        means = X_train.mean(axis=0)
        X_train = (X_train - means)
        X_test = (X_test - means)

        if self.augmentation:

            print('Data Augmentation')

            # data augmentation
            self.data_generator = \
                ImageDataGenerator(
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        horizontal_flip=True)

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        return (X_train, y_train), (X_test, y_test)

    def lr_scheduler(self):

        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1

        print('LR scheduler')
        self.scheduler = LearningRateScheduler(scheduler)

    def build_model(self, loss, P=None):

        if self.type[:-1] == 'resnet':
            model = cifar10_resnet(int(self.type[-1]), self, self.decay, loss)

        self.compile(model, loss, P)


class CIFAR100Model(KerasModel):

    def __init__(self, num_batch=32):
        self.num_batch = num_batch
        self.classes = 100  # 100 classes
        self.img_channels = 3
        self.img_rows = 32
        self.img_cols = 32
        self.filters = 32
        self.num_pool = 2
        self.num_conv = 3

        self.epochs = 150
        self.augmentation = True
        self.optimizer = SGD(lr=0.1, momentum=0.9, decay=0.0)
        self.decay = 10 ** -3
        self.lr_scheduler()

    def lr_scheduler(self):

        def scheduler(epoch):
            if epoch > 120:
                return 0.001
            elif epoch > 80:
                return 0.01
            else:
                return 0.1

        print('LR scheduler')
        self.scheduler = LearningRateScheduler(scheduler)

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()
        X_train = X_train.reshape(X_train.shape[0], self.img_rows,
                                  self.img_cols, self.img_channels)
        X_test = X_test.reshape(X_test.shape[0], self.img_rows, self.img_cols,
                                self.img_channels)

        means = X_train.mean(axis=0)
        # std = np.std(X_train)
        X_train = (X_train - means)  # / std
        X_test = (X_test - means)  # / std

        if self.augmentation:

            print('Data Augmentation')

            # data augmentation
            self.data_generator = \
                ImageDataGenerator(
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        horizontal_flip=True)

        # they are 2D originally in cifar
        y_train = y_train.ravel()
        y_test = y_test.ravel()

        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss, P=None):

        model = cifar10_resnet(7, self, self.decay, loss)
        self.compile(model, loss, P)


class IMDBModel(KerasModel):

    def __init__(self, num_batch=32):
        self.num_batch = num_batch
        self.max_features = 5000
        self.maxlen = 400
        self.embedding_dims = 50
        self.hidden_dims = 256
        self.epochs = 50
        self.classes = 2
        self.optimizer = None

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = \
            imdb.load_data(num_words=self.max_features, seed=11)

        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)

        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss, P=None):

        input = Input(shape=(self.maxlen,))

        x = Embedding(self.max_features, self.embedding_dims)(input)
        x = SpatialDropout1D(0.8)(x)
        x = Activation('relu')(x)

        x = Flatten()(x)
        output = Dense(self.classes, kernel_initializer='he_normal')(x)

        if loss in yes_bound:
            output = BatchNormalization(axis=1)(output)

        if loss in yes_softmax:
            output = Activation('softmax')(output)

        model = Model(inputs=input, outputs=output)
        self.compile(model, loss, P)


class LSTMModel(KerasModel):

    def __init__(self, num_batch=32):
        self.num_batch = num_batch
        self.max_features = 5000
        self.maxlen = 400
        self.embedding_dims = 512
        self.lstm_dim = 512
        self.hidden_dims = 128
        self.epochs = 50
        self.classes = 2
        self.optimizer = None

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = \
            imdb.load_data(num_words=self.max_features, seed=11)

        X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)

        return (X_train, y_train), (X_test, y_test)

    def build_model(self, loss, P=None):

        input = Input(shape=(self.maxlen,))

        x = Embedding(self.max_features, self.embedding_dims)(input)
        x = SpatialDropout1D(0.8)(x)

        x = LSTM(self.lstm_dim, kernel_initializer='uniform')(x)

        x = Dense(self.embedding_dims, kernel_initializer='he_normal')(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)

        output = Dense(self.classes, kernel_initializer='he_normal')(x)

        if loss in yes_bound:
            output = BatchNormalization(axis=1)(output)

        if loss in yes_softmax:
            output = Activation('softmax')(output)

        model = Model(inputs=input, outputs=output)
        self.compile(model, loss, P)


class NoiseEstimator():

    def __init__(self, classifier, row_normalize=True, alpha=0.0,
                 filter_outlier=False, cliptozero=False, verbose=0):
        """classifier: an ALREADY TRAINED model. In the ideal case, classifier
        should be powerful enough to only make mistakes due to label noise."""

        self.classifier = classifier
        self.row_normalize = row_normalize
        self.alpha = alpha
        self.filter_outlier = filter_outlier
        self.cliptozero = cliptozero
        self.verbose = verbose

    def fit(self, X):

        # number of classes
        c = self.classifier.classes
        T = np.empty((c, c))

        # predict probability on the fresh sample
        eta_corr = self.classifier.predict_proba(X)

        # find a 'perfect example' for each class
        for i in np.arange(c):

            if not self.filter_outlier:
                idx_best = np.argmax(eta_corr[:, i])
            else:
                eta_thresh = np.percentile(eta_corr[:, i], 97,
                                           interpolation='higher')
                robust_eta = eta_corr[:, i]
                robust_eta[robust_eta >= eta_thresh] = 0.0
                idx_best = np.argmax(robust_eta)

            for j in np.arange(c):
                T[i, j] = eta_corr[idx_best, j]

        self.T = T
        return self

    def predict(self):

        T = self.T
        c = self.classifier.classes

        if self.cliptozero:
            idx = np.array(T < 10 ** -6)
            T[idx] = 0.0

        if self.row_normalize:
            row_sums = T.sum(axis=1)
            T /= row_sums[:, np.newaxis]

        if self.verbose > 0:
            print(T)

        if self.alpha > 0.0:
            T = self.alpha * np.eye(c) + (1.0 - self.alpha) * T

        if self.verbose > 0:
            print(T)
            print(np.linalg.inv(T))

        return T
