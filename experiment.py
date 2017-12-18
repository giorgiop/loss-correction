from __future__ import print_function
import sys
import os
import getopt
import pickle

import numpy as np

from keras.utils import to_categorical
from keras.optimizers import Adagrad
from keras import backend as K
from models import noisify_with_P
from models import noisify_binary_asymmetric
from models import noisify_cifar10_asymmetric, noisify_mnist_asymmetric
from models import noisify_cifar100_asymmetric
from models import MNISTModel, CIFAR10Model, CIFAR100Model
from models import IMDBModel, LSTMModel
from models import NoiseEstimator

np.random.seed(1337)  # for reproducibility


def build_file_name(loc, dataset, loss, noise, asymmetric, run):

    return (os.path.dirname(os.path.realpath(__file__)) +
            '/output/' + loc +
            dataset + '_' +
            loss + '_' +
            str(noise) + '_' +
            str(asymmetric) + '_' +
            str(run))

def train_and_evaluate(dataset, loss, noise, run=0, num_batch=32,
                       asymmetric=0):

    val_split = 0.1

    if dataset == 'mnist':
        kerasModel = MNISTModel(num_batch=num_batch)
        kerasModel.optimizer = Adagrad()
    elif dataset == 'cifar10_deep':
        kerasModel = CIFAR10Model(num_batch=num_batch, type='deep')
    elif dataset[8:-1] == 'resnet':
        kerasModel = CIFAR10Model(num_batch=num_batch, type=dataset[8:])
    elif dataset == 'cifar100':
        kerasModel = CIFAR100Model(num_batch=num_batch)
    elif dataset == 'imdb':
        kerasModel = IMDBModel(num_batch=num_batch)
        kerasModel.optimizer = Adagrad()
    elif dataset == 'lstm':
        kerasModel = LSTMModel(num_batch=num_batch)
        kerasModel.optimizer = Adagrad(lr=0.001)
    else:
        ValueError('No dataset given.')
        import sys
        sys.exit()

    # an important data-dependent configuration
    if dataset == 'cifar100':
        filter_outlier = False
    else:
        filter_outlier = True

    # the data, shuffled and split between train and test sets
    print('Loading %s ...' % dataset)
    X_train, X_test, y_train, y_test = kerasModel.get_data()
    print('Done.')

    # apply label noise
    if asymmetric == 0:
        y_train, P = noisify_with_P(y_train, kerasModel.classes, noise,
                                    random_state=run)
    elif asymmetric == 1:
        if dataset == 'mnist':
            y_train, P = noisify_mnist_asymmetric(y_train, noise,
                                                  random_state=run)
        elif dataset == 'cifar100':
            y_train, P = noisify_cifar100_asymmetric(y_train, noise,
                                                     random_state=run)
        elif dataset[:7] == 'cifar10':
            y_train, P = noisify_cifar10_asymmetric(y_train, noise,
                                                    random_state=run)
        else:  # binary classes
            y_train, P = noisify_binary_asymmetric(y_train, noise,
                                                   random_state=run)

    print('T: \n', P)

    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, kerasModel.classes)
    Y_test = to_categorical(y_test, kerasModel.classes)

    # keep track of the best model
    model_file = build_file_name('tmp_model/', dataset, loss, noise,
                                 asymmetric, run)

    # this is the case when we post-train changing the loss
    if loss == 'est_backward':

        vanilla_file = build_file_name('tmp_model/', dataset, 'crossentropy',
                                       noise, asymmetric, run)

        if not os.path.isfile(vanilla_file):
            ValueError('Need to train with crossentropy first !')

        # first compile the vanilla_crossentropy model with the saved weights
        kerasModel.build_model('crossentropy', P=None)
        kerasModel.load_model(vanilla_file)

        # estimate P
        est = NoiseEstimator(classifier=kerasModel, alpha=0.0,
                             filter_outlier=filter_outlier)

        # use all X_train
        P_est = est.fit(X_train).predict()
        print('Condition number:', np.linalg.cond(P_est))
        print('T estimated: \n', P)

        # compile the model with the new estimated loss
        kerasModel.build_model('backward', P=P_est)

    elif loss == 'est_forward':
        vanilla_file = build_file_name('tmp_model/', dataset, 'crossentropy',
                                       noise, asymmetric, run)

        if not os.path.isfile(vanilla_file):
            ValueError('Need to train with crossentropy first !')

        # first compile the vanilla_crossentropy model with the saved weights
        kerasModel.build_model('crossentropy', P=None)
        kerasModel.load_model(vanilla_file)

        # estimate P
        est = NoiseEstimator(classifier=kerasModel, alpha=0.0,
                             filter_outlier=filter_outlier)
        # use all X_train
        P_est = est.fit(X_train).predict()
        print('T estimated:', P)

        # compile the model with the new estimated loss
        kerasModel.build_model('forward', P=P_est)

    else:
        # compile the model
        kerasModel.build_model(loss, P)

    # fit the model
    history = kerasModel.fit_model(model_file, X_train, Y_train,
                                   validation_split=val_split)

    # history_file = build_file_name('history/', dataset, loss,
    #                                noise, asymmetric, run)

    # decomment for writing history
    # with open(history_file, 'wb') as f:
    #    pickle.dump(history, f)
    #    print('History dumped at ' + str(history_file))

    # LOAD THE BEST MODEL FOUND -- this should be useless as it is called twice
    # inside fit_model
    # kerasModel.load_model(model_file)

    # test
    score = kerasModel.evaluate_model(X_test, Y_test)

    # clean models, unless it is vanilla_crossentropy --to be used by P_est
    if loss != 'crossentropy':
        os.remove(model_file)

    return score


if __name__ == "__main__":

    def error_and_exit():
        print('Usage: ' + str(__file__) + ' -d dataset -l loss '
              '-n noise_rate [-a asymmetric_noise -r n_runs]')
        sys.exit()

    opts, args = getopt.getopt(sys.argv[1:], "d:l:n:a:r:")

    # dataset/model cases: mnist, imdb, lstm, cifa100
    # cifar10_resnet{n} with a digit. See the call to cifar10 models

    # loss: crossentropy, backward, forward, unhinged, sigmoid, ramp, savage, boot_soft

    n_runs = 1
    num_batch = 128
    asymmetric = 0  # symmetric noise by default

    dataset, loss, noise = None, None, None

    for opt, arg in opts:
        if opt == '-d':
            dataset = arg
        elif opt == '-l':
            loss = arg
        elif opt == '-n':
            noise = np.array(arg).astype(np.float)
        elif opt == '-a':
            asymmetric = np.array(arg).astype(np.int)
        elif opt == '-r':
            n_runs = np.array(arg).astype(np.int)
        else:
            error_and_exit()

    # compulsory params
    if dataset is None or loss is None or noise is None:
        error_and_exit()

    print("Params: dataset=%s, loss=%s, noise=%s, asymmetric=%d, "
          % (dataset, loss, noise, asymmetric))

    accuracies = []
    # implicit random initialization
    for i in range(n_runs):
        accuracies.append(train_and_evaluate(dataset, loss, noise, i,
                                             num_batch, asymmetric))
        print("*** # RUN %d: accuracy=%.2f" % (i, accuracies[i]))

    print(accuracies)
    print(np.mean(accuracies), np.std(accuracies))

    K.clear_session()
