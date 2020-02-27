import numpy as np
import scipy.io as sio
from collections import namedtuple
from sklearn.preprocessing import OneHotEncoder


def get_ib_data():
    nb_classes = 2
    data = create_ib_data()

    x_train, c_train, x_test, c_test = data
    enc = OneHotEncoder()
    y_train = enc.fit_transform(c_train.reshape(-1, 1)).toarray()
    y_test = enc.fit_transform(c_test.reshape(-1, 1)).toarray()

    Dataset = namedtuple('Dataset', ['X', 'y', 'c', 'nb_classes'])
    train = Dataset(x_train, y_train, c_train, nb_classes)
    test = Dataset(x_test, y_test, c_test, nb_classes)
    del x_train, x_test, y_train, y_test
    return train, test


def create_ib_data():
    data_sets_org = load_data()
    data_sets = data_shuffle(data_sets_org, [80], shuffle_data=True)
    X_train, y_train, X_test, y_test = data_sets.train.data, data_sets.train.labels[:,0], data_sets.test.data, data_sets.test.labels[:,0]
    return X_train, y_train, X_test, y_test


def load_data():
    """Load the data
    name - the name of the dataset
    return object with data and labels
    """
    print ('Loading Data...')
    d = sio.loadmat('./dataset/var_u.mat')
    F = d['F']
    y = d['y']
    C = type('type_C', (object,), {})
    data_sets = C()
    data_sets.data = F
    data_sets.labels = np.squeeze(np.concatenate((y[None, :], 1 - y[None, :]), axis=0).T)
    return data_sets


def shuffle_in_unison_inplace(a, b):
    """Shuffle the arrays randomly"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def data_shuffle(data_sets_org, percent_of_train, shuffle_data=False):
    """Divided the data to train and test and shuffle it"""
    perc = lambda i, t: np.rint((i * t) / 100).astype(np.int32)
    C = type('type_C', (object,), {})
    data_sets = C()
    stop_train_index = perc(percent_of_train[0], data_sets_org.data.shape[0])
    start_test_index = stop_train_index
    # if percent_of_train > min_test_data:
    #     start_test_index = perc(min_test_data, data_sets_org.data.shape[0])
    data_sets.train = C()
    data_sets.test = C()
    if shuffle_data:
        shuffled_data, shuffled_labels = shuffle_in_unison_inplace(data_sets_org.data, data_sets_org.labels)
    else:
        shuffled_data, shuffled_labels = data_sets_org.data, data_sets_org.labels
    data_sets.train.data = shuffled_data[:stop_train_index, :]
    data_sets.train.labels = shuffled_labels[:stop_train_index, :]
    data_sets.test.data = shuffled_data[start_test_index:, :]
    data_sets.test.labels = shuffled_labels[start_test_index:, :]
    return data_sets


if '__main__' == __name__:
    trn, tst = get_ib_data()
    trn
