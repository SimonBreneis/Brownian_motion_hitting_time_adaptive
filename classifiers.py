import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras
import utilities as util


def shallow_binary_NN(signatures, labels, lr=1e-05, validation_split=1./3, batch_size=1000, epochs=1000, verbose=2):
    model = Sequential([
        Dense(2, input_shape=(signatures.shape[1],), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.optimizer = keras.optimizers.Adam()
    model.optimizer.lr = lr
    model.fit(x=signatures, y=labels, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
              shuffle=True, verbose=verbose)


def deep_binary_NN(signatures, labels, lr=1e-06, validation_split=1./3, batch_size=1000, epochs=1000, verbose=2):
    model = Sequential([
        Dense(signatures.shape[1]+1, input_shape=(signatures.shape[1],), activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.optimizer = keras.optimizers.Adam()
    model.optimizer.lr = lr
    model.fit(x=signatures, y=labels, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
              shuffle=True, verbose=verbose)


def deeper_binary_NN(signatures, labels, lr=1e-06, validation_split=1. / 3, batch_size=1000, epochs=1000, verbose=2):
    model = Sequential([
        Dense(signatures.shape[1] + 1, input_shape=(signatures.shape[1],), activation='relu'),
        Dense(signatures.shape[1]+1, activation='relu'),
        Dense(signatures.shape[1]+1, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.optimizer = keras.optimizers.Adam()
    model.optimizer.lr = lr
    model.fit(x=signatures, y=labels, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
              shuffle=True, verbose=verbose)


def shallow_binary_NN_LASSO(signatures, labels, lr=1e-05, validation_split=1. / 3, batch_size=1000,
                                     epochs=1000, verbose=2, reg=0.5, n_important=9, n_lasso=100):
    importance_counter = np.zeros(shape=(signatures.shape[0],), dtype=int)

    for i in range(n_lasso):
        print(f"We are in the {i}th run of LASSO.")
        model = Sequential([
            Dense(2, input_shape=(signatures.shape[1],), activation='softmax',
                  kernel_regularizer=keras.regularizers.l1(reg))
        ])

        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.optimizer = keras.optimizers.Adam()
        model.optimizer.lr = lr
        model.fit(x=signatures, y=labels, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                  shuffle=True, verbose=0)
        for layer in model.layers:
            absolute_values = np.sum(np.fabs(layer.get_weights()[0]), axis=1)
            important_indices = np.argpartition(absolute_values, -n_important)[-n_important:]
            for index in important_indices:
                importance_counter[index] += 1

    for i in range(101):
        print(f'There are {len(importance_counter[importance_counter >= i])} indices that were selected at least {i} times.')

    print('The 20 most important indices together with their frequencies are: ')
    indices_by_importance = np.argsort(importance_counter)
    for i in range(1, 21):
        print(f'Index {indices_by_importance[-i]} occurred {importance_counter[indices_by_importance[-i]]} times.')

    unimportant_indices = np.delete(range(signatures.shape[1]), indices_by_importance[-n_important:])
    signatures = np.delete(signatures, unimportant_indices, axis=1)

    shallow_binary_NN(signatures, labels, 10*lr, validation_split, batch_size, epochs, verbose)


def RF(signatures, labels, n_estimators=1000, validation_split=1./3, verbose=2):
    x_train, x_test, y_train, y_test = train_test_split(signatures, labels, test_size=validation_split, random_state=42)

    rf = RandomForestClassifier(n_estimators=n_estimators, verbose=verbose)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    print(confusion_matrix(y_test, predictions))


def linear_regression(signatures, labels, validation_split=1./3):
    x_train, x_test, y_train, y_test = train_test_split(signatures, labels, test_size=validation_split, random_state=42)

    model = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000).fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(confusion_matrix(y_test, predictions))


def BS_find_q(n_train=60000, n_test=40000, degs=np.array([1, 2, 4, 8, 16, 32, 64, 128]), time_steps=1000):
    # Finding good values for q in the theorem on hitting [e,infinity) of the BS model

    x = np.zeros(shape=(n_train + n_test, len(degs)))
    y = np.zeros(shape=(n_train + n_test,))

    for i in range(n_train + n_test):
        brownian_path = util.brownian_motion(1., time_steps)
        path = np.exp(brownian_path)
        y[i] = float(util.check_hitting(path, np.exp(1)))
        for j in range(len(degs)):
            x[i, j] = np.minimum(np.trapz(path ** degs[j], dx=1. / time_steps) / np.exp(degs[j]), time_steps)

    for j in range(len(degs)):
        model = Sequential([
            Dense(2, input_shape=(1,), activation='softmax', kernel_initializer=keras.initializers.zeros())
        ])

        model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.optimizer = keras.optimizers.Adam()
        model.optimizer.lr = 0.001
        model.fit(x=x[:, j], y=y, validation_split=n_test / float(n_train), batch_size=10, epochs=100, shuffle=True,
                  verbose=2)
        print(degs[j])
        print(model.weights)
