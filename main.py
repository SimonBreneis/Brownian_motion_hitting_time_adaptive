import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization
from keras import optimizers
# from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from sklearn.ensemble import RandomForestClassifier


def brownian_motion(T, N):
    normals = np.random.normal(0, np.sqrt(float(T) / N), N)
    normals = np.concatenate((np.array([0.]), normals))
    return np.cumsum(normals)


def brownian_time(T, N):
    time = np.array([i * T / N for i in range(0, N + 1)]).reshape(N + 1, 1)
    brownian = brownian_motion(T, N).reshape(N + 1, 1)
    return np.concatenate((time, brownian), axis=1)


def check_hitting(path, threshold):
    return np.amax(path) >= threshold


def integrate_against_indefinite(x, y_diff):
    """
    Computes the indefinite integral int x dy.
    :param x: Vector of x-values
    :param y_diff: Vector of y-increments
    :return: The vector of definite integrals
    """
    x_midpoint = (x[:, :-1] + x[:, 1:])/2
    integral_increments = x_midpoint * y_diff
    indefinite_integral = np.zeros(shape=x.shape)
    indefinite_integral[:, 1:] = np.cumsum(integral_increments, axis=-1)
    return indefinite_integral


def get_average_accuracy_nn(train_features, train_labels, test_features, test_labels, number_nn=1, lr=0.0001, batch_size=None, epochs=100, activation='softmax', loss='sparse_categorical_crossentropy'):
    if batch_size is None:
        batch_size = int(np.sqrt(train_features.shape[0]))
    accuracies = np.empty(shape=(number_nn,))
    for nn in range(0, number_nn):
        model = Sequential([
            Dense(2, input_shape=(train_features.shape[1],), activation=activation)
        ])

        model.compile(loss=loss, metrics=['accuracy'])
        model.optimizer = keras.optimizers.Adam()
        model.optimizer.lr = lr
        model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, shuffle=True,
                  verbose=0)
        predictions = np.around(model.predict(x=test_features)[:, 1])
        predictions_correct = (predictions == test_labels)
        accuracies[nn] = float(np.sum(predictions_correct)) / len(predictions_correct)
    return np.average(accuracies)


def get_average_accuracy_rf(train_features, train_labels, test_features, test_labels, number_rf=1, number_trees=100):
    accuracies = np.empty(shape=(number_rf,))
    for i in range(0, number_rf):
        rf = RandomForestClassifier(n_estimators=number_trees, verbose=0)
        rf.fit(train_features, train_labels)
        predictions = np.around(rf.predict(test_features))
        predictions_correct = (predictions == test_labels)
        accuracies[i] = float(np.sum(predictions_correct)) / len(predictions_correct)
    return np.average(accuracies)


class SignatureTree:
    def __init__(self, train_paths, validation_paths, train_labels, validation_labels, initial_signature_level):
        # A family of paths is given as [ [ [path1dim1value1, path1dim1value2, ...], [path1dim2value1, ...], ...], ...]
        self.train_increments = train_paths[:, :, 1:] - train_paths[:, :, :-1]
        self.validation_increments = validation_paths[:, :, 1:] - validation_paths[:, :, :-1]
        self.test_increments = None
        self.train_labels = train_labels
        self.validation_labels = validation_labels
        self.test_labels = None
        self.path_dim = train_paths.shape[1]
        self.path_len = train_paths.shape[2]
        self.signature_nodes = []
        self.signature_nodes.append(SignatureNode(train_values=np.ones(shape=(train_paths.shape[0], self.path_len)),
                                                  validation_values=np.ones(
                                                      shape=(validation_paths.shape[0], self.path_len)),
                                                  root=self, predecessor=None, index=[], is_active=True))
        for signature_node in self.signature_nodes:
            if signature_node.get_level() < initial_signature_level:
                signature_node.compute_successors()
                successors = signature_node.get_all_successors()
                for successor in successors:
                    self.add_signature_node(successor)

    def add_signature_node(self, signature_node):
        self.signature_nodes.append(signature_node)
        signature_node.set_is_active(True)

    def get_path_dim(self):
        return self.path_dim

    def get_train_increments(self, i):
        return self.train_increments[:, i, :]

    def get_validation_increments(self, i):
        return self.validation_increments[:, i, :]

    def get_test_increments(self, i):
        return self.test_increments[:, i, :]

    def get_number_nodes(self):
        return len(self.signature_nodes)

    def get_indices(self):
        indices = []
        for signature_node in self.signature_nodes:
            indices.append(signature_node.get_index())
        return indices

    def get_signature_leaves(self):
        signature_leaves = []
        for signature_node in self.signature_nodes:
            signature_node.compute_successors()
            signature_leaves.extend(signature_node.get_successor_leaves())
        return signature_leaves

    def get_number_samples(self, kind="train"):
        if kind == "train":
            return self.train_increments.shape[0]
        if kind == "validation":
            return self.validation_increments.shape[0]
        return self.test_increments.shape[0]

    def get_normalized_features(self, kind="train"):
        number_nodes = self.get_number_nodes()
        features = np.empty(shape=(self.get_number_samples(kind), number_nodes))
        for i in range(0, number_nodes):
            features[:, i] = self.signature_nodes[i].get_normalized_features(kind)
        return features

    def find_new_node(self, number_rf=1, number_trees=100):
        number_nodes = self.get_number_nodes() + 1
        train_features = np.empty(shape=(self.get_number_samples("train"), number_nodes))
        validation_features = np.empty(shape=(self.get_number_samples("validation"), number_nodes))
        train_features[:, :-1] = self.get_normalized_features("train")
        validation_features[:, :-1] = self.get_normalized_features("validation")
        signature_leaves = self.get_signature_leaves()
        average_accuracies = np.empty(shape=(len(signature_leaves)))
        for leave_nr in range(0, len(signature_leaves)):
            train_features[:, -1] = signature_leaves[leave_nr].get_normalized_features("train")
            validation_features[:, -1] = signature_leaves[leave_nr].get_normalized_features("validation")
            average_accuracies[leave_nr] = get_average_accuracy_rf(train_features=train_features, train_labels=self.train_labels, test_features=validation_features, test_labels=self.validation_labels, number_rf=number_rf, number_trees=number_trees)
        best_leave_nr = np.argmax(average_accuracies)
        self.add_signature_node(signature_leaves[best_leave_nr])

    def find_nodes_number(self, number_nodes=2, number_rf=1, number_trees=100):
        for i in range(0, number_nodes):
            self.find_new_node(number_rf=number_rf, number_trees=number_trees)

    def get_accuracy(self, test_paths, test_labels, number_rf=1, number_trees=100):
        self.test_increments = test_paths[:, :, 1:] - test_paths[:, :, :-1]
        self.test_labels = test_labels
        self.signature_nodes[0].compute_test_values(test_values=np.ones(shape=(test_paths.shape[0], test_paths.shape[2])))
        return get_average_accuracy_rf(train_features=self.get_normalized_features("train"), train_labels=self.train_labels, test_features=self.get_normalized_features("test"), test_labels=self.test_labels, number_rf=number_rf, number_trees=number_trees)


class SignatureNode:
    def __init__(self, train_values, validation_values, root, predecessor, index, is_active):
        self.train_values = train_values
        self.validation_values = validation_values
        self.test_values = None
        self.root = root
        self.is_active = is_active
        self.predecessor = predecessor
        self.successors = []
        self.successors_computed = False
        self.index = index

    def compute_successors(self):
        if not self.successors_computed:
            for i in range(0, self.root.get_path_dim()):
                train_values_successor = integrate_against_indefinite(self.train_values, self.root.get_train_increments(i))
                validation_values_successor = integrate_against_indefinite(self.validation_values, self.root.get_validation_increments(i))
                index_successor = self.index.copy()
                index_successor.append(i+1)
                self.successors.append(SignatureNode(train_values=train_values_successor,
                                                     validation_values=validation_values_successor,
                                                     root=self.root, predecessor=self,
                                                     index=index_successor, is_active=False))
        self.successors_computed = True

    def compute_test_values(self, test_values):
        self.test_values = test_values
        for successor in self.successors:
            if successor.get_is_active():
                test_values_successor = integrate_against_indefinite(self.test_values, self.root.get_test_increments(successor.get_index()[-1] - 1))
                successor.compute_test_values(test_values_successor)

    def get_level(self):
        return len(self.index)

    def get_index(self):
        return self.index

    def get_is_active(self):
        return self.is_active

    def set_is_active(self, is_active):
        self.is_active = is_active

    def get_successors_computed(self):
        return self.successors_computed

    def get_all_successors(self):
        return self.successors

    def get_raw_features(self, kind="train"):
        if kind == "train":
            return self.train_values[:, -1]
        if kind == "validation":
            return self.validation_values[:, -1]
        return self.test_values[:, -1]

    def get_normalized_features(self, kind="train"):
        return preprocessing.MinMaxScaler().fit_transform(self.get_raw_features(kind).reshape(-1, 1)).flatten()

    def get_successor_leaves(self):
        successor_leaves = []
        for successor in self.successors:
            if not successor.get_is_active():
                successor_leaves.append(successor)
        return successor_leaves


'''
train_paths = np.array([[[0., 0.5, 1., 1.5], [0., 1., 2., 3.]],
                        [[0., 0.5, 1., 1.5], [0., 5., -1., -2.]]])

validation_paths = np.array([[[0., 0.5, 1., 1.5], [0., 1., 2., 3.]],
                        [[0., 0.5, 1., 1.5], [0., 5., -1., -2.]]])

test_paths = np.array([[[0., 0.5, 1., 1.5], [0., 1., 2., -10.]],
                        [[0., 0.5, 1., 1.5], [0., 5., -1., 20.]]])

train_labels = np.array([True, False])
validation_labels = np.array([True, False])
test_labels = np.array([True, False])
'''

number_train_paths = 2000
number_validation_paths = 2000
number_test_paths = 2000
time_steps = 1000
T = 1.

train_paths = np.empty(shape=(number_train_paths, 2, time_steps+1))
validation_paths = np.empty(shape=(number_validation_paths, 2, time_steps+1))
test_paths = np.empty(shape=(number_test_paths, 2, time_steps+1))

train_labels = np.empty(shape=(number_train_paths,))
validation_labels = np.empty(shape=(number_validation_paths,))
test_labels = np.empty(shape=(number_test_paths,))

for i in range(0, number_train_paths):
    train_paths[i, :, :] = brownian_time(T, time_steps).transpose()
    train_labels[i] = check_hitting(train_paths[i, 1, :], 1.)

for i in range(0, number_validation_paths):
    validation_paths[i, :, :] = brownian_time(T, time_steps).transpose()
    validation_labels[i] = check_hitting(validation_paths[i, 1, :], 1.)

for i in range(0, number_test_paths):
    test_paths[i, :, :] = brownian_time(T, time_steps).transpose()
    test_labels[i] = check_hitting(test_paths[i, 1, :], 1.)

tree = SignatureTree(train_paths=train_paths, validation_paths=validation_paths, train_labels=train_labels, validation_labels=validation_labels, initial_signature_level=2)
tree.find_nodes_number(number_nodes=2, number_rf=10)
print(tree.get_indices())
print(tree.get_accuracy(test_paths=test_paths, test_labels=test_labels))
