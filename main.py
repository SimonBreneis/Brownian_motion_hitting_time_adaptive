import joblib
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
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed


def brownian_motion(T, N):
    normals = np.random.normal(0, np.sqrt(float(T) / N), N)
    normals = np.concatenate((np.array([0.]), normals))
    return np.cumsum(normals)


def brownian_time(T, N):
    time = np.array([i * T / N for i in range(0, N + 1)]).reshape(N + 1, 1)
    brownian = brownian_motion(T, N).reshape(N + 1, 1)
    return np.concatenate((time, brownian), axis=1).transpose()


def brownian_motion_multidimensional(T, N, dim):
    return np.array([brownian_motion(T, N) for _ in range(dim)])


def check_hitting(path, threshold):
    return np.amax(path) >= threshold


def integrate_against_indefinite(x, y_diff):
    """
    Computes the indefinite integral int x dy.
    :param x: Vector of x-values
    :param y_diff: Vector of y-increments
    :return: The vector of definite integrals
    """
    x_midpoint = (x[:, :-1] + x[:, 1:]) / 2
    integral_increments = x_midpoint * y_diff
    indefinite_integral = np.zeros(shape=x.shape)
    indefinite_integral[:, 1:] = np.cumsum(integral_increments, axis=-1)
    return indefinite_integral


def get_average_accuracy_nn(train_features, train_labels, test_features, test_labels, number_nn=1, lr=0.0001,
                            batch_size=None, epochs=100, activation='softmax', loss='sparse_categorical_crossentropy'):
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


def predictor_rf(train_features, train_labels, test_features, args):
    number_trees = args['number_trees']
    rf = RandomForestClassifier(n_estimators=number_trees, verbose=0)
    rf.fit(train_features, train_labels)
    return np.around(rf.predict(test_features))


class SignatureForest:
    def __init__(self, train_paths, validation_paths, train_labels, validation_labels, initial_signature_level=0,
                 number_siganture_trees=10):
        self.train_paths = train_paths
        self.validation_paths = validation_paths
        self.train_labels = train_labels
        self.validation_labels = validation_labels
        self.signature_trees = []
        self.number_signature_trees = number_siganture_trees
        number_train_elements = int(len(train_labels) / np.sqrt(number_siganture_trees))
        number_validation_elements = int(len(validation_labels) / np.sqrt(number_siganture_trees))
        for _ in range(number_siganture_trees):
            tree_train_elements = np.random.randint(len(train_labels), size=number_train_elements)
            tree_validation_elements = np.random.randint(len(validation_labels), size=number_validation_elements)
            tree_train_paths = np.array([train_paths[element, :, :] for element in tree_train_elements])
            tree_train_labels = np.array([train_labels[element] for element in tree_train_elements])
            tree_validation_paths = np.array([validation_paths[element, :, :] for element in tree_validation_elements])
            tree_validation_labels = np.array([validation_labels[element] for element in tree_validation_elements])
            self.signature_trees.append(
                SignatureTree(train_paths=tree_train_paths, validation_paths=tree_validation_paths,
                              train_labels=tree_train_labels, validation_labels=tree_validation_labels,
                              initial_signature_level=initial_signature_level))

    def find_nodes_number(self, number_nodes=2, mode="full", predictor=None, number_predictions=10, args=None):
        for signature_tree in self.signature_trees:
            if mode == "total_elimination":
                print("Starting new Tree...")
            signature_tree.find_nodes_number(number_nodes=number_nodes, mode=mode, predictor=predictor,
                                             number_predictions=number_predictions,
                                             args=args)

    def predict(self, test_paths, predictor=None, number_predictions=10, args=None):
        signature_tree_accuracies = np.array([signature_tree.get_accuracy(test_paths=self.validation_paths,
                                                                          test_labels=self.validation_labels,
                                                                          predictor=predictor,
                                                                          number_predictions=number_predictions,
                                                                          args=args) for signature_tree in
                                              self.signature_trees])
        # print(signature_tree_accuracies)
        print("Average SignatureTree accuracy: {0}".format(np.average(signature_tree_accuracies)))
        print("Minimal SignatureTree accuracy: {0}".format(np.amin(signature_tree_accuracies)))
        print("Maximal SignatureTree accuracy: {0}".format(np.amax(signature_tree_accuracies)))
        full_predictions = np.array([signature_tree.predict(test_paths=test_paths,
                                                            predictor=predictor,
                                                            number_predictions=number_predictions,
                                                            args=args) for signature_tree in self.signature_trees])
        return np.around(
            np.dot(full_predictions.transpose(), signature_tree_accuracies) / np.sum(signature_tree_accuracies))

    def get_accuracy(self, test_paths, test_labels, predictor=None, number_predictions=10, args=None):
        predictions = self.predict(test_paths=test_paths, predictor=predictor, number_predictions=number_predictions,
                                   args=args)
        return float(np.sum(predictions == test_labels)) / len(predictions)

    def get_number_nodes(self):
        return np.array([signature_tree.get_number_nodes() for signature_tree in self.signature_trees])

    def get_number_nodes_average(self):
        return np.average(self.get_number_nodes())

    def get_number_nodes_maximum(self):
        return np.amax(self.get_number_nodes())

    def get_depth(self):
        return np.array([signature_tree.get_depth() for signature_tree in self.signature_trees])

    def get_depth_average(self):
        return np.average(self.get_depth())

    def get_depth_maximum(self):
        return np.amax(self.get_depth())

    def get_indices(self):
        return np.array([signature_tree.get_indices() for signature_tree in self.signature_trees])


class SignatureTree:
    def __init__(self, train_paths, validation_paths, train_labels, validation_labels, initial_signature_level):
        # A family of paths is given as [ [ [path1dim1value1, path1dim1value2, ...], [path1dim2value1, ...], ...], ...]
        self.train_paths = train_paths
        self.validation_paths = validation_paths
        self.test_paths = None
        self.train_increments = train_paths[:, :, 1:] - train_paths[:, :, :-1]
        self.validation_increments = validation_paths[:, :, 1:] - validation_paths[:, :, :-1]
        self.test_increments = None
        self.train_labels = train_labels
        self.validation_labels = validation_labels
        self.test_labels = None
        self.path_dim = train_paths.shape[1]
        self.training_active = True
        self.signature_nodes = []
        self.signature_leaves = []
        self.signature_nodes.append(
            SignatureNode(train_values=np.ones(shape=(train_paths.shape[0], train_paths.shape[2])),
                          validation_values=np.ones(
                              shape=(validation_paths.shape[0], validation_paths.shape[2])),
                          root=self, index=[], is_active=True, is_right_successor=True))
        self.signature_nodes[0].compute_successors()
        self.signature_leaves.extend(self.signature_nodes[0].get_successor_leaves())

        for signature_node in self.signature_nodes:
            if signature_node.get_level() < initial_signature_level:
                signature_node.compute_successors()
                successors = signature_node.get_all_successors()
                for successor in successors:
                    self.add_signature_node_initial(successor)
        self.current_validation_accuracy = 0
        self.current_validation_accuracies_leaves = None

    def add_signature_node(self, mode):
        if mode == "full":
            return self.add_signature_node_full()
        if mode == "linear":
            return self.add_signature_node_linear()
        if mode == "elimination":
            return self.add_signature_node_elimination()

    def add_signature_node_initial(self, signature_node):
        if signature_node.get_index() not in self.get_indices():
            self.signature_nodes.append(signature_node)
            signature_node.compute_successors()
            signature_node.set_is_active(True)
            if signature_node in self.signature_leaves:
                self.signature_leaves.remove(signature_node)
            self.signature_leaves.extend(signature_node.get_successor_leaves())
            return True
        return False

    def add_signature_node_full(self):
        best_node_index = np.argmax(self.current_validation_accuracies_leaves)
        best_node = self.signature_leaves[best_node_index]
        if best_node.get_index() not in self.get_indices():
            self.signature_nodes.append(best_node)
            best_node.compute_successors()
            best_node.set_is_active(True)
            if best_node in self.signature_leaves:
                self.signature_leaves.remove(best_node)
            self.signature_leaves.extend(best_node.get_successor_leaves())
            return True
        return False

    def add_signature_node_linear(self):
        best_node_index = np.argmax(self.current_validation_accuracies_leaves)
        best_node = self.signature_leaves[best_node_index]
        if best_node.get_index() not in self.get_indices():
            self.signature_nodes.append(best_node)
            best_node.compute_successors()
            best_node.set_is_active(True)
            self.signature_leaves = best_node.get_successor_leaves()
            return True
        return False

    def add_signature_node_elimination(self):
        added_node = False
        best_node_index = np.argmax(self.current_validation_accuracies_leaves)
        best_node = self.signature_leaves[best_node_index]
        if best_node.get_index() not in self.get_indices() and self.current_validation_accuracies_leaves[
            best_node_index] > self.current_validation_accuracy:
            self.signature_nodes.append(best_node)
            best_node.compute_successors()
            best_node.set_is_active(True)
            added_node = True
            if best_node in self.signature_leaves:
                self.signature_leaves.remove(best_node)
                self.current_validation_accuracies_leaves = np.delete(self.current_validation_accuracies_leaves,
                                                                      best_node_index)
        median_validation_accuracy_leaves = np.median(self.current_validation_accuracies_leaves)
        number_leaves = len(self.current_validation_accuracies_leaves)
        leaves_to_delete = []
        for node_index in range(number_leaves):
            if self.current_validation_accuracies_leaves[node_index] < median_validation_accuracy_leaves:
                leaves_to_delete.append(node_index)
        for node_index in sorted(leaves_to_delete, reverse=True):
            del self.signature_leaves[node_index]
        self.current_validation_accuracies_leaves = np.delete(self.current_validation_accuracies_leaves,
                                                              leaves_to_delete)
        self.signature_leaves.extend(best_node.get_successor_leaves())
        if len(self.signature_leaves) == 1:
            self.signature_leaves = []
        return added_node

    def get_path_dim(self):
        return self.path_dim

    def get_train_values(self, i):
        return self.train_paths[:, i, :]

    def get_validation_values(self, i):
        return self.validation_paths[:, i, :]

    def get_test_values(self, i):
        return self.test_paths[:, i, :]

    def get_train_increments(self, i):
        return self.train_increments[:, i, :]

    def get_validation_increments(self, i):
        return self.validation_increments[:, i, :]

    def get_test_increments(self, i):
        return self.test_increments[:, i, :]

    def get_training_active(self):
        return self.training_active

    def get_number_nodes(self):
        return len(self.signature_nodes)

    def get_depth(self):
        depth = 0
        indices = self.get_indices()
        for index in indices:
            depth = max(depth, len(index))
        return depth

    def get_indices(self):
        indices = []
        for signature_node in self.signature_nodes:
            indices.append(signature_node.get_index())
        return indices

    def get_signature_node_and_leave_indices(self):
        known_indices = []
        for signature_node in self.signature_nodes:
            known_indices.append(signature_node.get_index())
        for signature_leave in self.signature_leaves:
            known_indices.append(signature_leave.get_index())
        return known_indices

    def get_number_samples(self, kind="train"):
        if kind == "train":
            return self.train_increments.shape[0]
        if kind == "validation":
            return self.validation_increments.shape[0]
        if self.test_increments is None:
            return 0
        return self.test_increments.shape[0]

    def get_normalized_features(self, kind="train"):
        number_nodes = self.get_number_nodes()
        features = np.empty(shape=(self.get_number_samples(kind), number_nodes))
        for i in range(0, number_nodes):
            features[:, i] = self.signature_nodes[i].get_normalized_features(kind)
        return features

    def get_accuracy_including_leave(self, signature_leave, train_features, validation_features, predictor=None,
                                     number_predictions=10,
                                     args=None):
        train_features[:, -1] = signature_leave.get_normalized_features("train")
        validation_features[:, -1] = signature_leave.get_normalized_features("validation")
        return self.get_accuracy_given_features(train_features=train_features, train_labels=self.train_labels,
                                                test_features=validation_features,
                                                test_labels=self.validation_labels, predictor=predictor,
                                                number_predictions=number_predictions, args=args)

    def find_new_node(self, mode="full", predictor=None, number_predictions=10, args=None):
        self.current_validation_accuracy = self.get_accuracy_given_features(
            train_features=self.get_normalized_features("train"), train_labels=self.train_labels,
            test_features=self.get_normalized_features("validation"), test_labels=self.validation_labels,
            predictor=predictor,
            number_predictions=10,
            args=args)
        number_nodes = self.get_number_nodes() + 1
        train_features = np.empty(shape=(self.get_number_samples("train"), number_nodes))
        validation_features = np.empty(shape=(self.get_number_samples("validation"), number_nodes))
        train_features[:, :-1] = self.get_normalized_features("train")
        validation_features[:, :-1] = self.get_normalized_features("validation")
        signature_leaves = self.signature_leaves
        self.current_validation_accuracies_leaves = Parallel(
            n_jobs=int(max(1, cpu_number / min(cpu_number, number_predictions))))(
            delayed(self.get_accuracy_including_leave)(signature_leave=signature_leave, train_features=train_features,
                                                       validation_features=validation_features,
                                                       predictor=predictor, number_predictions=number_predictions,
                                                       args=args) for
            signature_leave in signature_leaves)
        return self.add_signature_node(mode)

    def find_nodes_number(self, number_nodes=2, mode="full", predictor=None, number_predictions=10, args=None):
        if not self.training_active:
            return False
        if mode == "total_elimination":
            while self.training_active:
                if len(self.signature_leaves) != 0:
                    self.find_new_node(mode="elimination", predictor=predictor,
                                       number_predictions=number_predictions, args=args)
                else:
                    self.training_active = False
            return True
        for i in range(0, number_nodes):
            found_new_node = False
            while not found_new_node:
                if len(self.signature_leaves) != 0:
                    found_new_node = self.find_new_node(mode=mode, predictor=predictor,
                                                        number_predictions=number_predictions, args=args)
                else:
                    self.training_active = False
                    return False
        return True

    def all_predictions(self, test_paths, predictor, number_predictions=10, args=None):
        self.test_paths = test_paths
        self.test_increments = test_paths[:, :, 1:] - test_paths[:, :, :-1]
        self.signature_nodes[0].compute_test_values(
            test_values=np.ones(shape=(test_paths.shape[0], test_paths.shape[2])))
        return self.all_predictions_given_features(train_features=self.get_normalized_features("train"),
                                                   train_labels=self.train_labels,
                                                   test_features=self.get_normalized_features("test"),
                                                   predictor=predictor, number_predictions=number_predictions,
                                                   args=args)

    def all_predictions_given_features(self, train_features, train_labels, test_features, predictor=None,
                                       number_predictions=10, args=None):
        return np.array(
            Parallel(n_jobs=min(number_predictions, cpu_number))(delayed(predictor)(train_features=train_features,
                                                                                    train_labels=train_labels,
                                                                                    test_features=test_features,
                                                                                    args=args) for _ in
                                                                 range(number_predictions)))

    def get_accuracy_given_features(self, train_features, train_labels, test_features, test_labels, predictor=None,
                                    number_predictions=10, args=None):
        all_predictions = self.all_predictions_given_features(train_features=train_features, train_labels=train_labels,
                                                              predictor=predictor, test_features=test_features,
                                                              number_predictions=number_predictions, args=args)
        accuracies = np.array(
            [float(np.sum(all_predictions[prediction, :] == test_labels)) / len(test_labels) for prediction in
             range(number_predictions)])
        return np.average(accuracies)

    def predict(self, test_paths, predictor, number_predictions=10, args=None):
        all_predictions = self.all_predictions(test_paths=test_paths, predictor=predictor,
                                               number_predictions=number_predictions, args=args)
        return np.around(np.sum(all_predictions, axis=0) / float(number_predictions))

    def get_accuracy(self, test_paths, test_labels, predictor, number_predictions=10, args=None):
        all_predictions = self.all_predictions(test_paths=test_paths, predictor=predictor,
                                               number_predictions=number_predictions, args=args)
        accuracies = np.array(
            [float(np.sum(all_predictions[prediction, :] == test_labels)) / len(test_labels) for prediction in
             range(number_predictions)])
        return np.average(accuracies)


class SignatureNode:
    def __init__(self, train_values, validation_values, root, index, is_active, is_right_successor):
        self.train_values = train_values
        self.validation_values = validation_values
        self.test_values = None
        self.train_increments = train_values[:, 1:] - train_values[:, :-1]
        self.validation_increments = validation_values[:, 1:] - validation_values[:, :-1]
        self.test_increments = None
        self.root = root
        self.is_active = is_active
        self.successors = []
        self.successors_computed = False
        self.index = index
        self.is_right_successor = is_right_successor

    def compute_successors(self):
        if not self.successors_computed:
            known_indices = self.root.get_signature_node_and_leave_indices()
            for i in range(0, self.root.get_path_dim()):
                index_successor = self.index.copy()
                index_successor.append(i + 1)
                if index_successor not in known_indices:
                    train_values_successor = integrate_against_indefinite(self.train_values,
                                                                          self.root.get_train_increments(i))
                    validation_values_successor = integrate_against_indefinite(self.validation_values,
                                                                               self.root.get_validation_increments(i))
                    self.successors.append(SignatureNode(train_values=train_values_successor,
                                                         validation_values=validation_values_successor,
                                                         root=self.root, index=index_successor, is_active=False,
                                                         is_right_successor=True))
                index_successor = self.index.copy()
                index_successor.insert(0, i + 1)
                is_this_index_symmetric = [x for x in index_successor if x != (i + 1)]
                if index_successor not in known_indices and len(is_this_index_symmetric) != 0:
                    train_values_successor = integrate_against_indefinite(self.root.get_train_values(i),
                                                                          self.train_increments)
                    validation_values_successor = integrate_against_indefinite(self.root.get_validation_values(i),
                                                                               self.validation_increments)
                    self.successors.append(SignatureNode(train_values=train_values_successor,
                                                         validation_values=validation_values_successor,
                                                         root=self.root, index=index_successor, is_active=False,
                                                         is_right_successor=False))
        self.successors_computed = True

    def compute_test_values(self, test_values):
        self.test_values = test_values
        self.test_increments = test_values[:, 1:] - test_values[:, :-1]
        for successor in self.successors:
            if successor.get_is_active():
                if successor.get_is_right_successor():
                    test_values_successor = integrate_against_indefinite(self.test_values,
                                                                         self.root.get_test_increments(
                                                                             successor.get_index()[-1] - 1))
                    successor.compute_test_values(test_values_successor)
                else:
                    test_values_successor = integrate_against_indefinite(
                        self.root.get_test_values(successor.get_index()[0] - 1), self.test_increments)
                    successor.compute_test_values(test_values_successor)

    def get_level(self):
        return len(self.index)

    def get_index(self):
        return self.index

    def get_is_active(self):
        return self.is_active

    def set_is_active(self, is_active):
        self.is_active = is_active

    def get_is_right_successor(self):
        return self.is_right_successor

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


# Data for testing
number_train_paths = 200
number_validation_paths = 200
number_test_paths = 200
time_steps = 20
T = 1.
number_nodes_vec = (1, 1, 2)
number_rf = 10
number_trees = 20
number_signature_trees = 5

'''
# Data for actual numerical experiments
number_train_paths = 2000
number_validation_paths = 2000
number_test_paths = 2000
time_steps = 200
T = 1.
number_nodes_vec = (1, 1, 2, 4, 8, 16, 32, 64, 128)
number_rf = 100
number_trees = 20
number_signature_trees = 100
'''

args = {"number_trees": number_trees}

cpu_number = joblib.cpu_count() - 2

train_paths = np.array(
    Parallel(n_jobs=cpu_number)(
        delayed(brownian_time)(T, time_steps) for _ in range(number_train_paths)))
train_labels = np.array(
    Parallel(n_jobs=cpu_number)(delayed(check_hitting)(train_paths[i, 1, :], 1) for i in range(number_train_paths)))

validation_paths = np.array(
    Parallel(n_jobs=cpu_number)(
        delayed(brownian_time)(T, time_steps) for _ in range(number_validation_paths)))
validation_labels = np.array(Parallel(n_jobs=cpu_number)(
    delayed(check_hitting)(validation_paths[i, 1, :], 1) for i in range(number_validation_paths)))

test_paths = np.array(
    Parallel(n_jobs=cpu_number)(
        delayed(brownian_time)(T, time_steps) for _ in range(number_test_paths)))
test_labels = np.array(
    Parallel(n_jobs=cpu_number)(delayed(check_hitting)(test_paths[i, 1, :], 1) for i in range(number_test_paths)))

forest = SignatureForest(train_paths=train_paths, validation_paths=validation_paths, train_labels=train_labels,
                         validation_labels=validation_labels, initial_signature_level=0,
                         number_siganture_trees=number_signature_trees)
total_time = 0
total_nodes = 0
number_nodes = 1
tic = time.perf_counter()
forest.find_nodes_number(number_nodes=number_nodes, mode="total_elimination", predictor=predictor_rf,
                         number_predictions=number_rf,
                         args=args)
toc = time.perf_counter()
total_time += toc - tic
total_nodes += number_nodes
print('It took {0} seconds to finish the total elimination SignatureForest.'.format(total_time))
print(f"Average depth: {forest.get_depth_average()}, Maximal depth: {forest.get_depth_maximum()}")
print(
    f"Average number of nodes: {forest.get_number_nodes_average()}, Maximal number of nodes: {forest.get_number_nodes_maximum()}")
print('The accuracy is {0}.'.format(
    forest.get_accuracy(test_paths=test_paths, test_labels=test_labels, predictor=predictor_rf,
                        number_predictions=number_rf,
                        args=args)))

forest = SignatureForest(train_paths=train_paths, validation_paths=validation_paths, train_labels=train_labels,
                         validation_labels=validation_labels, initial_signature_level=0,
                         number_siganture_trees=number_signature_trees)
total_time = 0
total_nodes = 0
for number_nodes in number_nodes_vec:
    tic = time.perf_counter()
    forest.find_nodes_number(number_nodes=number_nodes, mode="elimination", predictor=predictor_rf,
                             number_predictions=number_rf,
                             args=args)
    toc = time.perf_counter()
    total_time += toc - tic
    total_nodes += number_nodes
    print('It took {0} seconds to find {1} nodes in the elimination SignatureForest.'.format(total_time, total_nodes))
    print(f"Average depth: {forest.get_depth_average()}, Maximal depth: {forest.get_depth_maximum()}")
    print('The accuracy is {0}.'.format(
        forest.get_accuracy(test_paths=test_paths, test_labels=test_labels, predictor=predictor_rf,
                            number_predictions=number_rf,
                            args=args)))

forest = SignatureForest(train_paths=train_paths, validation_paths=validation_paths, train_labels=train_labels,
                         validation_labels=validation_labels, initial_signature_level=0,
                         number_siganture_trees=number_signature_trees)
total_time = 0
total_nodes = 0
for number_nodes in number_nodes_vec:
    tic = time.perf_counter()
    forest.find_nodes_number(number_nodes=number_nodes, mode="linear", predictor=predictor_rf,
                             number_predictions=number_rf,
                             args=args)
    toc = time.perf_counter()
    total_time += toc - tic
    total_nodes += number_nodes
    print('It took {0} seconds to find {1} nodes in the linear SignatureForest.'.format(total_time, total_nodes))
    print(f"Average depth: {forest.get_depth_average()}, Maximal depth: {forest.get_depth_maximum()}")
    print('The accuracy is {0}.'.format(
        forest.get_accuracy(test_paths=test_paths, test_labels=test_labels, predictor=predictor_rf,
                            number_predictions=number_rf,
                            args=args)))

forest = SignatureForest(train_paths=train_paths, validation_paths=validation_paths, train_labels=train_labels,
                         validation_labels=validation_labels, initial_signature_level=0,
                         number_siganture_trees=number_signature_trees)
total_time = 0
total_nodes = 0
for number_nodes in number_nodes_vec:
    tic = time.perf_counter()
    forest.find_nodes_number(number_nodes=number_nodes, mode="full", predictor=predictor_rf,
                             number_predictions=number_rf,
                             args=args)
    toc = time.perf_counter()
    total_time += toc - tic
    total_nodes += number_nodes
    print('It took {0} seconds to find {1} nodes in the full SignatureForest.'.format(total_time, total_nodes))
    print(f"Average depth: {forest.get_depth_average()}, Maximal depth: {forest.get_depth_maximum()}")
    print('The accuracy is {0}.'.format(
        forest.get_accuracy(test_paths=test_paths, test_labels=test_labels, predictor=predictor_rf,
                            number_predictions=number_rf,
                            args=args)))
