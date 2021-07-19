import numpy as np
import signaturenode as sn


class SignatureTree:
    def __init__(self, train_paths, val_paths, train_labels, val_labels, initial_level=0, predictor=None, n_pred=1):
        # A family of paths is given as [ [ [path1dim1value1, path1dim1value2, ...], [path1dim2value1, ...], ...], ...]
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = None
        self.train_increments = train_paths[:, :, 1:] - train_paths[:, :, :-1]
        self.val_increments = val_paths[:, :, 1:] - val_paths[:, :, :-1]
        self.test_increments = None
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = None
        self.training_active = True
        self.predictor = predictor
        self.n_pred = n_pred
        self.nodes = []
        self.leaves = []
        self.nodes.append(sn.SignatureNode(train_paths=np.ones(shape=(train_paths.shape[0], train_paths.shape[2])),
                                           val_paths=np.ones(shape=(val_paths.shape[0], val_paths.shape[2])),
                                           root=self, index=[], is_active=True, is_right_successor=True))
        self.nodes[0].compute_successors()
        self.leaves.extend(self.nodes[0].successor_leaves())

        for node in self.nodes:
            if node.level() < initial_level:
                node.compute_successors()
                successors = node.successors()
                for successor in successors:
                    self.add_node_initial(successor)
        self.val_accuracy = 0
        self.leave_accuracies = None

    def add_node(self, mode):
        if mode == "full":
            return self.add_node_full()
        if mode == "linear":
            return self.add_node_linear()
        if mode == "elimination":
            return self.add_node_elimination()

    def add_node_initial(self, node):
        if node.index not in self.indices():
            self.nodes.append(node)
            node.compute_successors()
            node.is_active = True
            if node in self.leaves:
                self.leaves.remove(node)
            self.leaves.extend(node.successor_leaves())
            return True
        return False

    def add_node_full(self):
        best_index = np.argmax(self.leave_accuracies)
        best_node = self.leaves[best_index]
        if best_node.index not in self.indices():
            self.nodes.append(best_node)
            best_node.compute_successors()
            best_node.is_active = True
            if best_node in self.leaves:
                self.leaves.remove(best_node)
            self.leaves.extend(best_node.successor_leaves())
            return True
        return False

    def add_node_linear(self):
        best_index = np.argmax(self.leave_accuracies)
        best_node = self.leaves[best_index]
        if best_node.index not in self.indices():
            self.nodes.append(best_node)
            best_node.compute_successors()
            best_node.is_active = True
            self.leaves = best_node.successor_leaves()
            return True
        return False

    def add_node_elimination(self):
        added_node = False
        best_index = np.argmax(self.leave_accuracies)
        best_node = self.leaves[best_index]
        if best_node.index not in self.indices() and self.leave_accuracies[best_index] > self.val_accuracy:
            self.nodes.append(best_node)
            best_node.compute_successors()
            best_node.is_active = True
            added_node = True
            if best_node in self.leaves:
                self.leaves.remove(best_node)
                self.leave_accuracies = np.delete(self.leave_accuracies, best_index)
        median_leave_accuracy = np.median(self.leave_accuracies)
        leaves_to_delete = []
        for i in range(len(self.leave_accuracies)):
            if self.leave_accuracies[i] <= median_leave_accuracy:
                leaves_to_delete.append(i)
        for i in sorted(leaves_to_delete, reverse=True):
            del self.leaves[i]
        self.leave_accuracies = np.delete(self.leave_accuracies, leaves_to_delete)
        self.leaves.extend(best_node.successor_leaves())
        return added_node

    def path_dim(self):
        return self.train_paths.shape[1]

    def get_train_paths(self, dim):
        return self.train_paths[:, dim, :]

    def get_val_paths(self, dim):
        return self.val_paths[:, dim, :]

    def get_test_paths(self, dim):
        return self.test_paths[:, dim, :]

    def get_train_increments(self, dim):
        return self.train_increments[:, dim, :]

    def get_val_increments(self, dim):
        return self.val_increments[:, dim, :]

    def get_test_increments(self, dim):
        return self.test_increments[:, dim, :]

    def n_nodes(self):
        return len(self.nodes)

    def depth(self):
        depth = 0
        indices = self.indices()
        for index in indices:
            depth = max(depth, len(index))
        return depth

    def indices(self):
        return [node.index for node in self.nodes]

    def node_and_leave_indices(self):
        indices = self.indices()
        for leave in self.leaves:
            indices.append(leave.index)
        return indices

    def n_samples(self, kind):
        if kind == "train":
            return self.train_increments.shape[0]
        if kind == "validation":
            return self.val_increments.shape[0]
        if self.test_increments is None:
            return 0
        return self.test_increments.shape[0]

    def features(self, kind):
        n_nodes = self.n_nodes()
        features = np.empty(shape=(self.n_samples(kind), n_nodes))
        for i in range(n_nodes):
            features[:, i] = self.nodes[i].normalized_features(kind)
        return features

    def find_new_node(self, mode):
        train_x = self.features("train")
        val_x = self.features("validation")
        self.val_accuracy = self.accuracy_given_features(train_x, self.train_labels, val_x, self.val_labels)
        train_features = np.empty(shape=(self.n_samples("train"), self.n_nodes()+1))
        val_features = np.empty(shape=(self.n_samples("validation"), self.n_nodes()+1))
        train_features[:, :-1] = train_x
        val_features[:, :-1] = val_x
        self.leave_accuracies = np.empty(len(self.leaves))
        for i in range(len(self.leaves)):
            train_features[:, -1] = self.leaves[i].normalized_features("train")
            val_features[:, -1] = self.leaves[i].normalized_features("validation")
            self.leave_accuracies[i] = self.accuracy_given_features(train_features, self.train_labels, val_features,
                                                                    self.val_labels)
        return self.add_node(mode)

    def find_nodes(self, n_nodes, mode):
        if not self.training_active:
            return False
        if mode == "total_elimination":
            while self.training_active:
                if len(self.leaves) != 0:
                    self.find_new_node("elimination")
                else:
                    self.training_active = False
            return True
        for i in range(n_nodes):
            found_new_node = False
            while not found_new_node:
                if len(self.leaves) != 0:
                    found_new_node = self.find_new_node(mode)
                else:
                    self.training_active = False
                    return False
        return True

    def predictions(self, test_paths):
        self.test_paths = test_paths
        self.test_increments = test_paths[:, :, 1:] - test_paths[:, :, :-1]
        self.nodes[0].compute_test_paths(test_paths=np.ones(shape=(test_paths.shape[0], test_paths.shape[2])))
        return np.array([self.predictor(self.features("train"), self.train_labels, self.features("test"))
                         for _ in range(self.n_pred)])

    def accuracy_given_features(self, train_features, train_labels, test_features, test_labels):
        predictions = np.array([self.predictor(train_features, train_labels, test_features)
                                for _ in range(self.n_pred)])
        accuracies = np.array([float(np.sum(predictions[i, :] == test_labels)) / len(test_labels)
                               for i in range(self.n_pred)])
        return np.average(accuracies)

    def predict(self, test_paths):
        predictions = self.predictions(test_paths)
        return np.around(np.sum(predictions, axis=0) / float(self.n_pred))

    def accuracy(self, test_paths, test_labels):
        predictions = self.predictions(test_paths)
        accuracies = np.array([float(np.sum(predictions[i, :] == test_labels)) / len(test_labels)
                               for i in range(self.n_pred)])
        return np.average(accuracies)
