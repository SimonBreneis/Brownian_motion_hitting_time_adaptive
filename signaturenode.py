import numpy as np
import utilities as util


class SignatureNode:
    def __init__(self, train_paths, val_paths, root, index, is_active, is_right_successor):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = None
        self.train_increments = train_paths[:, 1:] - train_paths[:, :-1]
        self.val_increments = val_paths[:, 1:] - val_paths[:, :-1]
        self.test_increments = None
        self.root = root
        self.is_active = is_active
        self.successors = []
        self.successors_computed = False
        self.index = index
        self.is_right_successor = is_right_successor
        self.mu = np.average(train_paths[:, -1])
        self.sigma = np.std(train_paths[:, -1])

    def compute_successors(self):
        if not self.successors_computed:
            known_indices = self.root.node_and_leave_indices()
            for i in range(self.root.path_dim()):
                index_successor = self.index.copy()
                index_successor.append(i + 1)
                if index_successor not in known_indices:
                    train_values_successor = util.integrate(self.train_paths, self.root.get_train_increments(i))
                    val_values_successor = util.integrate(self.val_paths, self.root.get_val_increments(i))
                    self.successors.append(SignatureNode(train_values_successor, val_values_successor, self.root,
                                                         index_successor, is_active=False, is_right_successor=True))
                index_successor = self.index.copy()
                index_successor.insert(0, i + 1)
                is_this_index_symmetric = [x for x in index_successor if x != (i + 1)]
                if index_successor not in known_indices and len(is_this_index_symmetric) != 0:
                    train_values_successor = util.integrate(self.root.get_train_paths(i), self.train_increments)
                    val_values_successor = util.integrate(self.root.get_val_paths(i), self.val_increments)
                    self.successors.append(SignatureNode(train_values_successor, val_values_successor, self.root,
                                                         index_successor, is_active=False, is_right_successor=False))
        self.successors_computed = True

    def compute_test_paths(self, test_paths):
        self.test_paths = test_paths
        self.test_increments = test_paths[:, 1:] - test_paths[:, :-1]
        for successor in self.successors:
            if successor.is_active:
                if successor.is_right_successor:
                    test_values_successor = util.integrate(self.test_paths,
                                                           self.root.get_test_increments(successor.index[-1] - 1))
                    successor.compute_test_paths(test_values_successor)
                else:
                    test_values_successor = util.integrate(
                        self.root.get_test_paths(successor.index[0] - 1), self.test_increments)
                    successor.compute_test_paths(test_values_successor)

    def level(self):
        return len(self.index)

    def raw_features(self, kind):
        if kind == "train":
            return self.train_paths[:, -1]
        if kind == "validation":
            return self.val_paths[:, -1]
        return self.test_paths[:, -1]

    def normalized_features(self, kind):
        return (self.raw_features(kind)-self.mu)/(self.sigma if self.sigma != 0. else 1.)

    def successor_leaves(self):
        successor_leaves = []
        for successor in self.successors:
            if not successor.is_active:
                successor_leaves.append(successor)
        return successor_leaves
