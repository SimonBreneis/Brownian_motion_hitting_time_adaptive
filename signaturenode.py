import utilities as util
from sklearn.preprocessing import StandardScaler


class SignatureNode:
    def __init__(self, train_values, val_values, root, index, is_active, is_right_successor):
        self.train_values = train_values
        self.val_values = val_values
        self.test_values = None
        self.train_increments = train_values[:, 1:] - train_values[:, :-1]
        self.val_increments = val_values[:, 1:] - val_values[:, :-1]
        self.test_increments = None
        self.root = root
        self.is_active = is_active
        self.successors = []
        self.successors_computed = False
        self.index = index
        self.is_right_successor = is_right_successor

    def compute_successors(self):
        if not self.successors_computed:
            known_indices = self.root.node_and_leave_indices()
            for i in range(0, self.root.path_dim()):
                index_successor = self.index.copy()
                index_successor.append(i + 1)
                if index_successor not in known_indices:
                    train_values_successor = util.integrate_against_indefinite(self.train_values,
                                                                               self.root.get_train_increments(i))
                    val_values_successor = util.integrate_against_indefinite(self.val_values,
                                                                             self.root.get_val_increments(i))
                    self.successors.append(SignatureNode(train_values_successor, val_values_successor, self.root,
                                                         index_successor, is_active=False, is_right_successor=True))
                index_successor = self.index.copy()
                index_successor.insert(0, i + 1)
                is_this_index_symmetric = [x for x in index_successor if x != (i + 1)]
                if index_successor not in known_indices and len(is_this_index_symmetric) != 0:
                    train_values_successor = util.integrate_against_indefinite(self.root.get_train_paths(i),
                                                                               self.train_increments)
                    val_values_successor = util.integrate_against_indefinite(self.root.get_val_paths(i),
                                                                             self.val_increments)
                    self.successors.append(SignatureNode(train_values_successor, val_values_successor, self.root,
                                                         index_successor, is_active=False, is_right_successor=False))
        self.successors_computed = True

    def compute_test_values(self, test_values):
        self.test_values = test_values
        self.test_increments = test_values[:, 1:] - test_values[:, :-1]
        for successor in self.successors:
            if successor.is_active:
                if successor.is_right_successor:
                    test_values_successor = util.integrate_against_indefinite(self.test_values,
                                                                              self.root.get_test_increments(
                                                                                  successor.index[-1] - 1))
                    successor.compute_test_values(test_values_successor)
                else:
                    test_values_successor = util.integrate_against_indefinite(
                        self.root.get_test_paths(successor.index[0] - 1), self.test_increments)
                    successor.compute_test_values(test_values_successor)

    def level(self):
        return len(self.index)

    def raw_features(self, kind="train"):
        if kind == "train":
            return self.train_values[:, -1]
        if kind == "validation":
            return self.val_values[:, -1]
        return self.test_values[:, -1]

    def normalized_features(self, kind="train"):
        return StandardScaler().fit_transform(self.raw_features(kind).reshape(-1, 1)).flatten()

    def successor_leaves(self):
        successor_leaves = []
        for successor in self.successors:
            if not successor.is_active:
                successor_leaves.append(successor)
        return successor_leaves
