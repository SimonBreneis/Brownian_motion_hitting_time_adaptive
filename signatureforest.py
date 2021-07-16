import numpy as np
import signaturetree as st


class SignatureForest:
    def __init__(self, train_paths, val_paths, train_labels, val_labels, n_trees, initial_level=0, predictor=None,
                 n_pred=1):
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.trees = []
        self.n_trees = n_trees
        n_train_elements = int(len(train_labels) / np.sqrt(n_trees))
        n_val_elements = int(len(val_labels) / np.sqrt(n_trees))
        for _ in range(n_trees):
            tree_train_elements = np.random.randint(len(train_labels), size=n_train_elements)
            tree_val_elements = np.random.randint(len(val_labels), size=n_val_elements)
            tree_train_paths = np.array([train_paths[element, :, :] for element in tree_train_elements])
            tree_train_labels = np.array([train_labels[element] for element in tree_train_elements])
            tree_val_paths = np.array([val_paths[element, :, :] for element in tree_val_elements])
            tree_val_labels = np.array([val_labels[element] for element in tree_val_elements])
            self.trees.append(st.SignatureTree(tree_train_paths, tree_val_paths, tree_train_labels, tree_val_labels,
                                               initial_level, predictor, n_pred))

    def find_nodes(self, n_nodes, mode):
        for tree in self.trees:
            if mode == "total_elimination":
                print("Starting new Tree...")
            tree.find_nodes(n_nodes, mode)

    def predict(self, test_paths):
        tree_accuracies = np.array([tree.val_accuracy(self.val_paths, self.val_labels) for tree in self.trees])
        print("Average SignatureTree accuracy: {0}".format(np.average(tree_accuracies)))
        print("Minimal SignatureTree accuracy: {0}".format(np.amin(tree_accuracies)))
        print("Maximal SignatureTree accuracy: {0}".format(np.amax(tree_accuracies)))
        predictions = np.array([tree.predict(test_paths) for tree in self.trees])
        return np.around(np.dot(predictions.transpose(), tree_accuracies) / np.sum(tree_accuracies))

    def accuracy(self, test_paths, test_labels):
        predictions = self.predict(test_paths)
        return float(np.sum(predictions == test_labels)) / len(predictions)

    def n_nodes(self):
        return np.array([tree.n_nodes() for tree in self.trees])

    def avg_n_nodes(self):
        return np.average(self.n_nodes())

    def max_n_nodes(self):
        return np.amax(self.n_nodes())

    def depth(self):
        return np.array([tree.depth() for tree in self.trees])

    def avg_depth(self):
        return np.average(self.depth())

    def max_depth(self):
        return np.amax(self.depth())

    def indices(self):
        return np.array([tree.indices() for tree in self.trees])

    def set_predictor(self, predictor):
        for tree in self.trees:
            tree.set_predictor(predictor)

    def set_n_predictions(self, n_pred):
        for tree in self.trees:
            tree.set_n_predictions(n_pred)
