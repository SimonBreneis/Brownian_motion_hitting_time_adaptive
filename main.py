import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import signaturetree as st
import signatureforest as sf
import utilities as util
import iisignature as sig
import classifiers
import load


signatures, labels = load.get_BM_hitting_full_sig_level_10()
classifiers.linear_regression(signatures, labels)


n_paths = 300
time_steps = 100
T = 1.
dim = 100
paths = np.array([util.brownian_motion_multidimensional(T, time_steps, dim) for _ in range(n_paths)])
labels = np.array([util.check_hitting(paths[i, :, 0], 1.) for i in range(n_paths)])
signatures = np.array([sig.sig(paths[i, :, :], 2) for i in range(n_paths)])

tic = time.perf_counter()
classifiers.shallow_binary_NN(signatures, labels, lr=1e-03, batch_size=10, epochs=100)
toc = time.perf_counter()
print(f"Time: {np.round(toc-tic, 2)} seconds")

tic = time.perf_counter()
classifiers.RF(signatures, labels, n_estimators=100)
toc = time.perf_counter()
print(f"Time: {np.round(toc-tic, 2)} seconds")

tic = time.perf_counter()
classifiers.linear_regression(signatures, labels)
toc = time.perf_counter()
print(f"Time: {np.round(toc-tic, 2)} seconds")

n_train = int(n_paths/3.)
train_paths = np.transpose(paths[:n_train, :, :], [0, 2, 1])
val_paths = np.transpose(paths[n_train:2*n_train, :, :], [0, 2, 1])
test_paths = np.transpose(paths[2*n_train:, :, :], [0, 2, 1])
train_labels = labels[:n_train]
val_labels = labels[n_train:2*n_train]
test_labels = labels[2*n_train:]

n_nodes_vec = (1, 1, 2, 4, 8, 16, 32, 64, 128)
tree = st.SignatureTree(train_paths, val_paths, train_labels, val_labels, predictor=util.predictor_reg, n_pred=1)

total_time = 0
total_nodes = 0
for n_nodes in n_nodes_vec:
    tic = time.perf_counter()
    tree.find_nodes(n_nodes=n_nodes, mode="elimination")
    toc = time.perf_counter()
    total_time += toc - tic
    total_nodes += n_nodes
    print(f'It took {np.round(total_time, 2)} seconds to find {total_nodes} nodes in the {"elimination"} SignatureTree.')
    print(f"Depth: {tree.depth()}, Number of nodes: {tree.n_nodes()}")
    print(f'The accuracy is {tree.accuracy(test_paths=test_paths, test_labels=test_labels)}.')
    print(f"The nodes used are {tree.indices()}")


time.sleep(3600)

n_train = 50000
n_val = 50000
n_test = 50000
time_steps = 100
T = 1.
n_pred = 1
n_nodes_vec = (1, 1, 2, 4, 8, 16, 32, 64, 128)

path = np.random.uniform(size=(20, 50))
signature = sig.sig(path, 2)
print(signature)

util.test(n_train, n_val, n_test, n_nodes_vec, time_steps, T, util.predictor_reg, n_pred, "elimination")
util.test(n_train, n_val, n_test, n_nodes_vec, time_steps, T, util.predictor_reg, n_pred, "linear")
util.test(n_train, n_val, n_test, n_nodes_vec, time_steps, T, util.predictor_reg, n_pred, "full")
