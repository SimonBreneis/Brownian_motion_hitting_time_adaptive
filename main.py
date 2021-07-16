import joblib
import numpy as np
import matplotlib.pyplot as plt
import time
import signaturetree as st
import signatureforest as sf
import utilities as util


n_train = 20000
n_val = 20000
n_test = 10000
time_steps = 100
T = 1.
n_nodes_vec = (1, 1, 2, 4, 8, 16, 32, 64, 128)

util.test(n_train, n_val, n_test, n_nodes_vec, time_steps, T, util.predictor_reg, 1, "elimination")

