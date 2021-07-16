
'''
number_nodes_adaptive = np.array([1, 2, 4, 8, 16, 32, 64])
number_nodes_full_nonadaptive = np.array([3, 7, 15, 31, 63])
number_nodes_sparse_nonadaptive = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 31, 63])

errors_nn_full_nonadaptive = np.array([0.1217, 0.0684, 0.0461, 0.0347, 0.0254])
errors_nn_sparse_nonadaptive = np.array([0.0833, 0.0723, 0.0391, 0.0354, 0.0266, 0.0269, 0.0242, 0.0258, 0.0254, 0.0258, 0.0258, 0.0238])
errors_rf_full_nonadaptive = np.array([0.1671, 0.0658, 0.0476, 0.0376, 0.0346])
errors_rf_sparse_nonadaptive = np.array([0.1173, 0.0510, 0.0302, 0.0225, 0.0200, 0.0177, 0.0160, 0.0147, 0.0134, 0.0077, 0.0036, 0.0022])

errors_nn_full_adaptive = np.array([0.1080, 0.0730, 0.0464, 0.0360, 0.0330, 0.0323, 0.0314])
errors_nn_linear_adaptive = np.array([0.1129, 0.0661, 0.0424, 0.0390, 0.0314, 0.0316, 0.0306])
errors_nn_elimination_adaptive = np.array([0.1070, 0.0653, 0.0487, 0.0422, 0.0240, 0.0242, 0.0243])

errors_rf_full_adaptive = np.array([0.1538, 0.0627, 0.0377, 0.0268, 0.0225, 0.0222, 0.0333])
errors_rf_linear_adaptive = np.array([0.1579, 0.0642, 0.0345, 0.0261, 0.0261, 0.0324, 0.0436])
errors_rf_elimination_adaptive = np.array([0.1506, 0.0631, 0.0321, 0.0250, 0.0253, 0.0266, 0.0327])

plt.loglog(number_nodes_full_nonadaptive, errors_nn_full_nonadaptive, label="full")
plt.loglog(number_nodes_sparse_nonadaptive, errors_nn_sparse_nonadaptive, label="sparse")
plt.loglog(number_nodes_adaptive, errors_nn_full_adaptive, label="adaptive (full)")
#plt.loglog(number_nodes_adaptive, errors_nn_linear_adaptive, label="adaptive (linear)")
plt.loglog(number_nodes_adaptive, errors_nn_elimination_adaptive, label="adaptive (elimination)")
plt.title("Neural Networks")
plt.xlabel("Number of elements")
plt.ylabel("Probability of false prediction")
plt.legend(loc="upper right")
plt.show()

plt.loglog(number_nodes_full_nonadaptive, errors_rf_full_nonadaptive, label="full")
plt.loglog(number_nodes_sparse_nonadaptive, errors_rf_sparse_nonadaptive, label="sparse")
plt.loglog(number_nodes_adaptive, errors_rf_full_adaptive, label="adaptive (full)")
#plt.loglog(number_nodes_adaptive, errors_rf_linear_adaptive, label="adaptive (linear)")
plt.loglog(number_nodes_adaptive, errors_rf_elimination_adaptive, label="adaptive (elimination)")
plt.title("Random Forests")
plt.xlabel("Number of elements")
plt.ylabel("Probability of false prediction")
plt.legend(loc="upper right")
plt.show()
'''