import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
import signaturetree as st
import time


def brownian_motion(T, N):
    normals = np.random.normal(0, np.sqrt(float(T) / N), N)
    normals = np.concatenate((np.array([0.]), normals))
    return np.cumsum(normals)


def brownian_time(T, N):
    t = np.array([i * T / N for i in range(0, N + 1)]).reshape(N + 1, 1)
    brownian = brownian_motion(T, N).reshape(N + 1, 1)
    return np.concatenate((t, brownian), axis=1).transpose()


def check_hitting(path, threshold):
    return np.amax(path) >= threshold


def integrate(x, y_diff):
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


def check_hitting_levy(paths):
    x = paths[:, 0, :]
    y = paths[:, 1, :]
    x_diff = x[:, 1:] - x[:, :-1]
    y_diff = y[:, 1:] - y[:, :-1]
    levy_area = integrate(x, y_diff) - integrate(y, x_diff)
    return integrate(x, y_diff)[:, -1] + x[:, -1] > 0


def predictor_nn(train_features, train_labels, test_features):
    model = Sequential([
        Dense(2, input_shape=(train_features.shape[1],), activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.optimizer = keras.optimizers.Adam()
    model.optimizer.lr = 0.001
    model.fit(x=train_features, y=train_labels, batch_size=32, epochs=20, shuffle=True,
              verbose=0)
    return np.around(model.predict(x=test_features)[:, 1])


def predictor_rf(train_features, train_labels, test_features):
    rf = RandomForestClassifier(n_estimators=100, verbose=0)
    rf.fit(train_features, train_labels)
    return np.around(rf.predict(test_features))


def predictor_reg(train_features, train_labels, test_features):
    model = LogisticRegression(solver='liblinear', random_state=0).fit(train_features, train_labels)
    return model.predict(test_features)


def test(n_train, n_val, n_test, n_nodes_vec, time_steps, T, predictor, n_pred, mode):
    print(f"Number of training paths: {n_train}, Number of validation paths: {n_val}, Number of test paths: {n_test}.")
    print(f"Number of time steps: {time_steps}, Final time: {T}")

    train_paths = np.array([brownian_time(T, time_steps) for _ in range(n_train)])
    train_labels = np.array([check_hitting(train_path[1, :], 1) for train_path in train_paths])

    val_paths = np.array([brownian_time(T, time_steps) for _ in range(n_val)])
    val_labels = np.array([check_hitting(val_path[1, :], 1) for val_path in val_paths])

    test_paths = np.array([brownian_time(T, time_steps) for _ in range(n_test)])
    test_labels = np.array([check_hitting(test_path[1, :], 1) for test_path in test_paths])

    print("Path generation finished!")

    tree = st.SignatureTree(train_paths, val_paths, train_labels, val_labels, predictor=predictor, n_pred=n_pred)

    total_time = 0
    total_nodes = 0
    for n_nodes in n_nodes_vec:
        tic = time.perf_counter()
        tree.find_nodes(n_nodes=n_nodes, mode=mode)
        toc = time.perf_counter()
        total_time += toc - tic
        total_nodes += n_nodes
        print(f'It took {np.round(total_time, 2)} seconds to find {total_nodes} nodes in the {mode} SignatureTree.')
        print(f"Depth: {tree.depth()}, Number of nodes: {tree.n_nodes()}")
        print(f'The accuracy is {tree.accuracy(test_paths=test_paths, test_labels=test_labels)}.')
        print(f"The nodes used are {tree.indices()}")


def brownian_motion_multidimensional(T, N, dim):
    bm = np.empty(shape=(N+1, dim))
    for i in range(dim):
        bm[:, i] = brownian_motion(T, N)
    return bm


def brownian_adjoin_time(T, N, brownian):
    time = np.array([i * T / N for i in range(0, N + 1)]).reshape(N + 1, 1)
    brownian = brownian.reshape(N + 1, 1)
    return np.concatenate((time, brownian), axis=1)


def bisect(f, a, b, tol):
    """
    Finds minimum of convex f in [a,b] using bisection up to an error tolerance of tol.
    :param f: Function
    :param a: Left point of interval
    :param b: Right point of interval
    :param tol: Error tolerance
    :return: [argmin, min]
    """
    xl = a
    xr = b
    yl = f(xl)
    yr = f(xr)
    xm = (a+b)/2
    ym = f(xm)
    error = np.fabs(yr - yl)
    while error > tol:
        xml = (xl + xm)/2
        yml = f(xml)
        if yml < ym:
            xr = xm
            yr = ym
            xm = xml
            ym = yml
        else:
            xmr = (xm + xr)/2
            ymr = f(xmr)
            if ymr < ym:
                xl = xm
                yl = ym
                xm = xmr
                ym = ymr
            else:
                xl = xml
                yl = yml
                xr = xmr
                yr = ymr
        error = np.fabs(yr - yl)
    return np.array([xm, ym])


def bisect_no_interval(f, tol):
    """
    Finds minimum of function f on R up to some tolerance tol.
    :param f: The function
    :param tol: The tolerance
    :return: [argmin, min]
    """
    xl = -1
    xm = 0
    xr = 1
    yl = f(-1)
    ym = f(0)
    yr = f(1)
    interval_found = False
    while not interval_found:
        if ym <= yl and ym <= yr:
            interval_found = True
        elif yl < yr:
            xr = xm
            yr = ym
            xm = xl
            ym = yl
            xl *= 2
            yl = f(xl)
        else:
            xl = xm
            yl = ym
            xm = xr
            ym = yr
            xr *= 2
            yr = f(xr)
    return bisect(f, xl, xr, tol)
