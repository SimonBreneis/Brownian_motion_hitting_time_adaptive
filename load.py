import numpy as np


def get_BM_hitting_full_sig_level_10():
    file = open('C:/Users/simon/signatures_for_training_hitting_time_1d_brownian_time_enhanced_large', 'rb')
    signatures = np.load(file)
    file.close()

    file = open('C:/Users/simon/labels_for_training_hitting_time_1d_brownian_time_enhanced_large', 'rb')
    labels = np.load(file)
    file.close()

    signatures[:, 1:3] *= 4
    signatures[:, 3:7] *= 4 ** 2
    signatures[:, 7:15] *= 4 ** 3
    signatures[:, 15:31] *= 4 ** 4
    signatures[:, 31:63] *= 4 ** 5
    signatures[:, 63:127] *= 4 ** 6
    signatures[:, 127:255] *= 4 ** 7
    signatures[:, 255:511] *= 4 ** 8
    signatures[:, 511:1023] *= 4 ** 9
    signatures[:, 1023:2047] *= 4 ** 10

    return signatures, labels


def get_BM_hitting_sparse_sig_level_10():
    signatures, labels = get_BM_hitting_full_sig_level_10()
    simplified_signatures = np.empty(shape=(signatures.shape[0], 9))
    simplified_signatures[:, 0] = signatures[:, 5]
    simplified_signatures[:, 1] = signatures[:, 13]
    simplified_signatures[:, 2] = signatures[:, 29]
    simplified_signatures[:, 3] = signatures[:, 61]
    simplified_signatures[:, 4] = signatures[:, 125]
    simplified_signatures[:, 5] = signatures[:, 253]
    simplified_signatures[:, 6] = signatures[:, 509]
    simplified_signatures[:, 7] = signatures[:, 1021]
    simplified_signatures[:, 8] = signatures[:, 2045]

    return simplified_signatures, labels


def get_BM_hitting_logsig_level_10():
    file = open('C:/Users/simon/signatures_for_training_hitting_time_1d_brownian_time_enhanced_log_large', 'rb')
    log_signatures = np.load(file)
    file.close()

    file = open('C:/Users/simon/labels_for_training_hitting_time_1d_brownian_time_enhanced_log_large', 'rb')
    labels = np.load(file)
    file.close()

    log_signatures[:, 0:2] *= 4
    log_signatures[:, 2:3] *= 4 ** 2
    log_signatures[:, 3:5] *= 4 ** 3
    log_signatures[:, 5:8] *= 4 ** 4
    log_signatures[:, 8:14] *= 4 ** 5
    log_signatures[:, 14:23] *= 4 ** 6
    log_signatures[:, 23:41] *= 4 ** 7
    log_signatures[:, 41:71] *= 4 ** 8
    log_signatures[:, 71:127] *= 4 ** 9
    log_signatures[:, 127:226] *= 4 ** 10

    return log_signatures, labels


def get_2d_BM_hitting_sig_level_10():
    file = open('C:/Users/simon/signatures_for_training_hitting_time_2d_brownian', 'rb')
    signatures = np.load(file)
    file.close()

    file = open('C:/Users/simon/labels_for_training_hitting_time_2d_brownian', 'rb')
    labels = np.load(file)
    file.close()

    signatures[:, 1:3] *= 4
    signatures[:, 3:7] *= 4 ** 2
    signatures[:, 7:15] *= 4 ** 3
    signatures[:, 15:31] *= 4 ** 4
    signatures[:, 31:63] *= 4 ** 5
    signatures[:, 63:127] *= 4 ** 6
    signatures[:, 127:255] *= 4 ** 7
    signatures[:, 255:511] *= 4 ** 8
    signatures[:, 511:1023] *= 4 ** 9
    signatures[:, 1023:2047] *= 4 ** 10

    return signatures, labels
