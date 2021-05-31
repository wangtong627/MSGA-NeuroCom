from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import SpectralClustering
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, adjusted_rand_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def cluster_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return acc, nmi, ari

data_0 = sio.loadmat('Dataset/hhar_5.mat')
data_dict = dict(data_0)
G = sp.coo_matrix(data_dict['adj'])
X = np.matrix(data_dict['x'])
Y = np.squeeze(data_dict['y']) - 1
y_pred = SpectralClustering(affinity='nearest_neighbors', n_clusters=6, n_neighbors=5).fit_predict(X)

Y = np.array(Y).astype(int)
y_pred = np.array(y_pred).astype(int)

print(Y.shape)
print(y_pred.shape)

acc = cluster_acc(Y, y_pred)
_, nmi, ari = cluster_metrics(Y, y_pred)
print(acc)
print(nmi)
print(ari)

