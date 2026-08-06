import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, confusion_matrix, davies_bouldin_score, fowlkes_mallows_score, 
                             normalized_mutual_info_score, pairwise_distances, silhouette_score,)
from sklearn.preprocessing import LabelEncoder
from validclust import dunn


def clustering_accuracy(true_labels, pred_labels):
    """Compute clustering accuracy using the Hungarian algorithm.

    Finds the optimal label assignment that maximises accuracy between true and predicted labels.

    true_labels: ground-truth cluster labels
    pred_labels: predicted cluster labels
    """
    unique_true, y_true = np.unique(true_labels, return_inverse=True)
    _, y_pred = np.unique(pred_labels, return_inverse=True)

    C = confusion_matrix(y_true, y_pred, labels=np.arange(len(unique_true)))
    row_ind, col_ind = linear_sum_assignment(-C)  # negative because Hungarian minimizes cost
    accuracy = C[row_ind, col_ind].sum() / y_pred.size
    return accuracy


def compute_scores(scaled_data, labels, true_labels):
    """Compute internal and external CVIs for a given partition.

    scaled_data: scaled feature matrix
    labels: predicted cluster labels
    true_labels: ground-truth cluster labels
    """
    cal = calinski_harabasz_score(scaled_data, labels)
    dav = davies_bouldin_score(scaled_data, labels)
    dunn_score = dunn(pairwise_distances(scaled_data), labels)
    sil = silhouette_score(scaled_data, labels)

    acc = ari = fmi = nmi = None

    if true_labels is not None:
        encoded_labels = LabelEncoder().fit_transform(true_labels)
        acc = clustering_accuracy(encoded_labels, labels)
        ari = adjusted_rand_score(encoded_labels, labels)
        fmi = fowlkes_mallows_score(encoded_labels, labels)
        nmi = normalized_mutual_info_score(encoded_labels, labels)
        
    return cal, dav, dunn_score, sil, acc, ari, fmi, nmi
