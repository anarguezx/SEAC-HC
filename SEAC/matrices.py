import numpy as np
from tqdm import tqdm


def get_matrices_samples(partitions):
    """Build co-association and dissimilarity matrices across samples.

    partitions: list of label arrays, each of shape (n_samples,)
    """
    n_partitions = len(partitions)

    with tqdm(total=n_partitions, desc="Samples' matrices") as progress:
        partitions_array = np.array(partitions)       # (n_partitions, n_samples)
        labels_as_col = partitions_array[:, :, None]   # (n_partitions, n_samples, 1)
        labels_as_row = partitions_array[:, None, :]   # (n_partitions, 1, n_samples)

        same_cluster = labels_as_col == labels_as_row   # (n_partitions, n_samples, n_samples)
        co_association_matrix = same_cluster.sum(axis=0)
        dissimilarity_matrix = n_partitions - co_association_matrix

        np.fill_diagonal(co_association_matrix, n_partitions)
        np.fill_diagonal(dissimilarity_matrix, 0)

        progress.update(n_partitions)
    return co_association_matrix, dissimilarity_matrix

