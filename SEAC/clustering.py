import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from .matrices import get_matrices_samples
from .metrics import compute_scores
from .utils import create_csv, scale, labels_are_trivial, parse_label_array
from .plots import plot_dendrogram


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Column names for the base-partition CSV
BASE_COLUMNS = [
    "linkage", "distance", "clusters", "clustering",
    "CH", "DB", "DUNN", "SIL", "ACC", "ARI", "FMI", "NMI",
]

# Column index of the label array in primary_partitions.csv
PRIMARY_LABEL_COLUMN_INDEX = BASE_COLUMNS.index("clustering")

# Internal CVIs used with their corresponding optimization direction
INTERNAL_CVIS = {
    "CH": "maximize",
    "DB": "minimize",
    "DUNN": "maximize",
    "SIL": "maximize",
}

# Column names for the final-partition CSV
FINAL_COLUMNS = [
    "linkage", "clusters", "clustering",
    "CH", "DB", "DUNN", "SIL", "ACC", "ARI", "FMI", "NMI",
]

# Column index of the label array in final_partitions.csv
FINAL_LABEL_COLUMN_INDEX = FINAL_COLUMNS.index("clustering")



# ---------------------------------------------------------------------------
# Stage 1 – Primary partitions
# ---------------------------------------------------------------------------

def generate_primary_partitions(data, n_clusters, linkages, distances, output_dir, zero_indexed_labels, true_labels=None):
    """Run every (linkage, distance) pair and record evaluation scores.

    data: data matrix (n_samples, n_features)
    n_clusters: number of clusters
    linkages: linkage criteria
    distances: distance measures
    output_dir: where primary_partitions.csv and dendrograms are
    zero_indexed_labels: subtract 1 from labels so they start at 0
    true_labels: ground-truth labels for external CVIs
    """
    output_dir = Path(output_dir)

    scaled_data = scale(data)

    primary_partitions, combo_names, rows = [], [], []
    start = time.perf_counter()

    with tqdm(total=len(linkages) * len(distances), desc="Primary partitions") as progress:
        for link in linkages:
            for dist in distances:

                condensed_distances = pdist(scaled_data, metric=dist)

                if not np.all(np.isfinite(condensed_distances)):
                    print(f"[WARNING] Skipping {link}-{dist}: distance matrix contains non-finite values.")
                    progress.update(1)
                    continue

                ahc = linkage(condensed_distances, method=link)

                plot_dendrogram(ahc, output_dir, link, dist, title=f"Primary Partition Dendrogram ({link}, {dist})")

                labels = fcluster(ahc, t=n_clusters, criterion="maxclust")
                if zero_indexed_labels:
                    labels = labels - 1

                if labels_are_trivial(labels):
                    progress.update(1)
                    continue

                primary_partitions.append(labels)
                combo_names.append(f"{link}-{dist}")
                scores = compute_scores(scaled_data, labels, true_labels)
                rows.append([link, dist, n_clusters, labels, *scores])
                progress.update(1)

    create_csv(BASE_COLUMNS, rows, PRIMARY_LABEL_COLUMN_INDEX, output_dir, "primary_partitions")

    elapsed_time = time.perf_counter() - start
    print(f"[generate_primary_partitions] {elapsed_time:.3f} s")
    return primary_partitions, combo_names, elapsed_time


# ---------------------------------------------------------------------------
# Stage 2 – Select partitions by internal CVI
# ---------------------------------------------------------------------------

def select_partitions(primary_dir, output_dir, icvi):
    """Filter primary partitions by internal CVI score.

    For minimizing CVIs, partitions with scores at or below the mean are
    selected. For maximizing CVIs, partitions with scores at or above the
    mean are selected.

    output_dir: must contain primary_partitions.csv
    icvi: internal CVI to use for selection
    """
    primary_dir = Path(primary_dir)
    output_dir = Path(output_dir)
    start = time.perf_counter()

    df = pd.read_csv(primary_dir / "primary_partitions.csv")

    if icvi not in INTERNAL_CVIS:
        raise ValueError(
            f"Unknown internal CVI '{icvi}'. "
            f"Choose from: {', '.join(INTERNAL_CVIS)}"
        )
    
    scores = pd.to_numeric(df[icvi], errors="coerce")
    mean_score = scores.mean()

    if INTERNAL_CVIS[icvi] == "minimize":
        mask = scores <= mean_score
    else:
        mask = scores >= mean_score

    selected_df = df[mask]

    if selected_df.empty:
        raise ValueError(f"No partitions could be selected using the {icvi} score.")

    selected_indices = selected_df.index.to_numpy(dtype=int)

    np.savetxt(output_dir / "selected_partitions.txt", selected_indices, fmt="%d")

    selected_partitions = [parse_label_array(row["clustering"]) for _, row in selected_df.iterrows()]
    co_association_matrix, dissimilarity_matrix = get_matrices_samples(selected_partitions)

    np.savetxt(output_dir / "co_association_matrix_selected_samples.txt", co_association_matrix.astype(int), fmt="%d")
    np.savetxt(output_dir / "dissimilarity_matrix_selected_samples.txt", dissimilarity_matrix.astype(int), fmt="%d")

    elapsed_time = time.perf_counter() - start
    print(f"[select_partitions] {elapsed_time:.3f} s")
    return co_association_matrix, dissimilarity_matrix, elapsed_time


# ---------------------------------------------------------------------------
# Stage 3 – Final partitions on the dissimilarity matrix
# ---------------------------------------------------------------------------

def generate_final_partitions(data, n_clusters, linkages, dissimilarity_matrix, output_dir, zero_indexed_labels, true_labels=None):
    """Re-cluster using the consensus dissimilarity matrix built from selected partitions.

    data: data matrix (n_samples, n_features)
    true_labels: ground-truth labels for external CVIs
    n_clusters: number of clusters
    linkages: linkage criteria
    dissimilarity_matrix: square consensus matrix from select_partitions
    output_dir: where final_partitions.csv and dendrograms are
    zero_indexed_labels: subtract 1 from labels so they start at 0
    """
    output_dir = Path(output_dir)
    scaled_data = scale(data)
    condensed_dissimilarity_matrix = squareform(dissimilarity_matrix, checks=False)

    final_partitions, combo_names, rows = [], [], []
    start = time.perf_counter()

    with tqdm(total=len(linkages), desc="Final partitions") as progress:
        for link in linkages:

            ahc = linkage(condensed_dissimilarity_matrix, method=link)
            plot_dendrogram(ahc, output_dir, link, "SEAC", title=f"Final Partition Dendrogram ({link}, SEAC)")

            labels = fcluster(ahc, t=n_clusters, criterion="maxclust")
            if zero_indexed_labels:
                labels = labels - 1

            if labels_are_trivial(labels):
                progress.update(1)
                continue

            final_partitions.append(labels)
            combo_names.append(f"{link}-SEAC")
            scores = compute_scores(scaled_data, labels, true_labels)
            rows.append([link, n_clusters, labels, *scores])
            progress.update(1)

    create_csv(FINAL_COLUMNS, rows, FINAL_LABEL_COLUMN_INDEX, output_dir, "final_partitions")

    elapsed_time = time.perf_counter() - start
    print(f"[generate_final_partitions] {elapsed_time:.3f} s")
    return final_partitions, combo_names, elapsed_time
