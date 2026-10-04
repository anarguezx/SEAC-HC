import os

from .clustering import generate_primary_partitions, select_partitions, generate_final_partitions
from .plots import plot_partitions


def run_SEAC(data, true_labels, n_clusters, dataset_name, dataset_dir, icvi, 
                 linkages=['single', 'complete', 'average'], 
                 distances=['euclidean', 'cityblock', 'cosine'], 
                 zero_indexed_labels=False, annotate=False):
    """Run the full ensemble hierarchical clustering pipeline on a single dataset.

    data: feature matrix (n_samples, n_features)
    true_labels: ground-truth cluster labels
    n_clusters: number of clusters
    dataset_name: used as subdirectory name under results_dir
    dataset_dir: dataset output directory
    icvi: internal CVI used for partition selection
    linkages: linkage methods to use, defaults to LINKAGES
    distances: distance metrics to use, defaults to DISTANCES
    zero_indexed_labels: subtract 1 from labels so they start at 0
    annotate: if True, annotate each sample with its index in the plots
    """

    os.makedirs(dataset_dir, exist_ok=True)

    primary_partitions, combo_names, t_base = generate_primary_partitions(
        data, n_clusters, linkages, distances, dataset_dir,
        zero_indexed_labels=zero_indexed_labels, 
        true_labels=true_labels
    )

    if not primary_partitions:
        raise ValueError("No valid primary partitions were generated.")
    
    reference_labels = (true_labels if true_labels is not None else primary_partitions[0])

    plot_partitions(dataset_dir, data, primary_partitions, "primary_partitions", combo_names, reference_labels, annotate=annotate)

    _, dissimilarity_matrix, t_select = select_partitions(dataset_dir, icvi)

    final_partitions, final_combo_names, t_final = generate_final_partitions(
        data, n_clusters, linkages, dissimilarity_matrix, dataset_dir,
        zero_indexed_labels=zero_indexed_labels,
        true_labels=true_labels
    )

    if not final_partitions:
        raise ValueError("No valid final partitions were generated.")

    plot_partitions(dataset_dir, data, final_partitions, "final_partitions", final_combo_names, reference_labels, annotate=annotate)

    return {
        "Dataset": dataset_name,
        "# Samples": len(data),
        "# Clusters": n_clusters,
        "Primary partitions (s)": round(t_base, 3),
        "Selection (s)": round(t_select, 3),
        "Final partitions (s)": round(t_final, 3),
        "Total (s)": round(t_base + t_select + t_final, 3),
    }

