import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from .utils import align_labels


def plot_dendrogram(linkage_matrix, output_dir, linkage, distance, title):
    """Save a dendrogram plot. Skips if the tree is too deep for Python's recursion limit."""
    try:
        plt.figure(figsize=(8, 4))
        dendrogram(linkage_matrix)
        plt.title(title)
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dendrogram_{linkage}_{distance}.png")
    except RecursionError:
        print(
            f"[WARNING] Dendrogram ({linkage}, {distance}) skipped: "
            f"tree too deep (recursion limit: {sys.getrecursionlimit()})."
        )
    finally:
        plt.close()


def plot_partitions(output_dir, X, partitions, filename, combo_names, reference_labels, annotate=False, features=(0, 1)):
    """Plot all partitions in a grid, with labels aligned to a reference partition."""

    X = np.asarray(X, dtype=float)
    fx, fy = features
    n_cols = 3
    n_rows = int(np.ceil(len(partitions) / n_cols))

    plt.figure(figsize=(n_cols * 4, n_rows * 4))
    plt.subplots_adjust(hspace=0, wspace=0)

    for i, partition in enumerate(partitions):
        partition = align_labels(reference_labels, partition)
        unique_labels = np.unique(partition)

        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        color_map = dict(zip(unique_labels, colors))

        plt.subplot(n_rows, n_cols, i + 1)

        for label in unique_labels:
            mask = partition == label

            plt.scatter(
                X[mask, fx],
                X[mask, fy],
                s=40,
                color=color_map[label],
                label=f"Cluster {label}",
            )

            if annotate:
                for idx in np.where(mask)[0]:
                    plt.annotate(
                        str(idx),
                        (X[idx, fx], X[idx, fy]),
                        textcoords="offset points",
                        xytext=(2, 2),
                        fontsize=8,
                        ha="left",
                        va="bottom",
                        color="black",
                    )

        plt.title(f"Partition {i + 1} ({combo_names[i]})", fontsize=12)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.legend(markerscale=1, fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{filename}.png")
    plt.close()
