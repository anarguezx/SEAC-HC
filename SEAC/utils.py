import csv
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler


def scale(data):
    """Return a Min-Max scaled copy of data."""
    return MinMaxScaler().fit_transform(np.asarray(data, dtype=float))


def labels_are_trivial(labels):
    """Return True when all samples belong to the same cluster."""
    return len(set(labels)) == 1


def parse_label_array(raw):
    """Parse a label array serialised as '[1 2 3 …]'."""
    return np.fromstring(raw.strip("[]"), sep=" ", dtype=int)

def align_labels(reference_labels, new_labels):
    """Remap new_labels to best match reference_labels using the Hungarian algorithm.
 
    Used to ensure visual consistency across partition plots.
 
    reference_labels: label array to align to
    new_labels: label array to remap
    """
    ref_unique = np.unique(reference_labels)
    new_unique = np.unique(new_labels)
 
    cost_matrix = np.zeros((len(ref_unique), len(new_unique)), dtype=int)
    for i, r in enumerate(ref_unique):
        for j, n in enumerate(new_unique):
            cost_matrix[i, j] = -np.sum((reference_labels == r) & (new_labels == n))
 
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {new_unique[col]: ref_unique[row] for row, col in zip(row_ind, col_ind)}
 
    return np.array([mapping.get(label, label) for label in new_labels])


def create_csv(header, rows, label_column_index, output_dir, output_filename):
    """Write rows to a CSV file, serialising the label array column.

    Parameters
    ----------
    header: column names
    rows: one list per clustering result
    label_column_index:  column position of the label array to serialise
    output_dir: output directory
    output_filename: output file name (without .csv extension)
    """
    filepath = Path(output_dir) / f"{output_filename}.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            row_to_write = row.copy()
            row_to_write[label_column_index] = "[" + " ".join(map(str, row[label_column_index])) + "]"
            writer.writerow(row_to_write)
