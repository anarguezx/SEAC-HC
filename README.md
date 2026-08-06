# SEAC

**SEAC (Selection-based Evidence Accumulation Clustering)** is a Python package implementing the SEAC consensus hierarchical clustering framework.

SEAC generates multiple hierarchical clustering partitions, automatically recommends the most appropriate internal clustering validity index (CVI), selects representative partitions according to their quality, and constructs a consensus dissimilarity matrix from which the final clustering is obtained.

## Features

- Selection-based evidence accumulation framework for hierarchical clustering.
- Automatic recommendation of the internal clustering validity index (Calinski–Harabasz or Dunn).
- Quality-guided selection of representative partitions using internal clustering validity indices.
- Consensus clustering through co-association and dissimilarity matrices.
- Evaluation using internal and external clustering validity indices.
- Automatic generation of dendrograms and clustering visualizations.

## Installation

Clone the repository and install the package:

```bash
git clone https://github.com/anarguezx/SEAC-HC.git
cd SEAC
pip install .
```

## Quick Start

The repository includes two toy datasets under the `data/` directory:

- `toy_data.csv`
- `toy_data_unclear.csv`

To run the example included with the package:

```bash
python example.py
```

This script:

- loads the toy datasets,
- recommends the appropriate internal CVI,
- runs the complete SEAC pipeline,
- saves all generated results under the `results/` directory.

## Using SEAC

The following example shows the complete workflow on one dataset:

```python
import pandas as pd

from SEAC import recommend_icvi, run_SEAC

df = pd.read_csv("data/toy_data.csv")

data = df[["x", "y"]].values
true_labels = df["label"].values
n_clusters = len(set(true_labels))

icvi = recommend_icvi(data)

run_SEAC(
    data=data,
    true_labels=true_labels,
    n_clusters=n_clusters,
    dataset_name="toy_data",
    results_dir="results",
    icvi=icvi,
)
```

Ground-truth labels are optional. If `true_labels=None`, SEAC computes only the internal clustering validity indices.

## Output

For each processed dataset, SEAC creates a directory inside `results/` containing:

- `primary_partitions.csv`
- `final_partitions.csv`
- `selected_partitions.txt`
- `co_association_matrix_selected_samples.txt`
- `dissimilarity_matrix_selected_samples.txt`
- Dendrograms for all generated partitions
- Visualizations of the primary and final partitions

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.
