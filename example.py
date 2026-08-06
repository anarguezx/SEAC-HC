import pandas as pd
from SEAC.main import run_SEAC
from SEAC.recommender import recommend_icvi


RESULTS_DIR = "results"
DATASETS = [
    ("data/toy_data.csv",         "toy_data"),
    ("data/toy_data_unclear.csv", "toy_data_unclear"),
]


def load_toy_dataset(filepath):
    """Load a toy dataset, dropping the id column and separating features from labels.

    filepath: path to the CSV file
    """
    df = pd.read_csv(filepath)
    data = df[["x", "y"]].values
    true_labels = df["label"].values
    n_clusters = len(set(true_labels))
    return data, true_labels, n_clusters


def main():
    timing_records = []

    for filepath, dataset_name in DATASETS:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        data, true_labels, n_clusters = load_toy_dataset(filepath)

        icvi= recommend_icvi(data)

        timing = run_SEAC(
            data=data,
            true_labels=true_labels,
            n_clusters=n_clusters,
            dataset_name=dataset_name,
            results_dir=RESULTS_DIR,
            icvi=icvi,
            annotate=True,  # annotate sample indices
        )
        timing_records.append(timing)

        print(f"\nTiming summary for {dataset_name}:")
        for key, value in timing.items():
            print(f"  {key}: {value}")

    print(f"\n{'=' * 60}")
    print("All datasets processed. Results saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
