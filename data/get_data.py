"""
Data acquisition script for the Heart Disease UCI dataset.
Downloads the dataset from the UCI ML repository, saves the raw CSV files,
and prepares it for training and modeling.
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_data(output_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.Series]:
    """Download and save the Heart Disease dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading Heart Disease dataset from UCI ML repository...")
    heart_disease = fetch_ucirepo(id=45)
    features = heart_disease.data.features
    targets = heart_disease.data.targets

    binary_target = targets["num"].apply(lambda x: 1 if x > 0 else 0)

    features.to_csv(output_path / "features.csv", index=False)
    binary_target.to_csv(output_path / "target.csv", index=False)
    pd.concat([features, binary_target.rename("target")], axis=1).to_csv(
        output_path / "full_data.csv", index=False
    )

    print(f"Dataset saved to: {output_path}")
    print(f"Features shape: {features.shape}")
    print(f"Target shape: {binary_target.shape}")
    return features, binary_target


def load_data(features_path: str = "data/raw/features.csv", target_path: str = "data/raw/target.csv"):
    """Load previously downloaded feature and target files."""
    features = pd.read_csv(features_path)
    targets = pd.read_csv(target_path).iloc[:, 0]
    return features, targets


if __name__ == "__main__":
    features, target = download_data()
    print("\nClass distribution:")
    print(target.value_counts())
    print("\nProportion:")
    print(target.value_counts(normalize=True))
