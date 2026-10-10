"""Unit tests for data processing and model logic."""

from pathlib import Path

import pandas as pd

from data.get_data import download_data


def test_download_creates_files(tmp_path):
    """Ensure that downloading the dataset creates the expected CSV files."""
    output_dir = tmp_path / "raw"
    features, target = download_data(str(output_dir))

    assert not features.empty
    assert not target.empty
    assert (output_dir / "features.csv").exists()
    assert (output_dir / "target.csv").exists()


def test_target_is_binary():
    """Check whether the target is binary after conversion."""
    output_dir = Path("data/raw")
    if not output_dir.exists():
        _, target = download_data(str(output_dir))
    else:
        target = pd.read_csv(output_dir / "target.csv").iloc[:, 0]

    assert set(target.unique()).issubset({0, 1})
