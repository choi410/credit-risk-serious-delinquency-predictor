"""Validate prediction, clustering CSVs, and generated figures."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from PIL import Image


def main(result_dir, figure_dir, model_dir):
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    model_dir = Path(model_dir)

    ranked = pd.read_csv(result_dir / "01_전체고객_위험순위.csv")
    review = pd.read_csv(result_dir / "02_추가심사대상.csv")
    initial_pass = pd.read_csv(result_dir / "03_1차기준통과.csv")
    clustered = pd.read_csv(result_dir / "04_고객별_군집결과.csv")
    delinquency = pd.read_csv(
        result_dir / "06_학습데이터_군집별연체율.csv"
    )
    comparison = pd.read_csv(result_dir / "07_클러스터링_모델비교.csv")
    metadata = json.loads(
        (model_dir / "clustering_metadata.json").read_text(encoding="utf-8")
    )

    assert len(ranked) == len(review) + len(initial_pass)
    assert len(clustered) == len(ranked)
    assert ranked["Id"].is_unique and clustered["Id"].is_unique
    assert ranked["RiskScore"].is_monotonic_decreasing
    assert ranked["RiskRank"].tolist() == list(range(1, len(ranked) + 1))
    assert set(review["Id"]).isdisjoint(set(initial_pass["Id"]))
    assert set(ranked["Id"]) == set(review["Id"]) | set(initial_pass["Id"])

    probability_columns = sorted(
        column for column in clustered.columns if column.startswith("ClusterProb_")
    )
    probabilities = clustered[probability_columns].to_numpy()
    assert len(probability_columns) == metadata["n_clusters"]
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)
    assert np.allclose(
        probabilities.max(axis=1), clustered["ClusterConfidence"], atol=1e-8
    )
    assert set(clustered["ClusterID"]) == set(range(metadata["n_clusters"]))
    expected_names = set(metadata["segment_names"].values())
    assert set(clustered["SegmentName"]) == expected_names

    assert delinquency["CustomerCount"].sum() == metadata["training_rows"]
    selected = comparison[comparison["Selected"].eq(True)]
    assert len(selected) == 1
    assert selected.iloc[0]["Algorithm"] == "GMM"
    assert int(selected.iloc[0]["Clusters"]) == metadata["n_clusters"]
    assert bool(selected.iloc[0]["Eligible"])
    assert metadata["target_used_for_fit"] is False

    expected_figures = [
        "01_군집분포_PCA.png",
        "02_군집별_특성_히트맵.png",
        "03_군집별_고객수.png",
        "04_군집별_실제연체율.png",
        "05_군집신뢰도_분포.png",
        "06_군집수_선정지표.png",
    ]
    for name in expected_figures:
        path = figure_dir / name
        assert path.exists() and path.stat().st_size > 10000
        with Image.open(path) as image:
            width, height = image.size
        assert width >= 2000 and height >= 1200

    print("VALIDATION=PASS")
    print(f"CUSTOMERS={len(ranked):,}")
    print(f"CLUSTERS={metadata['n_clusters']}")
    print(f"FIGURES={len(expected_figures)}")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(root / "result"))
    parser.add_argument("--figure-dir", default=str(root / "figures"))
    parser.add_argument("--model-dir", default=str(root / "models"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.result_dir, args.figure_dir, args.model_dir)
