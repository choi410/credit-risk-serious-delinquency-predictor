"""Compare clustering candidates, fit the final GMM, and export reports."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from clustering import (
    CLUSTER_COLUMNS,
    CLUSTER_METADATA_FILE,
    CLUSTER_MODEL_FILE,
    CreditClusteringPreprocessor,
    build_segment_names,
    canonicalize_membership,
    cleaned_profile_frame,
)
from clustering_visualization import generate_training_figures
from preprocessing import INPUT_COLUMNS, RANDOM_STATE, load_training_data


COMPARISON_FILE = "07_클러스터링_모델비교.csv"
DELINQUENCY_FILE = "06_학습데이터_군집별연체율.csv"


def evaluate_labels(values, labels):
    counts = pd.Series(labels).value_counts(normalize=True)
    if len(counts) < 2:
        return np.nan, np.nan, np.nan, float(counts.min())
    return (
        float(silhouette_score(values, labels)),
        float(calinski_harabasz_score(values, labels)),
        float(davies_bouldin_score(values, labels)),
        float(counts.min()),
    )


def compare_candidates(processed, min_k, max_k, selection_rows, silhouette_rows):
    rng = np.random.default_rng(RANDOM_STATE)
    selection_count = min(selection_rows, len(processed))
    selection_indices = rng.choice(len(processed), size=selection_count, replace=False)
    selection = processed[selection_indices]
    evaluation_count = min(silhouette_rows, selection_count)
    evaluation_indices = rng.choice(selection_count, size=evaluation_count, replace=False)
    evaluation = selection[evaluation_indices]

    rows = []
    for clusters in range(min_k, max_k + 1):
        kmeans = KMeans(
            n_clusters=clusters,
            n_init=10,
            max_iter=300,
            random_state=RANDOM_STATE,
        ).fit(selection)
        evaluation_labels = kmeans.predict(evaluation)
        selection_labels = kmeans.predict(selection)
        silhouette, calinski, davies, _ = evaluate_labels(
            evaluation, evaluation_labels
        )
        min_share = float(pd.Series(selection_labels).value_counts(normalize=True).min())
        rows.append(
            {
                "Algorithm": "K-Means",
                "Clusters": clusters,
                "Silhouette": silhouette,
                "CalinskiHarabasz": calinski,
                "DaviesBouldin": davies,
                "BIC": np.nan,
                "AIC": np.nan,
                "BICPerRow": np.nan,
                "AICPerRow": np.nan,
                "MinClusterShare": min_share,
                "Converged": True,
            }
        )

        gmm = GaussianMixture(
            n_components=clusters,
            covariance_type="diag",
            reg_covar=1e-5,
            max_iter=300,
            n_init=3,
            random_state=RANDOM_STATE,
        ).fit(selection)
        evaluation_labels = gmm.predict(evaluation)
        selection_labels = gmm.predict(selection)
        silhouette, calinski, davies, _ = evaluate_labels(
            evaluation, evaluation_labels
        )
        min_share = float(pd.Series(selection_labels).value_counts(normalize=True).min())
        bic = float(gmm.bic(selection))
        aic = float(gmm.aic(selection))
        rows.append(
            {
                "Algorithm": "GMM",
                "Clusters": clusters,
                "Silhouette": silhouette,
                "CalinskiHarabasz": calinski,
                "DaviesBouldin": davies,
                "BIC": bic,
                "AIC": aic,
                "BICPerRow": bic / selection_count,
                "AICPerRow": aic / selection_count,
                "MinClusterShare": min_share,
                "Converged": bool(gmm.converged_),
            }
        )
        print(f"후보 평가 완료: k={clusters}", flush=True)

    comparison = pd.DataFrame(rows)
    comparison["Eligible"] = (
        comparison["Converged"].eq(True)
        & comparison["MinClusterShare"].ge(0.02)
    )
    valid_gmm = comparison[
        comparison["Algorithm"].eq("GMM")
        & comparison["Eligible"].eq(True)
    ]
    if valid_gmm.empty:
        valid_gmm = comparison[comparison["Algorithm"].eq("GMM")]
    selected_index = valid_gmm["BICPerRow"].idxmin()
    selected_k = int(comparison.loc[selected_index, "Clusters"])
    comparison["Selected"] = False
    comparison.loc[selected_index, "Selected"] = True
    comparison["SelectionRows"] = selection_count
    comparison["SilhouetteRows"] = evaluation_count
    return comparison, selected_k


def build_cluster_profile(profile, labels, confidence, segment_names):
    working = profile.copy()
    working["ClusterID"] = labels
    working["ClusterConfidence"] = confidence
    grouped = working.groupby("ClusterID")
    summary = pd.DataFrame(
        {
            "ClusterID": sorted(np.unique(labels)),
        }
    ).set_index("ClusterID")
    summary["SegmentName"] = pd.Series(segment_names)
    summary["CustomerCount"] = grouped.size()
    summary["Share"] = grouped.size() / len(working)
    summary["ClusterConfidenceMean"] = grouped["ClusterConfidence"].mean()
    for column in INPUT_COLUMNS:
        summary[f"Median_{column}"] = grouped[column].median()
    return summary.reset_index()


def main(
    data_dir,
    model_dir,
    result_dir,
    figure_dir,
    min_k=2,
    max_k=6,
    selection_rows=50000,
    silhouette_rows=3000,
):
    model_dir = Path(model_dir)
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_training_data(data_dir)
    preprocessor = CreditClusteringPreprocessor()
    print(f"군집 전처리 학습: {len(X):,}명", flush=True)
    processed = preprocessor.fit_transform(X)
    comparison, selected_k = compare_candidates(
        processed,
        min_k,
        max_k,
        selection_rows,
        silhouette_rows,
    )

    print(f"최종 GMM 학습: k={selected_k}, {len(X):,}명", flush=True)
    model = GaussianMixture(
        n_components=selected_k,
        covariance_type="diag",
        reg_covar=1e-5,
        max_iter=300,
        n_init=3,
        random_state=RANDOM_STATE,
    ).fit(processed)
    feature_names = list(preprocessor.get_feature_names_out())
    labels, probabilities, canonical_raw_order = canonicalize_membership(
        model, processed, feature_names
    )
    confidence = probabilities.max(axis=1)
    profile = cleaned_profile_frame(X)
    segment_names = build_segment_names(profile, labels)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(processed)
    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "pca": pca,
        "canonical_raw_order": canonical_raw_order,
        "segment_names": segment_names,
        "input_columns": INPUT_COLUMNS,
    }
    model_path = model_dir / CLUSTER_MODEL_FILE
    joblib.dump(bundle, model_path, compress=3)

    training_profile = build_cluster_profile(
        profile, labels, confidence, segment_names
    )
    delinquency = pd.DataFrame(
        {
            "ClusterID": labels,
            "SeriousDlqin2yrs": y.to_numpy(),
            "ClusterConfidence": confidence,
        }
    ).groupby("ClusterID").agg(
        CustomerCount=("SeriousDlqin2yrs", "size"),
        SeriousDlqin2yrsCount=("SeriousDlqin2yrs", "sum"),
        SeriousDlqin2yrsRate=("SeriousDlqin2yrs", "mean"),
        ClusterConfidenceMean=("ClusterConfidence", "mean"),
    ).reset_index()
    delinquency.insert(
        1, "SegmentName", delinquency["ClusterID"].map(segment_names)
    )
    delinquency["Share"] = delinquency["CustomerCount"] / len(X)

    comparison.to_csv(
        result_dir / COMPARISON_FILE, index=False, encoding="utf-8-sig"
    )
    delinquency.to_csv(
        result_dir / DELINQUENCY_FILE, index=False, encoding="utf-8-sig"
    )
    training_profile.to_csv(
        result_dir / "05_학습데이터_군집특성요약.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "purpose": "customer segmentation; target excluded from clustering fit",
        "algorithm": "Gaussian Mixture Model",
        "covariance_type": "diag",
        "n_clusters": selected_k,
        "cluster_id_order": "ascending feature-based credit stress",
        "segment_names": {str(key): value for key, value in segment_names.items()},
        "input_columns": INPUT_COLUMNS,
        "processed_columns": CLUSTER_COLUMNS,
        "preprocessing": {
            "special_values": "96/98 delinquency values converted to missing with indicator",
            "age_zero": "converted to missing",
            "imputation": "median fitted on training data",
            "winsorization": "0.1% and 99.9% training quantiles",
            "skew_transform": "log1p on nonnegative skewed features",
            "scaling": "RobustScaler, 10th to 90th percentile range",
        },
        "selection_rule": "GMM chosen for probabilistic membership; minimum BIC among converged GMM candidates with minimum cluster share >= 2%",
        "selection_sample_rows": int(comparison["SelectionRows"].iloc[0]),
        "silhouette_sample_rows": int(comparison["SilhouetteRows"].iloc[0]),
        "candidate_clusters": [min_k, max_k],
        "target_used_for_fit": False,
        "target_used_for_posthoc_interpretation": True,
        "used_as_classifier_feature": False,
        "classifier_feature_note": "segmentation is reported separately so the previously validated HGB score and threshold remain unchanged",
        "training_rows": int(len(X)),
        "pca_explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (model_dir / CLUSTER_METADATA_FILE).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    generate_training_figures(delinquency, comparison, figure_dir, selected_k)

    print(f"군집 모델 저장 완료: {model_path}")
    print(f"선택된 군집 수: {selected_k}")
    for cluster_id, name in sorted(segment_names.items()):
        count = int((labels == cluster_id).sum())
        rate = float(delinquency.loc[
            delinquency["ClusterID"].eq(cluster_id), "SeriousDlqin2yrsRate"
        ].iloc[0])
        print(f"  {cluster_id}: {name} | {count:,}명 | 실제 연체율 {rate:.2%}")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default=r"C:\Users\KDS01\Downloads\GiveMeSomeCredit"
    )
    parser.add_argument("--model-dir", default=str(root / "models"))
    parser.add_argument("--result-dir", default=str(root / "result"))
    parser.add_argument("--figure-dir", default=str(root / "figures"))
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--selection-rows", type=int, default=50000)
    parser.add_argument("--silhouette-rows", type=int, default=3000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        args.data_dir,
        args.model_dir,
        args.result_dir,
        args.figure_dir,
        args.min_k,
        args.max_k,
        args.selection_rows,
        args.silhouette_rows,
    )
