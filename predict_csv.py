"""Rank customers by risk score and split the results into review groups."""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from clustering import load_clustering_bundle, predict_clusters
from clustering_visualization import generate_prediction_figures


ALL_FILE = "01_전체고객_위험순위.csv"
REVIEW_FILE = "02_추가심사대상.csv"
PASS_FILE = "03_1차기준통과.csv"
SUMMARY_FILE = "00_결과요약.csv"
CLUSTER_CUSTOMER_FILE = "04_고객별_군집결과.csv"
CLUSTER_SUMMARY_FILE = "05_군집별_특성요약.csv"


def load_metadata(model_dir):
    metadata_path = Path(model_dir) / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"모델 메타데이터가 없습니다: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def prepare_input(input_path, required_columns):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {input_path}")

    raw = pd.read_csv(input_path)
    if raw.empty:
        raise ValueError("입력 CSV에 고객 데이터가 없습니다.")

    if "Id" in raw.columns:
        customer_id = raw["Id"].copy()
    elif "Unnamed: 0" in raw.columns:
        customer_id = raw["Unnamed: 0"].copy()
    else:
        customer_id = pd.Series(np.arange(1, len(raw) + 1), name="Id")

    missing_columns = [column for column in required_columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"필수 입력 컬럼이 없습니다: {missing_columns}")

    original = raw[required_columns].copy()
    numeric = original.apply(pd.to_numeric, errors="coerce")
    invalid = original.notna() & numeric.isna()
    if invalid.any().any():
        locations = []
        for column in invalid.columns:
            count = int(invalid[column].sum())
            if count:
                locations.append(f"{column}({count}건)")
        raise ValueError("숫자로 변환할 수 없는 값이 있습니다: " + ", ".join(locations))

    negative_columns = [
        column for column in numeric.columns if (numeric[column].dropna() < 0).any()
    ]
    if negative_columns:
        raise ValueError(f"음수가 허용되지 않는 컬럼이 있습니다: {negative_columns}")

    return numeric, customer_id.reset_index(drop=True)


def build_result(
    X,
    customer_id,
    risk_score,
    threshold,
    cluster_labels,
    segment_names,
    cluster_confidence,
    cluster_probabilities,
    cluster_coordinates,
):
    result = X.reset_index(drop=True).copy()
    result.insert(0, "Id", customer_id)
    result.insert(1, "RiskScore", risk_score)
    result.insert(2, "RiskScorePercent", risk_score * 100)
    result.insert(3, "Threshold", threshold)
    result.insert(4, "ReviewRequired", (risk_score >= threshold).astype(int))
    result.insert(
        5,
        "Decision",
        np.where(
            result["ReviewRequired"].eq(1),
            "대출 보류(추가 심사)",
            "1차 기준 통과",
        ),
    )
    result.insert(6, "ClusterID", cluster_labels)
    result.insert(7, "SegmentName", segment_names)
    result.insert(8, "ClusterConfidence", cluster_confidence)
    result.insert(9, "ClusterPCA1", cluster_coordinates[:, 0])
    result.insert(10, "ClusterPCA2", cluster_coordinates[:, 1])
    for cluster_id in reversed(range(cluster_probabilities.shape[1])):
        result.insert(
            11,
            f"ClusterProb_{cluster_id}",
            cluster_probabilities[:, cluster_id],
        )
    result = result.sort_values(
        ["RiskScore", "Id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "RiskRank", np.arange(1, len(result) + 1))
    return result


def write_results(result, output_dir, metadata, input_path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    review = result[result["ReviewRequired"].eq(1)].copy()
    initial_pass = result[result["ReviewRequired"].eq(0)].copy()
    threshold = float(metadata["threshold"])

    result.to_csv(output_dir / ALL_FILE, index=False, encoding="utf-8-sig")
    review.to_csv(output_dir / REVIEW_FILE, index=False, encoding="utf-8-sig")
    initial_pass.to_csv(output_dir / PASS_FILE, index=False, encoding="utf-8-sig")

    probability_columns = [
        column for column in result.columns if column.startswith("ClusterProb_")
    ]
    cluster_customer_columns = [
        "RiskRank",
        "Id",
        "ClusterID",
        "SegmentName",
        "ClusterConfidence",
        *probability_columns,
        "ClusterPCA1",
        "ClusterPCA2",
        "RiskScore",
        "ReviewRequired",
        "Decision",
    ]
    result[cluster_customer_columns].to_csv(
        output_dir / CLUSTER_CUSTOMER_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    grouped = result.groupby(["ClusterID", "SegmentName"], sort=True)
    cluster_summary = grouped.agg(
        CustomerCount=("Id", "size"),
        MeanRiskScore=("RiskScore", "mean"),
        MedianRiskScore=("RiskScore", "median"),
        ReviewRate=("ReviewRequired", "mean"),
        MeanClusterConfidence=("ClusterConfidence", "mean"),
    )
    cluster_summary["Share"] = cluster_summary["CustomerCount"] / len(result)
    for column in metadata["input_columns"]:
        cluster_summary[f"Median_{column}"] = grouped[column].median()
    cluster_summary.reset_index().to_csv(
        output_dir / CLUSTER_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        {
            "항목": [
                "입력 파일",
                "전체 고객 수",
                "추가 심사 대상 수",
                "추가 심사 대상 비율",
                "1차 기준 통과 수",
                "1차 기준 통과 비율",
                "적용 임계값",
                "사용 모델",
                "클러스터링 모델",
                "군집 수",
                "정렬 기준",
                "판정 주의사항",
            ],
            "값": [
                Path(input_path).name,
                len(result),
                len(review),
                f"{len(review) / len(result):.2%}",
                len(initial_pass),
                f"{len(initial_pass) / len(result):.2%}",
                f"{threshold:.12f}",
                metadata["model"],
                "Gaussian Mixture Model",
                int(result["ClusterID"].nunique()),
                "RiskScore 내림차순",
                "추가 심사 대상은 실제 대출 거절 확정이 아님",
            ],
        }
    )
    summary.to_csv(output_dir / SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return review, initial_pass


def main(input_path, output_dir, model_dir, figure_dir):
    model_dir = Path(model_dir)
    model_path = model_dir / "credit_risk_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"저장 모델이 없습니다: {model_path}\n"
            "먼저 train_and_export_model.py를 실행하세요."
        )

    metadata = load_metadata(model_dir)
    X, customer_id = prepare_input(input_path, metadata["input_columns"])
    model = joblib.load(model_path)
    risk_score = model.predict_proba(X)[:, 1]
    cluster_bundle = load_clustering_bundle(model_dir)
    (
        cluster_labels,
        segment_names,
        cluster_confidence,
        cluster_probabilities,
        cluster_coordinates,
    ) = predict_clusters(cluster_bundle, X)
    result = build_result(
        X,
        customer_id,
        risk_score,
        float(metadata["threshold"]),
        cluster_labels,
        segment_names,
        cluster_confidence,
        cluster_probabilities,
        cluster_coordinates,
    )
    review, initial_pass = write_results(result, output_dir, metadata, input_path)
    generate_prediction_figures(result, figure_dir)

    print("\n예측 및 CSV 분리 완료")
    print(f"전체 고객: {len(result):,}명")
    print(f"추가 심사 대상: {len(review):,}명 ({len(review) / len(result):.2%})")
    print(
        f"1차 기준 통과: {len(initial_pass):,}명 "
        f"({len(initial_pass) / len(result):.2%})"
    )
    print(f"적용 임계값: {float(metadata['threshold']):.12f}")
    print(f"군집 수: {result['ClusterID'].nunique()}개")
    print(f"결과 폴더: {Path(output_dir).resolve()}")
    print(f"군집 이미지 폴더: {Path(figure_dir).resolve()}")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(root / "input" / "customer_data.csv"))
    parser.add_argument("--output-dir", default=str(root / "result"))
    parser.add_argument("--model-dir", default=str(root / "models"))
    parser.add_argument("--figure-dir", default=str(root / "figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output_dir, args.model_dir, args.figure_dir)
