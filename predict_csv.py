"""Rank customers by risk score and split the results into review groups."""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd


ALL_FILE = "01_전체고객_위험순위.csv"
REVIEW_FILE = "02_추가심사대상.csv"
PASS_FILE = "03_1차기준통과.csv"
SUMMARY_FILE = "00_결과요약.csv"


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


def build_result(X, customer_id, risk_score, threshold):
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
                "RiskScore 내림차순",
                "추가 심사 대상은 실제 대출 거절 확정이 아님",
            ],
        }
    )
    summary.to_csv(output_dir / SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return review, initial_pass


def main(input_path, output_dir, model_dir):
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
    result = build_result(X, customer_id, risk_score, float(metadata["threshold"]))
    review, initial_pass = write_results(result, output_dir, metadata, input_path)

    print("\n예측 및 CSV 분리 완료")
    print(f"전체 고객: {len(result):,}명")
    print(f"추가 심사 대상: {len(review):,}명 ({len(review) / len(result):.2%})")
    print(
        f"1차 기준 통과: {len(initial_pass):,}명 "
        f"({len(initial_pass) / len(result):.2%})"
    )
    print(f"적용 임계값: {float(metadata['threshold']):.12f}")
    print(f"결과 폴더: {Path(output_dir).resolve()}")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(root / "input" / "customer_data.csv"))
    parser.add_argument("--output-dir", default=str(root / "result"))
    parser.add_argument("--model-dir", default=str(root / "models"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.output_dir, args.model_dir)
