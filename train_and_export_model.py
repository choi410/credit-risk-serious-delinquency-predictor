"""Fit the frozen final pipeline on all labeled rows and export it."""

from pathlib import Path
import argparse
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from preprocessing import (
    INPUT_COLUMNS,
    RANDOM_STATE,
    TARGET,
    THRESHOLD,
    load_training_data,
    make_model_pipeline,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(data_dir, model_dir):
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    X, y, train_path = load_training_data(data_dir)

    model = make_model_pipeline()
    print(f"모델 학습 시작: {len(X):,}명", flush=True)
    model.fit(X, y)

    model_path = model_dir / "credit_risk_model.joblib"
    joblib.dump(model, model_path, compress=3)

    metadata = {
        "program": "개인의 신용 및 재무정보를 활용한 향후 2년 내 심각한 연체 위험 예측 프로그램",
        "target": TARGET,
        "model": "Histogram Gradient Boosting",
        "preprocessing": "basic: special-value cleaning + 3 indicators + median imputation",
        "imbalance_strategy": "none",
        "calibration": "raw",
        "threshold": THRESHOLD,
        "threshold_rule": "maximum precision subject to OOF recall >= 0.75",
        "threshold_interpretation": "project review cutoff, not a legal approval threshold or literal 7.42% default probability",
        "input_columns": INPUT_COLUMNS,
        "training_rows": int(len(X)),
        "training_positive_rate": float(y.mean()),
        "random_state": RANDOM_STATE,
        "holdout_metrics_confirmatory": {
            "roc_auc": 0.8684379335640181,
            "pr_auc": 0.4102092639488756,
            "recall": 0.7546134663341646,
            "precision": 0.22913827048311375,
            "f1": 0.35153345724907065,
        },
        "labels": {
            "1": "대출 보류(추가 심사)",
            "0": "1차 기준 통과",
        },
        "training_file_sha256": sha256_file(train_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"모델 저장 완료: {model_path}")
    print(f"메타데이터 저장 완료: {metadata_path}")


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=r"C:\Users\KDS01\Downloads\GiveMeSomeCredit",
    )
    parser.add_argument("--model-dir", default=str(root / "models"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.model_dir)
