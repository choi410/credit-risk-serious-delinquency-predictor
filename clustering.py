"""Leakage-safe customer segmentation helpers for Give Me Some Credit."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from preprocessing import CreditFeatureEngineer, INPUT_COLUMNS, LATE_COLUMNS


CLUSTER_MODEL_FILE = "customer_segmentation_gmm.joblib"
CLUSTER_METADATA_FILE = "clustering_metadata.json"

INDICATOR_COLUMNS = [
    "late_special_value",
    "MonthlyIncome_missing",
    "NumberOfDependents_missing",
]
CLUSTER_COLUMNS = INPUT_COLUMNS
PROFILE_COLUMNS = INPUT_COLUMNS + INDICATOR_COLUMNS

LOG1P_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
CLIP_COLUMNS = [column for column in INPUT_COLUMNS if column != "age"] + ["age"]

FEATURE_LABELS_KO = {
    "RevolvingUtilizationOfUnsecuredLines": "신용한도 사용률",
    "age": "연령",
    "NumberOfTime30-59DaysPastDueNotWorse": "30~59일 연체",
    "DebtRatio": "부채비율",
    "MonthlyIncome": "월소득",
    "NumberOfOpenCreditLinesAndLoans": "신용계좌 수",
    "NumberOfTimes90DaysLate": "90일 이상 연체",
    "NumberRealEstateLoansOrLines": "부동산대출 수",
    "NumberOfTime60-89DaysPastDueNotWorse": "60~89일 연체",
    "NumberOfDependents": "부양가족 수",
    "late_special_value": "연체 특수값",
    "MonthlyIncome_missing": "소득 결측",
    "NumberOfDependents_missing": "부양가족 결측",
}


def cleaned_profile_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Apply the same special-value cleaning used by the supervised model."""
    missing = [column for column in INPUT_COLUMNS if column not in X.columns]
    if missing:
        raise ValueError(f"클러스터링 필수 컬럼이 없습니다: {missing}")
    frame = CreditFeatureEngineer().transform(X[INPUT_COLUMNS].copy())
    return frame[PROFILE_COLUMNS].replace([np.inf, -np.inf], np.nan)


class CreditClusteringPreprocessor(BaseEstimator, TransformerMixin):
    """Impute, winsorize, log-transform, and robust-scale credit features."""

    def __init__(self, clip_lower=0.001, clip_upper=0.999):
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper

    def fit(self, X, y=None):
        frame = cleaned_profile_frame(X)[CLUSTER_COLUMNS]
        self.feature_names_out_ = np.asarray(CLUSTER_COLUMNS, dtype=object)
        self.imputer_ = SimpleImputer(strategy="median")
        imputed = pd.DataFrame(
            self.imputer_.fit_transform(frame),
            columns=CLUSTER_COLUMNS,
            index=frame.index,
        )
        self.lower_bounds_ = imputed[CLIP_COLUMNS].quantile(self.clip_lower)
        self.upper_bounds_ = imputed[CLIP_COLUMNS].quantile(self.clip_upper)
        transformed = self._transform_values(imputed)
        self.scaler_ = RobustScaler(quantile_range=(10.0, 90.0))
        self.scaler_.fit(transformed)
        return self

    def transform(self, X):
        frame = cleaned_profile_frame(X)[CLUSTER_COLUMNS]
        imputed = pd.DataFrame(
            self.imputer_.transform(frame),
            columns=CLUSTER_COLUMNS,
            index=frame.index,
        )
        transformed = self._transform_values(imputed)
        return self.scaler_.transform(transformed)

    def _transform_values(self, frame):
        transformed = frame.copy()
        transformed[CLIP_COLUMNS] = transformed[CLIP_COLUMNS].clip(
            lower=self.lower_bounds_, upper=self.upper_bounds_, axis="columns"
        )
        transformed[LOG1P_COLUMNS] = np.log1p(
            transformed[LOG1P_COLUMNS].clip(lower=0)
        )
        return transformed[CLUSTER_COLUMNS]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def canonicalize_membership(model, processed, feature_names):
    """Order cluster IDs from lower to higher feature-based credit stress."""
    raw_labels = model.predict(processed)
    raw_probabilities = model.predict_proba(processed)
    processed_frame = pd.DataFrame(processed, columns=feature_names)
    processed_frame["raw_cluster"] = raw_labels

    stress_columns = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        *LATE_COLUMNS,
    ]
    stress = processed_frame.groupby("raw_cluster")[stress_columns].mean().mean(axis=1)
    canonical_raw_order = [int(value) for value in stress.sort_values().index]
    raw_to_canonical = {
        raw_id: canonical_id
        for canonical_id, raw_id in enumerate(canonical_raw_order)
    }
    canonical_labels = np.asarray(
        [raw_to_canonical[int(label)] for label in raw_labels], dtype=int
    )
    canonical_probabilities = raw_probabilities[:, canonical_raw_order]
    return canonical_labels, canonical_probabilities, canonical_raw_order


def build_segment_names(profile_frame, cluster_labels):
    """Create unique, feature-based Korean labels without using the target."""
    frame = profile_frame.copy()
    frame["ClusterID"] = cluster_labels
    medians = frame.groupby("ClusterID")[INPUT_COLUMNS].median()
    missing_means = frame.groupby("ClusterID")[INDICATOR_COLUMNS].mean()
    cluster_ids = list(medians.index.astype(int))
    available = set(cluster_ids)
    names = {}

    late_score = np.log1p(medians[LATE_COLUMNS]).sum(axis=1)
    utilization_score = np.log1p(
        medians["RevolvingUtilizationOfUnsecuredLines"].clip(lower=0)
    )
    debt_score = np.log1p(medians["DebtRatio"].clip(lower=0))
    stability_score = late_score + utilization_score + 0.5 * debt_score

    def assign_min(series, label):
        candidates = series.loc[sorted(available)]
        selected = int(candidates.idxmin())
        names[selected] = label
        available.remove(selected)

    def assign_max(series, label):
        candidates = series.loc[sorted(available)]
        selected = int(candidates.idxmax())
        names[selected] = label
        available.remove(selected)

    if available:
        assign_min(stability_score, "상대적 안정군")
    if available:
        assign_max(late_score, "연체 이력 집중군")
    if available:
        assign_max(utilization_score, "한도 사용 집중군")
    if available:
        assign_max(debt_score, "부채 부담군")
    if available:
        missing_score = missing_means[
            ["MonthlyIncome_missing", "NumberOfDependents_missing"]
        ].sum(axis=1)
        assign_max(missing_score, "소득정보 부족군")
    for number, cluster_id in enumerate(sorted(available), start=1):
        suffix = "" if len(available) == 1 else f" {number}"
        names[int(cluster_id)] = f"일반 신용활동군{suffix}"

    return names


def load_clustering_bundle(model_dir):
    model_path = Path(model_dir) / CLUSTER_MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(
            f"군집 모델이 없습니다: {model_path}\n"
            "먼저 train_clustering.py를 실행하세요."
        )
    return joblib.load(model_path)


def predict_clusters(bundle, X):
    processed = bundle["preprocessor"].transform(X)
    raw_probabilities = bundle["model"].predict_proba(processed)
    order = bundle["canonical_raw_order"]
    probabilities = raw_probabilities[:, order]
    labels = probabilities.argmax(axis=1).astype(int)
    confidence = probabilities.max(axis=1)
    coordinates = bundle["pca"].transform(processed)
    segment_names = pd.Series(labels).map(bundle["segment_names"]).to_numpy()
    return labels, segment_names, confidence, probabilities, coordinates
