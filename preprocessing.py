"""Give Me Some Credit CSV prediction program preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
THRESHOLD = 0.07424190003697577

INPUT_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

LATE_COLUMNS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Clean special values and add the three selected indicator features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()

        result["late_special_value"] = (
            result[LATE_COLUMNS].isin([96, 98]).any(axis=1).astype(int)
        )
        for column in LATE_COLUMNS:
            result[column] = result[column].replace([96, 98], np.nan)

        result["age"] = result["age"].replace(0, np.nan)
        result["MonthlyIncome_missing"] = result["MonthlyIncome"].isna().astype(int)
        result["NumberOfDependents_missing"] = (
            result["NumberOfDependents"].isna().astype(int)
        )
        return result


def make_model_pipeline():
    """Build the selected basic-preprocessing + unweighted HGB pipeline."""
    return Pipeline(
        [
            ("feature_engineering", CreditFeatureEngineer()),
            ("median_imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=250,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def load_training_data(data_dir):
    train_path = Path(data_dir) / "cs-training.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {train_path}")
    train = pd.read_csv(train_path).drop(columns=["Unnamed: 0"], errors="ignore")
    X = train[INPUT_COLUMNS].copy()
    y = train[TARGET].astype(int)
    return X, y, train_path
