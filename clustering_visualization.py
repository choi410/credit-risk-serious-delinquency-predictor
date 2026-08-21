"""PNG visualizations for customer segmentation results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clustering import FEATURE_LABELS_KO, INPUT_COLUMNS, LOG1P_COLUMNS


def configure_plotting():
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Malgun Gothic")


def save_figure(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def _ordered_segments(result):
    return (
        result[["ClusterID", "SegmentName"]]
        .drop_duplicates()
        .sort_values("ClusterID")["SegmentName"]
        .tolist()
    )


def generate_prediction_figures(result, figure_dir, random_state=42):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    order = _ordered_segments(result)
    palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(order))))

    sample_size = min(12000, len(result))
    sample = result.sample(sample_size, random_state=random_state)
    plt.figure(figsize=(11, 7))
    sns.scatterplot(
        data=sample,
        x="ClusterPCA1",
        y="ClusterPCA2",
        hue="SegmentName",
        hue_order=order,
        palette=palette,
        s=14,
        alpha=0.45,
        linewidth=0,
    )
    plt.title("고객 군집 분포 (PCA 2차원 투영)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="고객군", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(figure_dir / "01_군집분포_PCA.png")

    heat_features = [
        "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio",
        "MonthlyIncome",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "age",
        "NumberOfDependents",
    ]
    population = result[heat_features].copy()
    for column in heat_features:
        if column in LOG1P_COLUMNS:
            population[column] = np.log1p(population[column].clip(lower=0))
    transformed = population.assign(SegmentName=result["SegmentName"]).groupby(
        "SegmentName"
    )[heat_features].median().loc[order]
    center = population.median(axis=0)
    scale = (population.quantile(0.75) - population.quantile(0.25)).replace(0, 1)
    standardized = ((transformed - center) / scale).clip(-3, 3)
    standardized = standardized.rename(columns=FEATURE_LABELS_KO)
    plt.figure(figsize=(13, max(4.8, len(order) * 0.75 + 2)))
    sns.heatmap(
        standardized,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        vmin=-3,
        vmax=3,
        cbar_kws={"label": "전체 고객 대비 Robust Z-score"},
    )
    plt.title("군집별 신용·재무 특성 비교")
    plt.xlabel("특성")
    plt.ylabel("고객군")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    save_figure(figure_dir / "02_군집별_특성_히트맵.png")

    counts = result["SegmentName"].value_counts().reindex(order)
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False)
    total = counts.sum()
    for index, value in enumerate(counts.values):
        ax.text(index, value, f"{value:,}명\n({value / total:.1%})", ha="center", va="bottom")
    plt.title("군집별 고객 수")
    plt.xlabel("고객군")
    plt.ylabel("고객 수")
    plt.xticks(rotation=20, ha="right")
    save_figure(figure_dir / "03_군집별_고객수.png")

    confidence_thresholds = [0.60, 0.75, 0.90, 0.99, 0.999, 0.9999]
    rows = []
    for segment in order:
        values = result.loc[result["SegmentName"].eq(segment), "ClusterConfidence"]
        for threshold in confidence_thresholds:
            rows.append(
                {
                    "SegmentName": segment,
                    "ConfidenceThreshold": threshold,
                    "BelowRate": float((values < threshold).mean()),
                }
            )
    confidence_distribution = pd.DataFrame(rows)
    plt.figure(figsize=(11, 6))
    sns.lineplot(
        data=confidence_distribution,
        x="ConfidenceThreshold",
        y="BelowRate",
        hue="SegmentName",
        hue_order=order,
        palette=palette,
        marker="o",
    )
    plt.title("군집 소속 신뢰도 누적 분포")
    plt.xlabel("신뢰도 기준값")
    plt.ylabel("기준값 미만 고객 비율")
    plt.gca().yaxis.set_major_formatter(lambda value, position: f"{value:.1%}")
    plt.legend(title="고객군")
    save_figure(figure_dir / "05_군집신뢰도_분포.png")


def generate_training_figures(delinquency, comparison, figure_dir, selected_k):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_plotting()

    order = delinquency.sort_values("ClusterID")["SegmentName"].tolist()
    palette = dict(zip(order, sns.color_palette("tab10", n_colors=len(order))))
    plot_data = delinquency.set_index("SegmentName").loc[order].reset_index()
    plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=plot_data,
        x="SegmentName",
        y="SeriousDlqin2yrsRate",
        hue="SegmentName",
        palette=palette,
        legend=False,
    )
    for patch, value in zip(ax.patches, plot_data["SeriousDlqin2yrsRate"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height(),
            f"{value:.1%}",
            ha="center",
            va="bottom",
        )
    plt.title("군집별 실제 심각한 연체율 (학습 데이터)")
    plt.xlabel("고객군")
    plt.ylabel("SeriousDlqin2yrs 비율")
    plt.gca().yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    plt.xticks(rotation=20, ha="right")
    save_figure(figure_dir / "04_군집별_실제연체율.png")

    kmeans = comparison[comparison["Algorithm"].eq("K-Means")].sort_values("Clusters")
    gmm = comparison[comparison["Algorithm"].eq("GMM")].sort_values("Clusters")
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].plot(kmeans["Clusters"], kmeans["Silhouette"], marker="o", label="K-Means")
    axes[0].plot(gmm["Clusters"], gmm["Silhouette"], marker="s", label="GMM")
    axes[0].axvline(selected_k, color="gray", linestyle="--", label=f"선택 k={selected_k}")
    axes[0].set_title("군집 수별 Silhouette Score")
    axes[0].set_xlabel("군집 수 k")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].legend()

    axes[1].plot(gmm["Clusters"], gmm["BICPerRow"], marker="o", label="BIC/row")
    axes[1].plot(gmm["Clusters"], gmm["AICPerRow"], marker="s", label="AIC/row")
    axes[1].axvline(selected_k, color="gray", linestyle="--", label=f"선택 k={selected_k}")
    axes[1].set_title("GMM 군집 수별 정보기준")
    axes[1].set_xlabel("군집 수 k")
    axes[1].set_ylabel("낮을수록 우수")
    axes[1].legend()

    axes[2].plot(
        kmeans["Clusters"],
        kmeans["MinClusterShare"],
        marker="o",
        label="K-Means 최소 군집 비율",
    )
    axes[2].plot(
        gmm["Clusters"],
        gmm["MinClusterShare"],
        marker="o",
        label="GMM 최소 군집 비율",
    )
    axes[2].axhline(0.02, color="red", linestyle="--", label="최소 허용 2%")
    axes[2].axvline(selected_k, color="gray", linestyle="--", label=f"선택 k={selected_k}")
    axes[2].set_title("최소 군집 비율 제약")
    axes[2].set_xlabel("군집 수 k")
    axes[2].set_ylabel("가장 작은 군집의 비율")
    axes[2].yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    axes[2].legend()
    figure.suptitle("클러스터링 모델 및 군집 수 선정 근거")
    save_figure(figure_dir / "06_군집수_선정지표.png")
