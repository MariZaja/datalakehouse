"""Gold — Step 03c: K-Means Clustering on LDA Entity Means.

Reads gold/lda_reduction/eav/entity_lda_means.csv and clusters entities
using K-Means across all 9 LDA dimensions (audio, eeg, video × lda_1/2/3).
Tries k=2..41, picks the k with the highest mean silhouette score.

With --per-modality: runs separate clusterings for each modality (3 dims each).

Outputs a CSV with cluster assignments and a 2-D UMAP scatter plot.
"""
import argparse
import io
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import yaml
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as project_config
from minio_utils import download_object, upload_csv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gold_kmeans_clustering")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")
_MODALITIES = ("audio", "eeg", "video")
_LDA_DIMS = ("lda_1", "lda_2", "lda_3")
_FEATURE_COLS = [f"{m}_{d}" for m in _MODALITIES for d in _LDA_DIMS]
_MODALITY_COLS = {m: [f"{m}_{d}" for d in _LDA_DIMS] for m in _MODALITIES}
_K_RANGE = range(2, 42)


def _load_means_csv(minio_client, bucket: str, key: str) -> pd.DataFrame:
    data = download_object(minio_client, bucket, key)
    if data is None:
        raise FileNotFoundError(f"Object not found: {bucket}/{key}")
    return pd.read_csv(io.BytesIO(data))


def _run_kmeans(X: np.ndarray, tag: str) -> tuple[np.ndarray, int, dict[int, float]]:
    scores: dict[int, float] = {}
    for k in _K_RANGE:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        scores[k] = silhouette_score(X, labels)
        logger.info("  [%s] k=%d  silhouette=%.4f", tag, k, scores[k])
    above = [k for k, s in scores.items() if s > 0.5]
    best_k = max(above) if above else max(scores, key=scores.__getitem__)
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
    return labels, best_k, scores


def _scatter_ax(ax, coords, labels, entity_ids, title):
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=60)
    plt.colorbar(scatter, ax=ax, label="cluster")
    for i, eid in enumerate(entity_ids):
        ax.annotate(str(eid), coords[i], fontsize=6, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")


def _silhouette_ax(ax, scores, best_k, title):
    ks = list(scores.keys())
    sils = [scores[k] for k in ks]
    bars = ax.bar(ks, sils, color=["tab:orange" if k == best_k else "steelblue" for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("mean silhouette")
    ax.set_title(title)
    ax.set_xticks(ks)
    for bar, val in zip(bars, sils):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.3f}", ha="center", fontsize=8)


def _build_plot_combined(
    df: pd.DataFrame, labels: np.ndarray, best_k: int, scores: dict[int, float]
) -> bytes:
    coords = umap.UMAP(n_components=2, random_state=42).fit_transform(df[_FEATURE_COLS].values)

    fig, (ax_main, ax_sil) = plt.subplots(1, 2, figsize=(14, 6))
    _scatter_ax(ax_main, coords, labels, df["entity_id"], f"K-Means k={best_k} — all modalities (UMAP)")
    _silhouette_ax(ax_sil, scores, best_k, "Silhouette score by k")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _build_plot_per_modality(
    df: pd.DataFrame,
    modality_results: dict[str, tuple[np.ndarray, int, dict[int, float]]],
) -> bytes:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    scatter_row, sil_row = axes[0], axes[1]

    for ax_scatter, ax_sil, modality in zip(scatter_row, sil_row, _MODALITIES):
        labels, best_k, scores = modality_results[modality]
        cols = _MODALITY_COLS[modality]
        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(df[cols].values)
        _scatter_ax(ax_scatter, coords, labels, df["entity_id"], f"{modality}  k={best_k} (UMAP)")
        _silhouette_ax(ax_sil, scores, best_k, f"{modality} — silhouette by k")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold — Step 03c: K-Means Clustering.")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--per-modality", action="store_true",
        help="Run separate K-Means clustering per modality instead of on all 9 dims.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    gold_bucket = cfg["bucket_gold"]

    lda_prefix = cfg.get("lda_reduction", {}).get("output_prefix", "lda_reduction")
    eav_prefix = f"{lda_prefix}/eav"
    input_key = f"{eav_prefix}/entity_lda_means.csv"

    suffix = "_per_modality" if args.per_modality else ""
    output_clusters_key = f"{eav_prefix}/entity_clusters{suffix}.csv"
    output_plot_key = f"{eav_prefix}/entity_clusters{suffix}.png"

    logger.info("Loading %s/%s", gold_bucket, input_key)
    df = _load_means_csv(minio_client, gold_bucket, input_key)

    missing = [c for c in _FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    before = len(df)
    df = df.dropna(subset=_FEATURE_COLS).reset_index(drop=True)
    if len(df) < before:
        logger.warning("Dropped %d rows with missing LDA values", before - len(df))

    result = df[["entity_id"]].copy()

    if args.per_modality:
        modality_results: dict[str, tuple] = {}
        for modality in _MODALITIES:
            cols = _MODALITY_COLS[modality]
            X = StandardScaler().fit_transform(df[cols].values)
            logger.info("Selecting k for modality=%s (k=%d..%d)", modality, min(_K_RANGE), max(_K_RANGE))
            labels, best_k, scores = _run_kmeans(X, modality)
            logger.info("  → best k=%d  (silhouette=%.4f)", best_k, scores[best_k])
            for cid in sorted(set(labels)):
                members = df.loc[labels == cid, "entity_id"].tolist()
                logger.info("    cluster %d (%d): %s", cid, len(members), members)
            modality_results[modality] = (labels, best_k, scores)
            result[f"{modality}_cluster"] = labels
            result[f"{modality}_best_k"] = best_k
            result[f"{modality}_silhouette"] = round(scores[best_k], 4)

        plot_bytes = _build_plot_per_modality(df, modality_results)

    else:
        X = StandardScaler().fit_transform(df[_FEATURE_COLS].values)
        logger.info("Selecting k via silhouette score (k=%d..%d) on %d entities", min(_K_RANGE), max(_K_RANGE), len(df))
        labels, best_k, scores = _run_kmeans(X, "all")
        logger.info("Best k=%d  (silhouette=%.4f)", best_k, scores[best_k])
        for cid in sorted(set(labels)):
            members = df.loc[labels == cid, "entity_id"].tolist()
            logger.info("  cluster %d (%d): %s", cid, len(members), members)
        result["cluster"] = labels
        result["best_k"] = best_k
        result["silhouette"] = round(scores[best_k], 4)

        plot_bytes = _build_plot_combined(df, labels, best_k, scores)

    upload_csv(minio_client, gold_bucket, output_clusters_key, result)
    logger.info("Cluster assignments → %s/%s", gold_bucket, output_clusters_key)

    minio_client.put_object(
        gold_bucket, output_plot_key,
        io.BytesIO(plot_bytes), length=len(plot_bytes),
        content_type="image/png",
    )
    logger.info("Scatter plot → %s/%s", gold_bucket, output_plot_key)


if __name__ == "__main__":
    main()
