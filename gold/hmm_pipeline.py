"""Gold — Step 03: HMM Emotion Recognition Pipeline.

Loads gold-layer features + data-quality scores for one EAV entity,
aligns them, fits a QualityAwareGaussianHMM, and evaluates on a held-out trial split.

Typical usage:
    python gold/hmm_pipeline.py --entity subject01 --modalities audio video eeg

    # Multi-seed comparison (quality vs. no-quality, seeds 0..49):
    python gold/hmm_pipeline.py --entity subject01 --n-seeds 50

Data paths expected in the gold MinIO bucket:
    feature_extraction/eav/<entity>/<entity>_<modality>.parquet
    data_quality/eav/files/<entity>_data_quality.parquet
"""
import argparse
import io
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as project_config
from minio_utils import download_object
from ml import QualityAwareGaussianHMM

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hmm_pipeline")

_DEFAULT_CONFIG = str(Path(__file__).resolve().parent.parent / "pipeline_config.yaml")

# Quality flag → numeric score fed into the emission variance.
_FLAG_SCORE: Dict[str, float] = {"GOOD": 1.0, "NOISY": 0.5, "BAD": 0.1}
_DEFAULT_QUALITY = 0.5

# Columns that are metadata, not features.
_META_COLS = frozenset(
    {"file_id", "entity_id", "trial_id", "window_id", "window_start_s", "window_end_s"}
)

# EAV emotion classes (order defines integer labels).
_EAV_EMOTIONS = ["Anger", "Happiness", "Calm", "Sadness"]

# Modality → flag column name in data_quality parquet.
_MODALITY_FLAG_COL: Dict[str, str] = {
    "audio": "audio_flag",
    "video": "video_flag",
    "eeg":   "eeg_flag",
}

# EEG trial_id in feature parquets is 0-based (instance index in .mat);
# add this offset to align with 1-based audio/video trial IDs and data_quality.
_EEG_TRIAL_OFFSET = 1


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_parquet(minio_client, bucket: str, key: str) -> Optional[pd.DataFrame]:
    data = download_object(minio_client, bucket, key)
    if data is None:
        return None
    try:
        return pd.read_parquet(io.BytesIO(data))
    except Exception as e:
        logger.error("Failed to parse parquet %s/%s: %s", bucket, key, e)
        return None


def _flatten_tuple_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns stored as tuple-string literals (mne-features MultiIndex artefact).

    e.g. "('window_start_s', '')" → "window_start_s"
         "('pow_freq_bands', 'ch0_band0')" → "pow_freq_bands_ch0_band0"
    """
    import ast
    new_cols = []
    changed = False
    for col in df.columns:
        if isinstance(col, str) and col.startswith("("):
            try:
                parts = ast.literal_eval(col)
                flat = "_".join(str(p) for p in parts if p)
                new_cols.append(flat)
                changed = True
                continue
            except Exception:
                pass
        new_cols.append(col)
    if changed:
        df = df.copy()
        df.columns = new_cols
    return df


def _feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _META_COLS]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(
    minio_client,
    gold_bucket: str,
    entity_id: str,
    modality: str,
    feat_prefix: str,
) -> pd.DataFrame:
    key = f"{feat_prefix}/eav/{entity_id}/{entity_id}_{modality}.parquet"
    df = _load_parquet(minio_client, gold_bucket, key)
    if df is None:
        raise FileNotFoundError(f"Feature parquet not found: {gold_bucket}/{key}")
    df = _flatten_tuple_columns(df)
    if modality == "eeg" and "trial_id" in df.columns:
        df = df.copy()
        df["trial_id"] = df["trial_id"] + _EEG_TRIAL_OFFSET
    return df


def load_data_quality(
    minio_client,
    gold_bucket: str,
    entity_id: str,
    dq_prefix: str,
) -> pd.DataFrame:
    key = f"{dq_prefix}/eav/files/{entity_id}_data_quality.parquet"
    df = _load_parquet(minio_client, gold_bucket, key)
    if df is None:
        raise FileNotFoundError(f"Data quality parquet not found: {gold_bucket}/{key}")
    # window_id is "001_00": trial_id (1-based) + "_" + quality_win (floor-seconds).
    parsed = df["window_id"].str.split("_", n=1, expand=True)
    df = df.copy()
    df["_trial_id"] = parsed[0].astype(int)
    df["_quality_win"] = parsed[1].astype(int)
    return df


# ── Quality + emotion alignment ───────────────────────────────────────────────

def _flag_to_score(flag) -> float:
    if pd.isna(flag) or str(flag) not in _FLAG_SCORE:
        return _DEFAULT_QUALITY
    return _FLAG_SCORE[str(flag)]


def attach_quality_and_emotion(
    feat_df: pd.DataFrame,
    dq_df: pd.DataFrame,
    flag_col: str,
    window_size_s: float = 0.3,
) -> pd.DataFrame:
    """Left-join quality flag and emotion label onto feature rows.

    Quality windows in data_quality use floor(window_start_s) as the key,
    matching the 1-second buckets created during quality flag aggregation.
    If window_start_s is absent (older parquet), it is reconstructed from
    window_id using window_size_s.
    """
    feat_df = feat_df.copy()
    if "window_start_s" not in feat_df.columns:
        logger.warning(
            "window_start_s missing — reconstructing from window_id with window_size_s=%.3f",
            window_size_s,
        )
        feat_df["window_start_s"] = (feat_df["window_id"] - 1) * window_size_s
    feat_df["_quality_win"] = feat_df["window_start_s"].apply(lambda s: int(float(s)))

    qmap = (
        dq_df.set_index(["_trial_id", "_quality_win"])[[flag_col, "window_emotion"]]
        if flag_col in dq_df.columns
        else dq_df.set_index(["_trial_id", "_quality_win"])[["window_emotion"]]
    )

    feat_df = feat_df.join(qmap, on=["trial_id", "_quality_win"], how="left")

    if flag_col in feat_df.columns:
        feat_df["quality_score"] = feat_df[flag_col].apply(_flag_to_score)
    else:
        feat_df["quality_score"] = _DEFAULT_QUALITY

    return feat_df


# ── Sequence construction ─────────────────────────────────────────────────────

def build_sequences(
    feat_dfs: Dict[str, pd.DataFrame],
    feat_cols: Dict[str, List[str]],
    emotion_to_idx: Dict[str, int],
    use_quality: bool = True,
) -> List[Dict]:
    """Group feature windows by trial into HMM sequence dicts.

    Each sequence dict contains:
        obs_<modality>: np.ndarray (T, dim)  — may contain NaN; imputed later
        q_<modality>:   np.ndarray (T,)
        y:              np.ndarray (T,) integer labels
        trial_id:       int
    """
    # Only trials present in all modalities and with a valid emotion label.
    trial_sets = [set(df["trial_id"].dropna().unique()) for df in feat_dfs.values()]
    common_trials = sorted(trial_sets[0].intersection(*trial_sets[1:]))

    # Build emotion label per trial from any modality's data_quality join.
    first_df = next(iter(feat_dfs.values()))
    trial_emotion: Dict[int, int] = {}
    for trial_id, grp in first_df.groupby("trial_id"):
        emotions = grp["window_emotion"].dropna().unique()
        if len(emotions) == 1 and emotions[0] in emotion_to_idx:
            trial_emotion[int(trial_id)] = emotion_to_idx[emotions[0]]

    labeled_trials = [t for t in common_trials if t in trial_emotion]
    logger.info("%d trials with emotion labels out of %d common", len(labeled_trials), len(common_trials))

    sequences = []
    for trial_id in labeled_trials:
        label = trial_emotion[trial_id]
        modality_data = {}
        skip = False

        window_counts: Dict[str, int] = {}
        first_starts: Dict[str, List[float]] = {}

        for m, df in feat_dfs.items():
            rows = df[df["trial_id"] == trial_id].sort_values("window_start_s").reset_index(drop=True)
            if rows.empty:
                skip = True
                break
            window_counts[m] = len(rows)
            if "window_start_s" in rows.columns:
                first_starts[m] = rows["window_start_s"].iloc[:3].tolist()
            obs = rows[feat_cols[m]].values.astype(float)
            q = (
                rows["quality_score"].values.astype(float)
                if use_quality
                else np.ones(len(rows), dtype=float)
            )
            if use_quality:
                q = np.nan_to_num(q, nan=_DEFAULT_QUALITY)
            modality_data[m] = (obs, q)

        if skip:
            continue

        # Modality sync diagnostics: warn if window counts differ across modalities.
        if len(set(window_counts.values())) > 1:
            logger.warning(
                "Trial %s has unequal window counts across modalities: %s",
                trial_id, window_counts,
            )
            for m, ts in first_starts.items():
                logger.debug(
                    "  Trial %s modality '%s' first window_start_s: %s", trial_id, m, ts
                )

        # Align all modalities to the shortest window count.
        T = min(v[0].shape[0] for v in modality_data.values())
        if T == 0:
            continue

        seq: Dict = {"trial_id": trial_id, "y": np.full(T, label, dtype=int)}
        for m, (obs, q) in modality_data.items():
            seq[f"obs_{m}"] = obs[:T]
            seq[f"q_{m}"] = q[:T]
        sequences.append(seq)

    return sequences


# ── Normalisation ─────────────────────────────────────────────────────────────

def fit_imputers(
    sequences: List[Dict],
    modalities: List[str],
) -> Dict[str, SimpleImputer]:
    """Fit mean imputers on training sequences. Must be called before fit_scalers."""
    imputers = {}
    for m in modalities:
        X = np.vstack([seq[f"obs_{m}"] for seq in sequences])
        imputer = SimpleImputer(strategy="mean")
        imputer.fit(X)
        imputers[m] = imputer
    return imputers


def apply_imputers(
    sequences: List[Dict],
    imputers: Dict[str, SimpleImputer],
) -> List[Dict]:
    out = []
    for seq in sequences:
        new_seq = {k: v for k, v in seq.items()}
        for m, imputer in imputers.items():
            new_seq[f"obs_{m}"] = imputer.transform(seq[f"obs_{m}"])
        out.append(new_seq)
    return out


def fit_scalers(
    sequences: List[Dict],
    modalities: List[str],
) -> Dict[str, StandardScaler]:
    scalers = {}
    for m in modalities:
        X = np.vstack([seq[f"obs_{m}"] for seq in sequences])
        scaler = StandardScaler()
        scaler.fit(X)
        scalers[m] = scaler
    return scalers


def apply_scalers(
    sequences: List[Dict],
    scalers: Dict[str, StandardScaler],
) -> List[Dict]:
    out = []
    for seq in sequences:
        new_seq = {k: v for k, v in seq.items()}
        for m, scaler in scalers.items():
            new_seq[f"obs_{m}"] = scaler.transform(seq[f"obs_{m}"])
        out.append(new_seq)
    return out


# ── Quality diagnostics ───────────────────────────────────────────────────────

def log_quality_stats(feat_dfs: Dict[str, pd.DataFrame], modalities: List[str]) -> None:
    """Log quality score statistics per modality, overall and broken down by emotion."""
    for m in modalities:
        if m not in feat_dfs or "quality_score" not in feat_dfs[m].columns:
            continue
        df = feat_dfs[m]
        q = df["quality_score"].dropna()
        logger.info(
            "Quality [%s]: mean=%.3f median=%.3f std=%.3f min=%.3f max=%.3f (n=%d windows)",
            m, float(q.mean()), float(q.median()), float(q.std()),
            float(q.min()), float(q.max()), len(q),
        )
        if "window_emotion" in df.columns:
            for emotion, grp in df.groupby("window_emotion"):
                eq = grp["quality_score"].dropna()
                if eq.empty:
                    continue
                logger.info(
                    "  %-10s [%s]: mean=%.3f median=%.3f std=%.3f min=%.3f max=%.3f (n=%d)",
                    emotion, m,
                    float(eq.mean()), float(eq.median()), float(eq.std()),
                    float(eq.min()), float(eq.max()), len(eq),
                )


# ── Train / test split ────────────────────────────────────────────────────────

def stratified_split(
    sequences: List[Dict],
    test_frac: float,
    emotion_names: List[str],
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """Split sequences by trial, stratified by emotion class, with reproducible random shuffle."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    logger.info("Stratified split: seed=%d, test_frac=%.2f", seed, test_frac)

    by_class: Dict[int, List[Dict]] = defaultdict(list)
    for seq in sequences:
        by_class[int(seq["y"][0])].append(seq)

    train, test = [], []
    for cls_idx in range(len(emotion_names)):
        seqs = list(by_class[cls_idx])
        if not seqs:
            logger.warning("Class '%s' has no labeled sequences.", emotion_names[cls_idx])
            continue
        rng.shuffle(seqs)
        # Ensure at least 1 sample stays in training; rare classes may contribute 0 to test.
        n_test = min(max(1, int(len(seqs) * test_frac)), len(seqs) - 1)
        if n_test == 0:
            logger.warning(
                "Class '%s' has only 1 sequence — keeping it in training, skipping from test.",
                emotion_names[cls_idx],
            )
            train.extend(seqs)
            logger.info("  %-12s: %d total → %d train, 0 test", emotion_names[cls_idx], len(seqs), len(seqs))
        else:
            train.extend(seqs[:-n_test])
            test.extend(seqs[-n_test:])
            logger.info(
                "  %-12s: %d total → %d train, %d test",
                emotion_names[cls_idx], len(seqs), len(seqs) - n_test, n_test,
            )

    logger.info(
        "Split: %d train trials, %d test trials (test_frac=%.2f, seed=%d)",
        len(train), len(test), test_frac, seed,
    )
    return train, test


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model: QualityAwareGaussianHMM,
    test_sequences: List[Dict],
    emotion_names: List[str],
) -> Dict:
    y_true, y_pred = [], []

    for seq in test_sequences:
        obs = {m: seq[f"obs_{m}"] for m in model.modalities}
        q = {m: seq[f"q_{m}"] for m in model.modalities}
        pred_path = model.predict(obs, q, return_state_names=False)
        # Trial-level prediction: majority vote over per-window Viterbi states.
        pred_trial = Counter(pred_path).most_common(1)[0][0]
        y_true.append(int(seq["y"][0]))
        y_pred.append(int(pred_trial))

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n_classes = len(emotion_names)

    acc = float(np.mean(y_true_arr == y_pred_arr))

    per_class = {}
    for c, name in enumerate(emotion_names):
        tp = int(np.sum((y_true_arr == c) & (y_pred_arr == c)))
        fp = int(np.sum((y_true_arr != c) & (y_pred_arr == c)))
        fn = int(np.sum((y_true_arr == c) & (y_pred_arr != c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[name] = {
            "precision": round(prec, 4),
            "recall":    round(rec, 4),
            "f1":        round(f1, 4),
            "support":   tp + fn,
        }

    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "n_test_trials": len(y_true),
    }


def log_results(results: Dict, emotion_names: List[str]) -> None:
    logger.info("── Evaluation Results ──────────────────────────────────────")
    logger.info(
        "Accuracy: %.4f  macro_F1: %.4f  (n=%d trials)",
        results["accuracy"], results.get("macro_f1", 0.0), results["n_test_trials"],
    )
    logger.info(
        "  %-12s %9s %9s %9s %9s",
        "Class", "Precision", "Recall", "F1", "Support",
    )
    for name in emotion_names:
        if name not in results["per_class"]:
            continue
        pc = results["per_class"][name]
        logger.info(
            "  %-12s %9.4f %9.4f %9.4f %9d",
            name, pc["precision"], pc["recall"], pc["f1"], pc["support"],
        )
    logger.info("Confusion matrix (rows=true, cols=pred):")
    header = "  " + "".join(f"  {n[:6]:>6}" for n in emotion_names)
    logger.info(header)
    for i, row in enumerate(results["confusion_matrix"]):
        logger.info("  %-6s" + "  %6d" * len(row), emotion_names[i][:6], *row)
    logger.info("────────────────────────────────────────────────────────────")


# ── Pipeline building blocks ──────────────────────────────────────────────────

def _load_entity_data(
    minio_client,
    gold_bucket: str,
    feat_prefix: str,
    dq_prefix: str,
    entity_id: str,
    modalities: List[str],
    window_size: float,
) -> Optional[Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]]:
    """Load parquets and attach quality + emotion. Returns (feat_dfs, feat_cols) or None."""
    try:
        dq_df = load_data_quality(minio_client, gold_bucket, entity_id, dq_prefix)
        feat_dfs: Dict[str, pd.DataFrame] = {}
        feat_cols_map: Dict[str, List[str]] = {}
        for m in modalities:
            df = load_features(minio_client, gold_bucket, entity_id, m, feat_prefix)
            cols = _feature_cols(df)
            flag_col = _MODALITY_FLAG_COL.get(m, f"{m}_flag")
            df = attach_quality_and_emotion(df, dq_df, flag_col, window_size_s=window_size)
            feat_dfs[m] = df
            feat_cols_map[m] = cols
            logger.info(
                "Modality %s: %d windows, %d feature cols, quality scores [%.2f, %.2f]",
                m, len(df), len(cols),
                df["quality_score"].min(), df["quality_score"].max(),
            )
        return feat_dfs, feat_cols_map
    except Exception as e:
        logger.error("[%s] Data load failed: %s", entity_id, e, exc_info=True)
        return None


def _build_and_remap(
    feat_dfs: Dict[str, pd.DataFrame],
    feat_cols_map: Dict[str, List[str]],
    entity_id: str,
    use_quality: bool,
    emotion_names_full: List[str],
) -> Optional[Tuple[List[Dict], List[str], int]]:
    """Build sequences and remap labels to a contiguous range. Returns (seqs, emotion_names, n_states)."""
    emotion_to_idx = {e: i for i, e in enumerate(emotion_names_full)}
    sequences = build_sequences(feat_dfs, feat_cols_map, emotion_to_idx, use_quality=use_quality)
    if not sequences:
        logger.error("[%s] No sequences built (use_quality=%s).", entity_id, use_quality)
        return None

    emotion_names = list(emotion_names_full)
    present_labels = sorted({int(seq["y"][0]) for seq in sequences})
    if set(present_labels) != set(range(len(emotion_names))):
        missing = [emotion_names[i] for i in range(len(emotion_names)) if i not in present_labels]
        logger.warning(
            "[%s] No labeled trials for %s — reducing to %d states.",
            entity_id, missing, len(present_labels),
        )
        remap = {old: new for new, old in enumerate(present_labels)}
        for seq in sequences:
            seq["y"] = np.vectorize(remap.__getitem__)(seq["y"])
        emotion_names = [emotion_names[i] for i in present_labels]

    return sequences, emotion_names, len(emotion_names)


def _fit_and_eval(
    sequences: List[Dict],
    modality_dims: Dict[str, int],
    emotion_names: List[str],
    modalities: List[str],
    n_states: int,
    test_frac: float,
    noise_strength: float,
    seed: int,
    entity_id: str,
) -> Optional[Dict]:
    """Split, impute (train-only), scale (train-only), fit HMM, evaluate. Returns results or None."""
    train_seqs, test_seqs = stratified_split(sequences, test_frac, emotion_names, seed=seed)
    if not train_seqs or not test_seqs:
        logger.error("[%s] Train or test split is empty (seed=%d).", entity_id, seed)
        return None

    # Impute missing feature values using training-set mean (fitted on train only).
    imputers = fit_imputers(train_seqs, modalities)
    train_seqs = apply_imputers(train_seqs, imputers)
    test_seqs = apply_imputers(test_seqs, imputers)

    scalers = fit_scalers(train_seqs, modalities)
    train_seqs = apply_scalers(train_seqs, scalers)
    test_seqs = apply_scalers(test_seqs, scalers)

    model = QualityAwareGaussianHMM(
        n_states=n_states,
        modality_dims=modality_dims,
        state_names=emotion_names[:n_states],
        noise_strength=noise_strength,
    )
    model.fit(train_seqs)

    results = evaluate(model, test_seqs, emotion_names)
    log_results(results, emotion_names)
    return results


# ── Single-entity entry point ─────────────────────────────────────────────────

def run_entity(
    minio_client,
    gold_bucket: str,
    feat_prefix: str,
    dq_prefix: str,
    entity_id: str,
    modalities: List[str],
    n_states: int,
    test_frac: float,
    noise_strength: float,
    window_size: float,
    use_quality: bool,
    seed: int = 0,
) -> Optional[Dict]:
    logger.info(
        "── Entity: %s | Modalities: %s | States: %d | Quality-aware: %s | seed=%d",
        entity_id, modalities, n_states, use_quality, seed,
    )

    data = _load_entity_data(
        minio_client, gold_bucket, feat_prefix, dq_prefix, entity_id, modalities, window_size
    )
    if data is None:
        return None
    feat_dfs, feat_cols_map = data

    log_quality_stats(feat_dfs, modalities)

    built = _build_and_remap(feat_dfs, feat_cols_map, entity_id, use_quality, _EAV_EMOTIONS)
    if built is None:
        return None
    sequences, emotion_names, n_states = built

    modality_dims = {m: len(feat_cols_map[m]) for m in modalities}
    return _fit_and_eval(
        sequences, modality_dims, emotion_names, modalities,
        n_states, test_frac, noise_strength, seed, entity_id,
    )


# ── Multi-seed comparison ─────────────────────────────────────────────────────

def _log_multi_seed_comparison(
    entity_id: str,
    results_q: List[Tuple[int, Dict]],
    results_no_q: List[Tuple[int, Dict]],
) -> None:
    logger.info("════ Multi-seed summary: %s ════", entity_id)

    def _report(results, label):
        if not results:
            logger.info("  %s: no successful runs", label)
            return None, None
        accs = [r["accuracy"] for _, r in results]
        f1s = [r.get("macro_f1", 0.0) for _, r in results]
        logger.info(
            "  %-14s acc: mean=%.4f std=%.4f | macro_F1: mean=%.4f std=%.4f  (%d seeds)",
            label + ":", float(np.mean(accs)), float(np.std(accs)),
            float(np.mean(f1s)), float(np.std(f1s)), len(results),
        )
        return accs, f1s

    _report(results_q, "with_quality")
    _report(results_no_q, "no_quality")

    seeds_q = {s for s, _ in results_q}
    seeds_no = {s for s, _ in results_no_q}
    common = sorted(seeds_q & seeds_no)
    if common:
        q_map = {s: r for s, r in results_q}
        no_map = {s: r for s, r in results_no_q}
        acc_deltas = [q_map[s]["accuracy"] - no_map[s]["accuracy"] for s in common]
        f1_deltas = [q_map[s].get("macro_f1", 0.0) - no_map[s].get("macro_f1", 0.0) for s in common]
        logger.info(
            "  delta acc (with_q - no_q): mean=%.4f std=%.4f  improved in %d/%d seeds",
            float(np.mean(acc_deltas)), float(np.std(acc_deltas)),
            sum(1 for d in acc_deltas if d > 0), len(common),
        )
        logger.info(
            "  delta F1  (with_q - no_q): mean=%.4f std=%.4f  improved in %d/%d seeds",
            float(np.mean(f1_deltas)), float(np.std(f1_deltas)),
            sum(1 for d in f1_deltas if d > 0), len(common),
        )
    logger.info("════════════════════════════════════════════════════════════")


def _run_multi_seed(
    minio_client,
    gold_bucket: str,
    feat_prefix: str,
    dq_prefix: str,
    entity_id: str,
    modalities: List[str],
    n_states: int,
    test_frac: float,
    noise_strength: float,
    window_size: float,
    seeds: List[int],
) -> None:
    """Load entity data once, then run split+train+eval for every seed and both quality conditions."""
    logger.info(
        "Multi-seed experiment: entity=%s, %d seeds, comparing with/without quality",
        entity_id, len(seeds),
    )

    # Expensive I/O happens once.
    data = _load_entity_data(
        minio_client, gold_bucket, feat_prefix, dq_prefix, entity_id, modalities, window_size
    )
    if data is None:
        return
    feat_dfs, feat_cols_map = data

    log_quality_stats(feat_dfs, modalities)

    # Build sequences for both quality conditions (cheap, no I/O).
    built_q = _build_and_remap(feat_dfs, feat_cols_map, entity_id, True, _EAV_EMOTIONS)
    built_no_q = _build_and_remap(feat_dfs, feat_cols_map, entity_id, False, _EAV_EMOTIONS)
    if built_q is None or built_no_q is None:
        return

    seqs_q, emotion_names_q, n_states_q = built_q
    seqs_no_q, emotion_names_no_q, n_states_no_q = built_no_q
    modality_dims = {m: len(feat_cols_map[m]) for m in modalities}

    results_q: List[Tuple[int, Dict]] = []
    results_no_q: List[Tuple[int, Dict]] = []

    for i, seed in enumerate(seeds):
        logger.info("── Seed %d/%d (seed=%d) ──", i + 1, len(seeds), seed)
        r_q = _fit_and_eval(
            seqs_q, modality_dims, emotion_names_q, modalities,
            n_states_q, test_frac, noise_strength, seed, entity_id,
        )
        r_no_q = _fit_and_eval(
            seqs_no_q, modality_dims, emotion_names_no_q, modalities,
            n_states_no_q, test_frac, noise_strength, seed, entity_id,
        )
        if r_q is not None:
            results_q.append((seed, r_q))
        if r_no_q is not None:
            results_no_q.append((seed, r_no_q))

    _log_multi_seed_comparison(entity_id, results_q, results_no_q)


# ── Aggregate reporting ───────────────────────────────────────────────────────

def _log_aggregate_summary(all_results: Dict[str, Optional[Dict]], emotion_names: List[str]) -> None:
    ok = {eid: r for eid, r in all_results.items() if r is not None}
    failed = [eid for eid, r in all_results.items() if r is None]

    logger.info("════════════════════════════════════════════════════════════")
    logger.info("Aggregate summary: %d / %d entities succeeded", len(ok), len(all_results))
    if failed:
        logger.info("Failed entities: %s", failed)

    if not ok:
        return

    accs = [r["accuracy"] for r in ok.values()]
    f1s = [r.get("macro_f1", 0.0) for r in ok.values()]
    logger.info(
        "Accuracy  — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
        float(np.mean(accs)), float(np.std(accs)), float(np.min(accs)), float(np.max(accs)),
    )
    logger.info(
        "Macro F1  — mean: %.4f  std: %.4f  min: %.4f  max: %.4f",
        float(np.mean(f1s)), float(np.std(f1s)), float(np.min(f1s)), float(np.max(f1s)),
    )

    logger.info("Per-entity results:")
    for eid, r in sorted(ok.items()):
        logger.info(
            "  %-6s  acc=%.4f  macro_f1=%.4f  (n=%d)",
            eid, r["accuracy"], r.get("macro_f1", 0.0), r["n_test_trials"],
        )

    logger.info("Mean per-class F1:")
    for name in emotion_names:
        f1s_class = [r["per_class"][name]["f1"] for r in ok.values() if name in r["per_class"]]
        if f1s_class:
            logger.info("  %-12s  %.4f", name, float(np.mean(f1s_class)))
    logger.info("════════════════════════════════════════════════════════════")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gold — Step 03: HMM Emotion Recognition")
    parser.add_argument("--config", default=_DEFAULT_CONFIG)

    entity_grp = parser.add_mutually_exclusive_group(required=True)
    entity_grp.add_argument("--entity", help="Single entity ID (e.g. e01)")
    entity_grp.add_argument(
        "--all-entities", action="store_true",
        help="Discover and run all entities found in the gold feature_extraction bucket",
    )

    parser.add_argument(
        "--modalities", nargs="+", default=["audio", "video", "eeg"],
        choices=["audio", "video", "eeg"],
    )
    parser.add_argument("--n-states", type=int, default=4)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--noise-strength", type=float, default=20.0,
                        help="Controls quality impact on inference variance. "
                             "0.0 = quality only affects training weights; "
                             ">0.0 = also inflates emission variance at inference. "
                             "Calibration grid: 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0")
    parser.add_argument(
        "--window-size", type=float, default=0.3,
        help="Window size in seconds used during feature extraction (needed to reconstruct "
             "window_start_s if it is absent from an older parquet)",
    )
    parser.add_argument(
        "--no-quality", action="store_true",
        help="Ignore data-quality flags; treat all windows as equal weight (q=1.0). "
             "Ignored in multi-seed mode (--n-seeds > 1), which always compares both conditions.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducible train/test split (single-seed mode).",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=1,
        help="Run with seeds 0..n-seeds-1. When > 1, enables multi-seed comparison mode: "
             "both quality and no-quality conditions are evaluated for each seed and aggregated.",
    )
    return parser.parse_args()


def _discover_entities(minio_client, gold_bucket: str, feat_prefix: str) -> List[str]:
    objects = minio_client.list_objects(gold_bucket, prefix=f"{feat_prefix}/eav/", recursive=False)
    return sorted(o.object_name.rstrip("/").split("/")[-1] for o in objects)


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    gold_bucket  = cfg["bucket_gold"]
    feat_prefix  = cfg.get("feature_extraction", {}).get("output_prefix", "feature_extraction")
    dq_prefix    = cfg.get("data_quality", {}).get("output_prefix", "data_quality")

    use_quality = not args.no_quality
    multi_seed = args.n_seeds > 1
    seeds = list(range(args.n_seeds))

    if args.all_entities:
        entities = _discover_entities(minio_client, gold_bucket, feat_prefix)
        logger.info("Discovered %d entities: %s", len(entities), entities)
    else:
        entities = [args.entity]

    common_kwargs = dict(
        minio_client=minio_client,
        gold_bucket=gold_bucket,
        feat_prefix=feat_prefix,
        dq_prefix=dq_prefix,
        modalities=args.modalities,
        n_states=args.n_states,
        test_frac=args.test_frac,
        noise_strength=args.noise_strength,
        window_size=args.window_size,
    )

    if multi_seed:
        if args.no_quality:
            logger.warning(
                "--no-quality is ignored in multi-seed mode; both conditions are compared."
            )
        logger.info("Multi-seed mode: seeds=0..%d, noise_strength=%.1f", args.n_seeds - 1, args.noise_strength)
        for entity_id in entities:
            _run_multi_seed(entity_id=entity_id, seeds=seeds, **common_kwargs)
    else:
        all_results: Dict[str, Optional[Dict]] = {}
        for entity_id in entities:
            all_results[entity_id] = run_entity(
                entity_id=entity_id,
                use_quality=use_quality,
                seed=args.seed,
                **common_kwargs,
            )
        if args.all_entities:
            _log_aggregate_summary(all_results, _EAV_EMOTIONS)
        elif all_results[entities[0]] is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
