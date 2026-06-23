import argparse
import io
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import yaml
from dotenv import load_dotenv

import config as project_config
from minio_utils import download_object, _group_objects_by_entity, upload_csv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("silver_annotation_quality")

EAV_COLUMNS = [
    "trial_id", "video_filename", "audio_filename", "eeg_mat_index",
    "emotion_class", "conversation_emotion_class", "conversation_idx", "conversation_occurrence",
    "valence_participant", "valence_experimenter", "arousal_participant", "arousal_experimenter",
    "valence_class_alignment", "arousal_class_alignment",
    "annotation_quality_flag",
    "avg_valence", "avg_arousal", "valence_arousal_emotion", "emotion_class_alignment",
]

KEMOCON_COLUMNS = [
    "seconds",
    "valence_rater_agreement", "arousal_rater_agreement",
    "cheerful_rater_agreement", "happy_rater_agreement", "angry_rater_agreement",
    "nervous_rater_agreement", "sad_rater_agreement",
    "bromp1_rater_agreement", "bromp2_rater_agreement",
    "annotation_quality_flag",
    "dominant_emotion", "dominant_emotion_value", "avg_valence", "avg_arousal", "valence_arousal_emotion", "emotion_class_alignment",
]

_DATASET_COLUMNS: Dict[str, List[str]] = {
    "EAV": EAV_COLUMNS,
    "K-EmoCon": KEMOCON_COLUMNS,
}

_ONE_HOT_COL_TO_CLASS: Dict[int, str] = {
    5: "Neutral",    6: "Neutral",
    7: "Sadness",    8: "Sadness",
    9: "Anger",     10: "Anger",
    11: "Happiness", 12: "Happiness",
    13: "Calm",      14: "Calm",
}



# ── meta_data.csv loader ───────────────────────────────────────────────────────

def load_meta_data(data: bytes) -> List[Dict[str, Any]]:
    """Parse global meta_data.csv → trial list with conversation mapping.

    Layout (skiprows=2, header=None):
      col 3  = Index (1-200)
      col 5  = NEU_LIS, col 6 = NEU_SPE
      col 7  = S_LIS,   col 8 = S_SPE
      col 9  = A_LIS,   col 10 = A_SPE
      col 11 = H_LIS,   col 12 = H_SPE
      col 13 = C_LIS,   col 14 = C_SPE
      col 17 = current (original) video filename
      col 18 = renamed video filename (base name, no extension)
      col 19 = audio filename (.wav; empty for Listening trials)
    """
    df = pd.read_csv(io.BytesIO(data), header=None, skiprows=2)

    # Pass 1: dominant (non-neutral) emotion class per conversation (block of 10)
    conv_emotion_map: Dict[int, str] = {}
    for _, row in df.iterrows():
        idx = pd.to_numeric(row.iloc[3], errors="coerce")
        if pd.isna(idx):
            continue
        conv_idx = (int(idx) - 1) // 10
        if conv_idx in conv_emotion_map:
            continue
        for col_pos, cls in _ONE_HOT_COL_TO_CLASS.items():
            val = pd.to_numeric(row.iloc[col_pos], errors="coerce")
            if pd.notna(val) and int(val) == 1 and cls != "Neutral":
                conv_emotion_map[conv_idx] = cls
                break

    # Per-emotion-class occurrence counter → questionnaire "conversation" key (1-indexed)
    class_count: Dict[str, int] = {}
    conv_occurrence: Dict[int, int] = {}
    for conv_idx in sorted(conv_emotion_map):
        cls = conv_emotion_map[conv_idx]
        class_count[cls] = class_count.get(cls, 0) + 1
        conv_occurrence[conv_idx] = class_count[cls]

    # Pass 2: build per-trial records
    trials: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        idx = pd.to_numeric(row.iloc[3], errors="coerce")
        if pd.isna(idx):
            continue
        trial_id = int(idx)
        conv_idx = (trial_id - 1) // 10

        emotion_class: Optional[str] = None
        for col_pos, cls in _ONE_HOT_COL_TO_CLASS.items():
            val = pd.to_numeric(row.iloc[col_pos], errors="coerce")
            if pd.notna(val) and int(val) == 1:
                emotion_class = cls
                break

        if emotion_class is None:
            logger.warning("meta_data trial %d: no active one-hot column", trial_id)
            continue

        video_filename = ""
        try:
            v = row.iloc[18]
            video_filename = str(v).strip() if pd.notna(v) else ""
        except IndexError:
            pass

        audio_filename = ""
        try:
            a = row.iloc[19]
            audio_filename = str(a).strip() if pd.notna(a) else ""
        except IndexError:
            pass

        trials.append({
            "trial_id": trial_id,
            "eeg_mat_index": trial_id - 1,  # 0-based index in .mat instances axis
            "emotion_class": emotion_class,
            "video_filename": video_filename,
            "audio_filename": audio_filename,
            "conversation_idx": conv_idx,
            "conversation_emotion_class": conv_emotion_map.get(conv_idx, "Unknown"),
            "conversation_occurrence": conv_occurrence.get(conv_idx, 0),
        })

    return sorted(trials, key=lambda t: t["trial_id"])


# ── questionnaire.xlsx loader ──────────────────────────────────────────────────

def _parse_questionnaire_sheet(df: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Parse one subject sheet.

    Layout per emotion-class block:
      row N:   section header in col A (0): "Subject's template. Emotion class: X"
      row N+1: column headers row (skipped)
      rows N+2 … N+6: data for conversations k=1..5
        subject side:      col B(1)=valence, col C(2)=arousal
        experimenter side: col F(5)=valence, col G(6)=arousal

    Blocks (row offsets are 0-based after pd.read_excel with header=None):
      Happy   — header at row 3  (A4),  data rows 5-9   (B5:C9,  F5:G9)
      Calm    — header at row 12 (A13), data rows 14-18 (B14:C18, F14:G18)
      Angry   — header at row 21 (A22), data rows 23-27 (B23:C27, F23:G27)
      Sad     — header at row 30 (A31), data rows 32-36 (B32:C36, F32:G36)
    """
    result: Dict[Tuple[str, int], Dict[str, float]] = {}
    for row_idx in range(len(df)):
        cell = df.iloc[row_idx, 0]
        if not isinstance(cell, str):
            continue
        m = re.search(r"emotion\s+class\s*:?\s*(\w+)", cell, re.IGNORECASE)
        if not m:
            continue
        raw_class = m.group(1).strip().capitalize()
        # Normalize questionnaire label → canonical emotion class name
        emotion_class = _QUESTIONNAIRE_CLASS_MAP.get(raw_class, raw_class)
        for k in range(1, 6):
            data_idx = row_idx + 1 + k  # +1 skips the column-header row
            if data_idx >= len(df):
                break
            r = df.iloc[data_idx]
            try:
                result[(emotion_class, k)] = {
                    "valence_participant":  float(r.iloc[1]),
                    "arousal_participant":  float(r.iloc[2]),
                    "valence_experimenter": float(r.iloc[5]),
                    "arousal_experimenter": float(r.iloc[6]),
                }
            except (ValueError, TypeError, IndexError) as e:
                logger.warning("Questionnaire parse error [%s k=%d row=%d]: %s",
                               emotion_class, k, data_idx, e)
    if not result:
        col0_strings = [
            str(df.iloc[i, 0]) for i in range(min(40, len(df)))
            if isinstance(df.iloc[i, 0], str)
        ]
        logger.warning("No questionnaire entries parsed. Col-0 string cells: %s", col0_strings)
    return result


# Map questionnaire emotion labels → canonical names used in meta_data / config
_QUESTIONNAIRE_CLASS_MAP: Dict[str, str] = {
    "Happy":   "Happiness",
    "Calm":    "Calm",
    "Angry":   "Anger",
    "Sad":     "Sadness",
}


def load_questionnaire(data: bytes) -> Dict[str, Dict[Tuple[str, int], Dict[str, float]]]:
    """Load all sheets → {sheet_name: {(emotion_class, k): scores}}."""
    xl = pd.ExcelFile(io.BytesIO(data))
    result: Dict[str, Dict[Tuple[str, int], Dict[str, float]]] = {}
    for sheet_name in xl.sheet_names:
        if not re.match(r"subject_\d+", sheet_name, re.IGNORECASE):
            continue
        df = xl.parse(sheet_name, header=None)
        entries = _parse_questionnaire_sheet(df)
        result[sheet_name] = entries
        logger.info("Questionnaire sheet '%s': %d entries", sheet_name, len(entries))
    return result


# ── Alignment & flag logic ─────────────────────────────────────────────────────

def _is_aligned(score: float, emotion_class: str, dimension: str, thresholds_cfg: Dict) -> bool:
    if score == 0.0:
        return False
    cls_thr = thresholds_cfg.get(emotion_class, {})
    min_v = cls_thr.get(f"{dimension}_min")
    max_v = cls_thr.get(f"{dimension}_max")
    if min_v is not None and score < min_v:
        return False
    if max_v is not None and score > max_v:
        return False
    return True


def _compute_flag(
    val_part: float,
    aro_part: float,
    val_exp: float,
    aro_exp: float,
    emotion_class: str,
    thresholds_cfg: Dict,
    disagreement_threshold: int,
) -> Tuple[str, str, str, str, str]:
    """Returns (valence_rater_agreement, arousal_rater_agreement,
               valence_class_alignment, arousal_class_alignment, flag)."""
    val_agree = "DISAGREE" if abs(val_part - val_exp) >= disagreement_threshold else "AGREE"
    aro_agree = "DISAGREE" if abs(aro_part - aro_exp) >= disagreement_threshold else "AGREE"
    val_align = "ALIGNED" if _is_aligned(val_part, emotion_class, "valence", thresholds_cfg) else "MISALIGNED"
    aro_align = "ALIGNED" if _is_aligned(aro_part, emotion_class, "arousal", thresholds_cfg) else "MISALIGNED"

    if val_align == "MISALIGNED" and aro_align == "MISALIGNED":
        flag = "BAD"
    elif val_align == "MISALIGNED" or aro_align == "MISALIGNED":
        flag = "NOISY"
    elif val_agree == "DISAGREE" and aro_agree == "DISAGREE":
        flag = "NOISY"
    elif val_agree == "DISAGREE" or aro_agree == "DISAGREE":
        flag = "ACCEPTABLE"
    else:
        flag = "GOOD"

    return val_agree, aro_agree, val_align, aro_align, flag


# ── Emotion labelling helpers ──────────────────────────────────────────────────

def _valence_arousal_to_emotion(valence: float, arousal: float) -> Optional[str]:
    """EAV mapping (scale centred at 0)."""
    if pd.isna(valence) or pd.isna(arousal):
        return None
    if valence >= 0 and arousal >= 0:
        return "Happiness"
    if valence <= 0 and arousal >= 0:
        return "Anger"
    if valence >= 0 and arousal <= 0:
        return "Calm"
    if valence <= 0 and arousal <= 0:
        return "Sadness"
    return None


def _valence_arousal_to_emotion_kemocon(valence: float, arousal: float) -> Optional[str]:
    """K-EmoCon mapping (Likert scale 1-5, centred at 3)."""
    if pd.isna(valence) or pd.isna(arousal):
        return None
    if valence >= 3 and arousal >= 3:
        return "Happiness"
    if valence <= 3 and arousal >= 3:
        return "Anger"
    if valence >= 3 and arousal <= 3:
        return "Calm"
    if valence <= 3 and arousal <= 3:
        return "Sadness"
    return None


_KEMOCON_EMOTION_DIMS = ["cheerful", "happy", "angry", "nervous", "sad"]


def _dominant_emotion_kemocon(window_rows: List[pd.Series]) -> Tuple[str, float]:
    """Returns (mapped_labels, max_sum).

    All matching rules are applied independently per dominant dim; the union of
    matched classes (deduplicated, fixed order) is returned as a comma-separated string.

    Rules:
      sum <= 4                        → Neutral   (global, any dim)
      dim in (happy, sad) & sum <= 6  → Calm
      dim in (cheerful, happy)        → Happiness
      dim == angry                    → Anger
      dim in (nervous, sad)           → Sadness
    """
    sums: Dict[str, float] = {}
    for dim in _KEMOCON_EMOTION_DIMS:
        vals = [pd.to_numeric(row.get(dim, float("nan")), errors="coerce") for row in window_rows]
        sums[dim] = sum(v for v in vals if pd.notna(v))

    max_sum = max(sums.values())
    dominant_dims = [dim for dim, s in sums.items() if s == max_sum]

    _ORDER = ["Neutral", "Calm", "Happiness", "Anger", "Sadness"]
    matched: set = set()

    if max_sum <= 4:
        matched.add("Neutral")

    for dim in dominant_dims:
        if dim in ("happy", "sad") and max_sum <= 6:
            matched.add("Calm")
        if dim in ("cheerful", "happy"):
            matched.add("Happiness")
        if dim == "angry":
            matched.add("Anger")
        if dim in ("nervous", "sad"):
            matched.add("Sadness")

    labels = [c for c in _ORDER if c in matched]
    return ", ".join(labels), max_sum


# ── EAV audit ─────────────────────────────────────────────────────────────────

def audit_eav_annotation_quality(
    minio_client,
    bucket: str,
    entity_objects: Dict[str, List[Any]],
    ds_cfg: Dict[str, Any],
) -> List[Dict]:
    dataset_label = ds_cfg["dataset_label"]
    disagreement_threshold = int(ds_cfg.get("rater_disagreement_threshold", 3))
    thresholds_cfg = ds_cfg.get("class_alignment_thresholds", {})

    meta_bytes = download_object(minio_client, bucket, ds_cfg["meta_data_path"])
    if meta_bytes is None:
        logger.error("Cannot load meta_data — aborting EAV audit")
        return []
    trials = load_meta_data(meta_bytes)
    logger.info("[EAV] meta_data: %d trials loaded", len(trials))

    quest_bytes = download_object(minio_client, bucket, ds_cfg["questionnaire_path"])
    if quest_bytes is None:
        logger.error("Cannot load questionnaire — aborting EAV audit")
        return []
    questionnaire = load_questionnaire(quest_bytes)
    logger.info("[EAV] questionnaire: %d subject sheets", len(questionnaire))

    all_pids = sorted(entity_objects.keys())
    logger.info("[EAV] Participants to audit: %d", len(all_pids))

    rows: List[Dict] = []

    for entity_id in all_pids:
        m = re.match(r"e(\d+)$", entity_id, re.IGNORECASE)
        if not m:
            logger.warning("[EAV] Unrecognized entity_id '%s' — skipping", entity_id)
            continue
        sheet_name = f"subject_{int(m.group(1))}"
        subject_quest = questionnaire.get(sheet_name)
        if subject_quest is None:
            logger.warning("[EAV] [%s] Questionnaire sheet '%s' not found", entity_id, sheet_name)
            continue

        for trial in trials:
            conv_emotion = trial["conversation_emotion_class"]
            conv_k = trial["conversation_occurrence"]
            quest_entry = subject_quest.get((conv_emotion, conv_k))

            row: Dict[str, Any] = {
                "dataset": dataset_label,
                "participant_id": entity_id,
                "trial_id": trial["trial_id"],
                "video_filename": trial["video_filename"],
                "audio_filename": trial["audio_filename"],
                "eeg_mat_index": trial["eeg_mat_index"],
                "emotion_class": trial["emotion_class"],
                "conversation_emotion_class": conv_emotion,
                "conversation_idx": trial["conversation_idx"],
                "conversation_occurrence": conv_k,
                "valence_participant": None,
                "valence_experimenter": None,
                "arousal_participant": None,
                "arousal_experimenter": None,
                "valence_class_alignment": None,
                "arousal_class_alignment": None,
                "annotation_quality_flag": None,
                "avg_valence": None,
                "avg_arousal": None,
                "valence_arousal_emotion": None,
                "emotion_class_alignment": None,
            }

            if quest_entry is None:
                logger.warning("[EAV] [%s] trial %d: no questionnaire entry (%s, k=%d)",
                               entity_id, trial["trial_id"], conv_emotion, conv_k)
            else:
                val_part = quest_entry["valence_participant"]
                aro_part = quest_entry["arousal_participant"]
                val_exp  = quest_entry["valence_experimenter"]
                aro_exp  = quest_entry["arousal_experimenter"]
                _, _, val_align, aro_align, flag = _compute_flag(
                    val_part, aro_part, val_exp, aro_exp,
                    conv_emotion, thresholds_cfg, disagreement_threshold,
                )
                avg_val = (val_part + val_exp) / 2
                avg_aro = (aro_part + aro_exp) / 2
                va_emotion = _valence_arousal_to_emotion(avg_val, avg_aro)
                row.update({
                    "valence_participant":     val_part,
                    "valence_experimenter":    val_exp,
                    "arousal_participant":     aro_part,
                    "arousal_experimenter":    aro_exp,
                    "valence_class_alignment": val_align,
                    "arousal_class_alignment": aro_align,
                    "annotation_quality_flag": flag,
                    "avg_valence":             avg_val,
                    "avg_arousal":             avg_aro,
                    "valence_arousal_emotion": va_emotion,
                    "emotion_class_alignment": "ALIGNED" if trial["conversation_emotion_class"] == va_emotion else "MISALIGNED",
                })

            rows.append(row)

        logger.info("[EAV] [%s] %d trial rows generated", entity_id, len(trials))

    return rows


_KEMOCON_LIKERT_DIMS = ["arousal", "valence", "cheerful", "happy", "angry", "nervous", "sad"]
_KEMOCON_BROMP1_COLS = ["boredom", "confusion", "delight", "concentration", "frustration", "surprise", "none_1"]
_KEMOCON_BROMP2_COLS = ["confrustion", "contempt", "dejection", "disgust", "eureka", "pride", "sorrow", "none_2"]

_KEMOCON_PERSPECTIVE_MAP: Dict[str, Tuple[str, str]] = {
    "self": ("self_annotations", ".self"),
    "partner": ("partner_annotations", ".partner"),
    "aggregated_external": ("aggregated_external_annotations", ".external"),
}


def _group_kemocon_annotation_files(
    minio_client, bucket: str, prefix: str, perspectives: List[str],
) -> Dict[str, Dict[str, str]]:
    """Group annotation files as {participant_id: {perspective: object_name}}.

    Layout: {prefix}/{perspective}_annotations/P{N}.{perspective_abbrev}.csv
    """
    full_prefix = prefix.rstrip("/") + "/"
    result: Dict[str, Dict[str, str]] = {}

    for perspective in perspectives:
        if perspective not in _KEMOCON_PERSPECTIVE_MAP:
            logger.warning("[K-EmoCon] No directory mapping for perspective '%s' — skipping", perspective)
            continue
        subdir, stem_suffix = _KEMOCON_PERSPECTIVE_MAP[perspective]
        subdir_prefix = full_prefix + subdir + "/"
        for obj in minio_client.list_objects(bucket, prefix=subdir_prefix, recursive=False):
            filename = obj.object_name[len(subdir_prefix):]
            if not filename or "/" in filename:
                continue
            stem = Path(filename).stem  # e.g. "P1.self"
            if not stem.endswith(stem_suffix):
                continue
            raw_id = stem[: -len(stem_suffix)]  # e.g. "P1"
            m = re.match(r"P(\d+)$", raw_id, re.IGNORECASE)
            participant_id = f"e{int(m.group(1)):02d}" if m else raw_id
            result.setdefault(participant_id, {})[perspective] = obj.object_name

    return result


def _kemocon_likert_agree(values: List[float]) -> str:
    """AGREE if any pair of non-null rater values has |diff| <= 1."""
    valid = [v for v in values if pd.notna(v)]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if abs(valid[i] - valid[j]) <= 1:
                return "AGREE"
    return "DISAGREE"


def _kemocon_bromp_agree(window_rows: List[pd.Series], cols: List[str]) -> str:
    """AGREE if any single BROMP category is selected (value == 'x') by >= 2 raters."""
    for col in cols:
        count = sum(
            1 for row in window_rows
            if col in row.index and isinstance(row[col], str) and row[col].strip().lower() == "x"
        )
        if count >= 2:
            return "AGREE"
    return "DISAGREE"


def _kemocon_quality_flag(flags: Dict[str, str]) -> str:
    """Hierarchy: BAD >= 5 DISAGREE; NOISY >= 3 or (arousal AND valence); ACCEPTABLE arousal XOR valence; GOOD."""
    n_dis = sum(1 for v in flags.values() if v == "DISAGREE")
    if n_dis >= 5:
        return "BAD"
    aro_dis = flags.get("arousal") == "DISAGREE"
    val_dis = flags.get("valence") == "DISAGREE"
    if n_dis >= 3 or (aro_dis and val_dis):
        return "NOISY"
    if aro_dis != val_dis:
        return "ACCEPTABLE"
    return "GOOD"


# ── K-EmoCon audit ────────────────────────────────────────────────────────────

def audit_kemocon_annotation_quality(
    minio_client,
    bucket: str,
    entity_objects: Dict[str, List[Any]],
    ds_cfg: Dict[str, Any],
) -> List[Dict]:
    dataset_label = ds_cfg["dataset_label"]
    annotations_prefix = ds_cfg["annotations_prefix"]
    perspectives: List[str] = ds_cfg.get("perspectives", ["self", "partner", "aggregated_external"])

    grouped = _group_kemocon_annotation_files(minio_client, bucket, annotations_prefix, perspectives)
    logger.info("[K-EmoCon] Annotation file groups: %d participants", len(grouped))

    rows: List[Dict] = []

    for participant_id in sorted(grouped):
        perspective_paths = grouped[participant_id]
        missing = [p for p in perspectives if p not in perspective_paths]
        if missing:
            logger.warning("[K-EmoCon] [%s] Missing perspectives %s — skipping", participant_id, missing)
            continue

        dfs: Dict[str, pd.DataFrame] = {}
        failed = False
        for perspective in perspectives:
            data = download_object(minio_client, bucket, perspective_paths[perspective])
            if data is None:
                logger.error("[K-EmoCon] [%s] Failed to download %s perspective", participant_id, perspective)
                failed = True
                break
            dfs[perspective] = pd.read_csv(io.BytesIO(data))
        if failed:
            continue

        # Trim from the beginning so all perspectives share identical row count
        min_len = min(len(df) for df in dfs.values())
        dfs = {p: df.iloc[len(df) - min_len:].reset_index(drop=True) for p, df in dfs.items()}
        logger.info("[K-EmoCon] [%s] %d windows after trim", participant_id, min_len)

        for win_idx in range(min_len):
            window_rows = [dfs[p].iloc[win_idx] for p in perspectives]

            flags: Dict[str, str] = {}
            for dim in _KEMOCON_LIKERT_DIMS:
                vals = [pd.to_numeric(row.get(dim, float("nan")), errors="coerce") for row in window_rows]
                flags[dim] = _kemocon_likert_agree(vals)
            flags["bromp1"] = _kemocon_bromp_agree(window_rows, _KEMOCON_BROMP1_COLS)
            flags["bromp2"] = _kemocon_bromp_agree(window_rows, _KEMOCON_BROMP2_COLS)

            val_vals = [pd.to_numeric(row.get("valence", float("nan")), errors="coerce") for row in window_rows]
            aro_vals = [pd.to_numeric(row.get("arousal", float("nan")), errors="coerce") for row in window_rows]
            avg_val = float(pd.Series(val_vals).mean())
            avg_aro = float(pd.Series(aro_vals).mean())
            va_emotion = _valence_arousal_to_emotion_kemocon(avg_val, avg_aro)
            dom_labels, dom_value = _dominant_emotion_kemocon(window_rows)
            aligned = "ALIGNED" if va_emotion and va_emotion in dom_labels.split(", ") else "MISALIGNED"

            rows.append({
                "dataset": dataset_label,
                "participant_id": participant_id,
                "seconds": win_idx * 5,
                "valence_rater_agreement": flags["valence"],
                "arousal_rater_agreement": flags["arousal"],
                "cheerful_rater_agreement": flags["cheerful"],
                "happy_rater_agreement": flags["happy"],
                "angry_rater_agreement": flags["angry"],
                "nervous_rater_agreement": flags["nervous"],
                "sad_rater_agreement": flags["sad"],
                "bromp1_rater_agreement": flags["bromp1"],
                "bromp2_rater_agreement": flags["bromp2"],
                "annotation_quality_flag": _kemocon_quality_flag(flags),
                "dominant_emotion": dom_labels,
                "dominant_emotion_value": dom_value,
                "avg_valence": avg_val,
                "avg_arousal": avg_aro,
                "valence_arousal_emotion": va_emotion,
                "emotion_class_alignment": aligned,
            })

        logger.info("[K-EmoCon] [%s] %d window rows generated", participant_id, min_len)

    return rows


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_annotation_quality(
    minio_client,
    silver_bucket: str,
    cfg: Dict[str, Any],
) -> List[Dict]:
    aq_cfg = cfg.get("annotation_quality", {})
    datasets_cfg = aq_cfg.get("datasets", {})
    all_rows: List[Dict] = []

    eav_cfg = datasets_cfg.get("eav")
    if eav_cfg:
        logger.info("=== Auditing EAV annotation quality ===")
        entity_objects = _group_objects_by_entity(
            minio_client, silver_bucket, eav_cfg["silver_files_prefix"]
        )
        logger.info("EAV entities in MinIO: %d", len(entity_objects))
        erows = audit_eav_annotation_quality(
            minio_client, silver_bucket, entity_objects, eav_cfg,
        )
        logger.info("EAV: %d report rows", len(erows))
        all_rows.extend(erows)

    kemocon_cfg = datasets_cfg.get("kemocon")
    if kemocon_cfg:
        logger.info("=== Auditing K-EmoCon annotation quality ===")
        krows = audit_kemocon_annotation_quality(
            minio_client, silver_bucket, {}, kemocon_cfg,
        )
        logger.info("K-EmoCon: %d report rows", len(krows))
        all_rows.extend(krows)

    return all_rows


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Silver — Step 05: Annotation Quality.")
    parser.add_argument("--config", default="pipeline_config.yaml", help="Path to YAML config.")
    return parser.parse_args()


_DATASET_DIR: Dict[str, str] = {
    "EAV": "eav",
    "K-EmoCon": "kemocon",
}


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    minio_client = project_config.config_minio()
    silver_bucket = cfg["bucket_silver"]
    aq_cfg = cfg.get("annotation_quality", {})
    output_prefix = aq_cfg.get("output_prefix", "05_annotation_quality").rstrip("/")

    logger.info("Starting Silver — Step 05: Annotation Quality")

    rows = run_annotation_quality(minio_client, silver_bucket, cfg)

    if not rows:
        logger.error("No rows produced — check logs for errors.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    logger.info("Total report rows: %d", len(df))

    flag_counts = df["annotation_quality_flag"].value_counts(dropna=False)
    logger.info("Flag distribution:\n%s", flag_counts.to_string())

    uploaded = 0
    for (dataset, participant_id), part_df in df.groupby(["dataset", "participant_id"], sort=True):
        columns = _DATASET_COLUMNS.get(dataset, list(part_df.columns))
        ds_dir = _DATASET_DIR.get(dataset, dataset.lower().replace("-", "").replace(" ", "_"))
        report_key = f"{output_prefix}/{ds_dir}/{participant_id}_annotation_quality.csv"
        upload_csv(minio_client, silver_bucket, report_key, part_df[columns])
        logger.info("[%s] [%s] %d rows → %s", dataset, participant_id, len(part_df), report_key)
        uploaded += 1

    logger.info("Uploaded %d participant report(s). Annotation Quality complete.", uploaded)


if __name__ == "__main__":
    main()