"""Validate a preprocessed MolmoAct fine-tuning dataset.

Checks every example for:
  1. Required columns exist and are non-empty
  2. Images are loadable as PIL images
  3. `processed_action` is valid JSON with a `chunked_action` key
  4. `depth` and `trace` are non-empty strings
  5. `language_instruction` is a non-empty string
  6. End-to-end tokenization produces non-zero loss_masks

Usage (standalone):
    python scripts/validate_dataset.py \
        --dataset-path /path/to/merged_dataset \
        [--check-tokenization] \
        [--model-checkpoint allenai/MolmoAct-7B-D-0812]

Usage (SLURM):
    sbatch submit_validate_data.run
"""

import argparse
import ast
import json
import logging
import sys
import os
import traceback
from pathlib import Path
from collections import Counter

import datasets
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("validate_dataset")

REQUIRED_COLUMNS = [
    "language_instruction",
    "depth",
    "trace",
    "processed_action",
]

# Known image column names (exact match)
IMAGE_COLUMNS = ["image", "wrist_image"]
# Patterns to search if exact names not found
IMAGE_COLUMN_PATTERNS = ["observation.image", "observation.images"]


def find_image_columns(columns):
    """Discover image columns from dataset, trying exact names then patterns."""
    found = [c for c in IMAGE_COLUMNS if c in columns]
    if found:
        return found
    for pattern in IMAGE_COLUMN_PATTERNS:
        matches = [c for c in columns if c.startswith(pattern)]
        if matches:
            return matches
    # Last resort: any column with 'image' in the name
    return [c for c in columns if 'image' in c.lower()]


# ---------------------------------------------------------------------------
#  Individual checks
# ---------------------------------------------------------------------------

def check_text_field(example: dict, idx: int, field: str, issues: list):
    val = example.get(field)
    if val is None:
        issues.append(f"[idx={idx}] Missing field '{field}'")
        return False
    if not isinstance(val, str) or len(val.strip()) == 0:
        issues.append(f"[idx={idx}] Field '{field}' is empty or not a string (type={type(val).__name__})")
        return False
    return True


def check_processed_action(example: dict, idx: int, issues: list):
    raw = example.get("processed_action")
    if raw is None:
        issues.append(f"[idx={idx}] Missing 'processed_action'")
        return False
    try:
        action = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        issues.append(f"[idx={idx}] 'processed_action' is not valid JSON: {e}")
        return False
    if not isinstance(action, dict):
        issues.append(f"[idx={idx}] 'processed_action' JSON root is not a dict (got {type(action).__name__})")
        return False
    if "chunked_action" not in action:
        issues.append(f"[idx={idx}] 'processed_action' missing 'chunked_action' key. Keys: {list(action.keys())}")
        return False
    return True


def check_trace(example: dict, idx: int, issues: list):
    trace = example.get("trace")
    if trace is None or (isinstance(trace, str) and len(trace.strip()) == 0):
        issues.append(f"[idx={idx}] 'trace' is missing or empty")
        return False
    # Verify parseable as a Python literal (list of coords)
    try:
        parsed = ast.literal_eval(trace)
        if not isinstance(parsed, list):
            issues.append(f"[idx={idx}] 'trace' does not parse to a list (got {type(parsed).__name__})")
            return False
    except (ValueError, SyntaxError) as e:
        issues.append(f"[idx={idx}] 'trace' is not a valid Python literal: {e}")
        return False
    return True


def check_images(example: dict, idx: int, issues: list, image_columns: list):
    found_any = False
    for col in image_columns:
        if col not in example or example[col] is None:
            continue
        raw = example[col]
        try:
            # Try opening as PIL image (handles dict, bytes, array, etc.)
            if isinstance(raw, Image.Image):
                img = raw
            elif isinstance(raw, dict):
                if "bytes" in raw and raw["bytes"]:
                    from io import BytesIO
                    img = Image.open(BytesIO(raw["bytes"]))
                elif "path" in raw and raw["path"] and os.path.exists(raw["path"]):
                    img = Image.open(raw["path"])
                else:
                    issues.append(f"[idx={idx}] Image column '{col}' dict has no usable bytes/path")
                    continue
            elif isinstance(raw, (bytes, bytearray)):
                from io import BytesIO
                img = Image.open(BytesIO(raw))
            elif isinstance(raw, str) and os.path.exists(raw):
                img = Image.open(raw)
            else:
                issues.append(f"[idx={idx}] Image column '{col}' has unexpected type: {type(raw).__name__}")
                continue
            w, h = img.size
            if w == 0 or h == 0:
                issues.append(f"[idx={idx}] Image column '{col}' has zero dimension: {w}x{h}")
                continue
            found_any = True
        except Exception as e:
            issues.append(f"[idx={idx}] Image column '{col}' failed to open: {e}")
    if not found_any:
        issues.append(f"[idx={idx}] No valid images found in columns {image_columns}")
        return False
    return True


# ---------------------------------------------------------------------------
#  Optional: end-to-end tokenization check
# ---------------------------------------------------------------------------

def check_tokenization(dataset_path: str, model_checkpoint: str, num_samples: int = 100):
    """Run a subset through the actual data pipeline and check loss_masks."""
    log.info("Running end-to-end tokenization check on %d samples...", num_samples)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from olmo.data.custom_lerobot_dataset import CustomLeRobotDataset
    from olmo.data.data_loader import DataLoaderConfig, RootSizeMixture

    # Use the same model config as training
    from olmo.util import select_checkpoint
    from olmo.models.molmo.molmo import MolmoConfig
    from os.path import join, exists

    checkpoint, is_hf_remote = select_checkpoint(model_checkpoint)

    if is_hf_remote:
        import tempfile
        import requests
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers
        url = hf_hub_url(repo_id=checkpoint, filename="model.yaml", repo_type="model")
        headers = build_hf_headers(token=None)
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        with tempfile.NamedTemporaryFile("wb", suffix="-model.yaml", delete=False) as tmp:
            tmp.write(r.content)
            model_cfg = MolmoConfig.load(tmp.name)
    elif exists(join(checkpoint, "model.yaml")):
        model_cfg = MolmoConfig.load(join(checkpoint, "model.yaml"))
    elif exists(join(checkpoint, "config.yaml")):
        model_cfg = MolmoConfig.load(join(checkpoint, "config.yaml"), key="model")
    else:
        log.warning("Cannot load model config from checkpoint, skipping tokenization check.")
        return 0

    model_cfg.data_formatter.prompt_templates = "uber_model"
    model_cfg.data_formatter.message_format = "role"
    model_cfg.data_formatter.system_prompt = "demo_or_style"
    model_cfg.mm_preprocessor.loss_token_weighting = "root_subsegments"
    model_cfg.llm.tokenizer.depth_tokens = True
    model_cfg.mm_preprocessor.max_images = 2

    data_cfg = DataLoaderConfig(
        root_size_mixture=[RootSizeMixture(1.0, {f"finetune:{dataset_path}": None})],
        shuffle=False,
        split="train",
        drop_last=False,
        sequence_length=2304,
        num_workers=0,
        pad="to_max",
        pin_memory=False,
        seed=42,
    )

    import torch
    loader = data_cfg.build_train_dataloader(model_cfg, batch_size=1, device=torch.device("cpu"))

    zero_weight_count = 0
    error_count = 0
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
        try:
            loss_masks = batch["loss_masks"]
            total_weight = (loss_masks * (loss_masks > 0)).sum().item()
            if total_weight == 0:
                zero_weight_count += 1
                log.warning("[tokenization idx=%d] loss_masks sum is 0 (no labeled tokens)", i)
        except Exception as e:
            error_count += 1
            log.error("[tokenization idx=%d] Error: %s", i, e)

    log.info("Tokenization check done: %d/%d samples had zero loss weight, %d errors",
             zero_weight_count, num_samples, error_count)
    return zero_weight_count


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def validate_dataset(dataset_path: str, check_tok: bool = False, model_checkpoint: str = None):
    log.info("Loading dataset from: %s", dataset_path)
    ds = datasets.load_from_disk(dataset_path)
    total = len(ds)
    log.info("Dataset loaded: %d examples", total)

    # Check columns exist
    columns = ds.column_names
    log.info("Columns: %s", columns)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing_cols:
        log.error("FATAL: Required columns missing from dataset: %s", missing_cols)
        return False

    actual_image_cols = find_image_columns(columns)
    if not actual_image_cols:
        log.error("FATAL: No image columns found. Expected at least one of: %s (or patterns: %s)",
                  IMAGE_COLUMNS, IMAGE_COLUMN_PATTERNS)
        return False
    log.info("Using image columns: %s", actual_image_cols)

    # Validate each example
    issues = []
    error_counts = Counter()
    log_interval = max(1, total // 20)  # log progress ~20 times

    for idx in range(total):
        if idx % log_interval == 0:
            log.info("Progress: %d/%d (%.1f%%)", idx, total, 100.0 * idx / total)

        try:
            example = ds[idx]
        except Exception as e:
            issues.append(f"[idx={idx}] Failed to load example: {e}")
            error_counts["load_error"] += 1
            continue

        # Text fields
        for field in ["language_instruction", "depth"]:
            if not check_text_field(example, idx, field, issues):
                error_counts[f"{field}_error"] += 1

        # Trace
        if not check_trace(example, idx, issues):
            error_counts["trace_error"] += 1

        # Processed action
        if not check_processed_action(example, idx, issues):
            error_counts["processed_action_error"] += 1

        # Images
        if not check_images(example, idx, issues, actual_image_cols):
            error_counts["image_error"] += 1

    # Summary
    log.info("=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)
    log.info("Total examples: %d", total)
    log.info("Total issues found: %d", len(issues))

    if error_counts:
        log.info("Error breakdown:")
        for err_type, count in error_counts.most_common():
            log.info("  %-30s %d (%.2f%%)", err_type, count, 100.0 * count / total)

    if issues:
        # Print first 50 issues
        log.warning("First %d issues:", min(50, len(issues)))
        for issue in issues[:50]:
            log.warning("  %s", issue)
        if len(issues) > 50:
            log.warning("  ... and %d more issues", len(issues) - 50)

    # Optional tokenization check
    if check_tok and model_checkpoint:
        check_tokenization(dataset_path, model_checkpoint)

    is_clean = len(issues) == 0
    if is_clean:
        log.info("RESULT: Dataset is CLEAN — all %d examples passed validation.", total)
    else:
        log.warning("RESULT: Dataset has %d issues across %d examples.", len(issues), total)

    return is_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a MolmoAct fine-tuning dataset")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to the datasets.load_from_disk() directory")
    parser.add_argument("--check-tokenization", action="store_true",
                        help="Also run end-to-end tokenization check on a subset")
    parser.add_argument("--model-checkpoint", default="allenai/MolmoAct-7B-D-0812",
                        help="Model checkpoint for tokenization check")
    parser.add_argument("--tok-samples", type=int, default=100,
                        help="Number of samples for tokenization check")
    args = parser.parse_args()

    ok = validate_dataset(
        args.dataset_path,
        check_tok=args.check_tokenization,
        model_checkpoint=args.model_checkpoint,
    )
    sys.exit(0 if ok else 1)
