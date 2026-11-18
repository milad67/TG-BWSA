"""Reuse each final RFNet project's inference and postprocessing functions.

This script never imports training scripts and never retrains a model. Each model
runs in a separate process to avoid mixing different `models`/`layers` modules.
"""
import argparse
import csv
import hashlib
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
from degradation import CONDITIONS, apply_condition, foreground_from_clean, noise_seed, to_chwd

FIELDS = ["case_id", "model", "training_regime", "checkpoint_id", "checkpoint_sha256",
          "condition", "target_modality", "noise_sigma", "blur_sigma_vox", "noise_seed",
          "dice_wt", "dice_tc", "dice_et", "hd95_wt_mm", "hd95_tc_mm", "hd95_et_mm"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_cases(path):
    cases = [x.strip() for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError("Case list is empty or contains duplicates.")
    if any(Path(x).name != x or x in (".", "..") for x in cases):
        raise ValueError("Case IDs must be plain names, not file paths.")
    return cases


def load_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    text = json.dumps(c)
    if "EDIT_" in text:
        raise ValueError("Fill all EDIT_ placeholders in config.json first.")
    if c.get("dataset") != "BraTS2023" or c.get("training_regime") != "full_input":
        raise ValueError("This experiment uses BraTS2023 and the existing full-input checkpoints.")
    if not c.get("channel_order_confirmed"):
        raise ValueError("Check the real loader/encoder channel order, then set channel_order_confirmed=true.")
    if not c.get("same_original_validation_cohort_confirmed"):
        raise ValueError("Confirm this is the original validation list for these checkpoints.")
    if sorted(c["channel_order"]) != sorted(["FLAIR", "T1ce", "T1", "T2"]):
        raise ValueError("Incorrect channel_order.")
    return c


def input_paths(c, case):
    root = Path(c["data_root"])
    return (root / "vol" / (case + "_vol.npy"),
            root / "seg" / (case + "_seg.npy"))


def get_foreground(c, case, clean):
    mask_dir = c.get("foreground_dir")
    supplied = np.load(Path(mask_dir) / (case + "_foreground.npy"), allow_pickle=False) if mask_dir else None
    return foreground_from_clean(clean, c["foreground_method"], supplied)


def preflight(c):
    cases = read_cases(c["test_file"])
    if len(cases) != int(c["expected_cases"]):
        raise ValueError("Found %s cases, expected %s. Verify the ORIGINAL list, not a new split." %
                         (len(cases), c["expected_cases"]))
    manifest = []
    for case in cases:
        v, s = input_paths(c, case)
        if not v.is_file() or not s.is_file():
            raise FileNotFoundError("Missing vol/seg for " + case)
        x = np.load(v, mmap_mode="r", allow_pickle=False)
        y = np.load(s, mmap_mode="r", allow_pickle=False)
        spatial = x.shape[:3] if c["layout"] == "HWDC" else x.shape[1:]
        channels = x.shape[-1] if c["layout"] == "HWDC" else x.shape[0]
        if x.ndim != 4 or channels != 4 or tuple(y.shape) != tuple(spatial):
            raise ValueError("Input/target shape mismatch: " + case)
        if not np.issubdtype(x.dtype, np.floating):
            raise ValueError("Expected preprocessed floating-point input: " + case)
        if not np.isin(np.unique(y), [0, 1, 2, 3]).all():
            raise ValueError("Expected the existing RFNet internal labels 0,1,2,3: " + case)
        manifest.append({"case_id": case, "volume": str(v.resolve()), "segmentation": str(s.resolve()),
                         "shape": list(x.shape), "dtype": str(x.dtype),
                         "volume_bytes": v.stat().st_size, "volume_mtime_ns": v.stat().st_mtime_ns,
                         "seg_bytes": s.stat().st_size, "seg_mtime_ns": s.stat().st_mtime_ns})
    # Only inspect one volume's foreground here; every case is checked on access.
    sample = to_chwd(np.load(input_paths(c, cases[0])[0], allow_pickle=False), c["layout"])
    _, foreground = get_foreground(c, cases[0], sample)
    return cases, manifest, foreground


def source_record(function):
    path = inspect.getsourcefile(function)
    return {"file": path, "sha256": sha256(path) if path else None,
            "function": function.__name__}


def run_model(c, config_path, model_config, clean_only=False, device="cuda"):
    import torch
    from torch.utils.data import DataLoader, Dataset

    cases, manifest, fg_example = preflight(c)
    name = model_config["name"]
    output = Path(c["output_dir"]) / name
    output.mkdir(parents=True, exist_ok=True)
    selected = CONDITIONS[:1] if clean_only else CONDITIONS
    expected_output = [output / (cond.name + ".csv") for cond in selected]
    if any(p.exists() for p in expected_output):
        raise FileExistsError("Result files already exist. Choose a NEW output_dir; do not mix runs.")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(output / "evaluation.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    repo = Path(model_config["repo_dir"]).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(repo)
    if any(p.name.startswith("train") for p in [Path(model_config["model_module"])]):
        raise ValueError("Import a model definition, not a training script.")
    sys.path.insert(0, str(repo))
    model_module = importlib.import_module(model_config["model_module"])
    predict_module = importlib.import_module(model_config["predict_module"])
    for module in (model_module, predict_module):
        if repo not in Path(module.__file__).resolve().parents:
            raise ImportError("Wrong module loaded: " + str(module.__file__))
    factory = getattr(model_module, model_config["model_class"])
    model = factory(**model_config["model_kwargs"])
    checkpoint = Path(model_config["checkpoint"]).resolve()
    # Only load the author's own trusted checkpoints. No network downloads.
    load_kw = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    state = torch.load(str(checkpoint), **load_kw)
    key = model_config["checkpoint_key"]
    if key:
        state = state[key]
    prefix = model_config.get("state_prefix_to_strip", "module.")
    if prefix:
        if all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]:v for k, v in state.items()}
        elif any(k.startswith(prefix) for k in state):
            raise ValueError("Mixed state-dict prefixes. Verify the checkpoint.")
    model.load_state_dict(state, strict=True)
    del state
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. This script does not substitute a different device silently.")
    model.to(device)
    if hasattr(model, "is_training"):
        model.is_training = False
    model = torch.nn.DataParallel(model)
    model.eval()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(int(c["evaluation_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(c["evaluation_seed"]))

    metric_name = model_config["metric_function"]
    original_metric = getattr(predict_module, metric_name)
    evaluator = getattr(predict_module, model_config["evaluation_function"])
    checkpoint_hash = sha256(checkpoint)
    metadata = {"config": c, "model_config": model_config,
                "config_sha256": sha256(config_path), "checkpoint_sha256": checkpoint_hash,
                "test_file_sha256": sha256(c["test_file"]), "cases": cases,
                "input_manifest": manifest, "foreground_example": fg_example,
                "utc_started": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU_TEST_ONLY",
                "metric_source": source_record(original_metric),
                "evaluator_source": source_record(evaluator),
                "model_source": source_record(factory),
                "degradation_source_sha256": sha256(Path(__file__).with_name("degradation.py")),
                "runner_source_sha256": sha256(__file__),
                "availability_mask": [True, True, True, True],
                "noise_seed_method": "SHA256(base_seed|case_id|modality), first 8 bytes, big endian; numpy.default_rng(PCG64)",
                "noise_stage": "after existing preprocessing, full volume, before inference windows",
                "postprocessing": "unchanged original metric function from the configured final project",
                "score_scale": "0 to 1", "hd95": "not computed by this wrapper; blank is not zero",
                "clean_only": clean_only}
    import scipy
    metadata["scipy"] = scipy.__version__
    save_json(output / "run_config.json", metadata)
    (output / "test_cases.txt").write_text("\n".join(cases) + "\n", encoding="utf-8")
    failures = []
    summary = []
    foreground_checks = {}
    active_case = {}

    class DegradedDataset(Dataset):
        def __init__(self, condition):
            self.condition = condition

        def __len__(self):
            return len(cases)

        def __getitem__(self, index):
            case = cases[index]
            v, s = input_paths(c, case)
            old = manifest[index]
            if (v.stat().st_size, v.stat().st_mtime_ns, s.stat().st_size, s.stat().st_mtime_ns) != (
                old["volume_bytes"], old["volume_mtime_ns"], old["seg_bytes"], old["seg_mtime_ns"]):
                raise ValueError("Input files changed during evaluation: " + case)
            clean = to_chwd(np.load(v, allow_pickle=False), c["layout"])
            fg, info = get_foreground(c, case, clean)
            foreground_checks[case] = info
            x = apply_condition(clean, fg, self.condition, c["channel_order"], case, c["noise_base_seed"])
            y = np.asarray(np.load(s, allow_pickle=False), dtype=np.int64)
            active_case["case_id"] = case
            return torch.from_numpy(x), torch.from_numpy(y), case

    try:
        for condition in selected:
            records = []

            def record_metric(*args, **kwargs):
                result = original_metric(*args, **kwargs)
                scores = result[model_config["metric_return_index"]]
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().numpy()
                scores = np.asarray(scores)
                if scores.ndim != 2 or scores.shape[0] != 1:
                    raise ValueError("Expected one case per metric call. Do not change inference batch size.")
                position = len(records)
                if position >= len(cases):
                    raise ValueError("Metric called more than once per case. Adapt the recorder to the final evaluator.")
                dice = scores[0, model_config["dice_indices"]].astype(float)
                if len(dice) != 3 or not np.isfinite(dice).all() or (dice < 0).any() or (dice > 1+1e-6).any():
                    raise ValueError("Unexpected Dice values. Check the final metric function and selected indices.")
                case = cases[position]
                if active_case.get("case_id") != case:
                    raise ValueError("Metric calls do not match the loaded case order. Adapt the recorder to the final evaluator.")
                records.append({"case_id": case, "model": name, "training_regime": c["training_regime"],
                    "checkpoint_id": checkpoint.name, "checkpoint_sha256": checkpoint_hash,
                    "condition": condition.name, "target_modality": condition.modality,
                    "noise_sigma": condition.noise_sigma, "blur_sigma_vox": condition.blur_sigma_vox,
                    "noise_seed": str(noise_seed(case, condition.modality, c["noise_base_seed"])) if condition.noise_sigma else "",
                    "dice_wt": float(dice[0]), "dice_tc": float(dice[1]), "dice_et": float(dice[2]),
                    "hd95_wt_mm": "", "hd95_tc_mm": "", "hd95_et_mm": ""})
                return result  # Keep the existing evaluator and PP behavior unchanged.

            setattr(predict_module, metric_name, record_metric)
            loader = DataLoader(DegradedDataset(condition), batch_size=1, shuffle=False, num_workers=0)
            logging.info("START %s / %s / N=%s", name, condition.name, len(cases))
            kwargs = dict(model_config["evaluation_kwargs"])
            kwargs["feature_mask"] = [True, True, True, True]
            kwargs["mask_name"] = condition.name
            model.eval()
            with torch.no_grad():
                evaluator(loader, model, **kwargs)
            if len(records) != len(cases):
                raise ValueError("Recorded %s cases, expected %s. Metric hook is not compatible with this evaluator." %
                                 (len(records), len(cases)))
            with open(output / (condition.name + ".csv"), "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(records)
            means = {key: float(np.mean([r[key] for r in records])) for key in ("dice_wt", "dice_tc", "dice_et")}
            summary.append({"model": name, "condition": condition.name, "n_cases": len(cases), **means})
            logging.info("DONE %s / %s / %s", name, condition.name, means)
    except Exception:
        failures.append({"error": traceback.format_exc(), "condition": condition.name})
        logging.exception("Evaluation stopped; incomplete runs are not presented as complete.")
        raise
    finally:
        setattr(predict_module, metric_name, original_metric)
        save_json(output / "failures.json", failures)
        save_json(output / "foreground_checks.json", foreground_checks)
        save_json(output / "summary.json", summary)
    save_json(output / "completion.json", {"complete": True, "n_cases": len(cases),
              "n_conditions": len(selected), "n_rows": len(cases)*len(selected),
              "utc_finished": datetime.now(timezone.utc).isoformat()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Name of one model; omit to launch each in its own process.")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="CPU is for synthetic tool checks. Original RFNet evaluator requires CUDA.")
    args = parser.parse_args()
    c = load_config(args.config)
    if args.preflight_only:
        cases, manifest, foreground = preflight(c)
        out = Path(c["output_dir"])
        save_json(out / "preflight.json", {"n_cases": len(cases), "input_manifest": manifest,
                                          "foreground_example": foreground,
                                          "config_sha256": sha256(args.config)})
        print("Input preflight passed for %s cases. No model was run." % len(cases))
        return
    if args.model:
        selected = [m for m in c["models"] if m["name"] == args.model]
        if len(selected) != 1:
            raise ValueError("Unknown or duplicate model name.")
        run_model(c, Path(args.config).resolve(), selected[0], args.clean_only, args.device)
    else:
        for m in c["models"]:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve()),
                   "--model", m["name"], "--device", args.device]
            if args.clean_only:
                cmd.append("--clean-only")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
