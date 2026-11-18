"""Paired patient-cluster confidence intervals for RFNet validation.

Splits are analyzed separately. Their overlapping validation patients are never
pooled as independent observations.
"""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/splits/brats2023"
MODELS = ("RFNet", "RFNet + BWSA", "RFNet + TG-BWSA")
EXPECTED_HASHES = {2026: "1f513c1ec747b42a7899dd93745ac1a52cc39103c46f565e694c3e5148330509",
                   2027: "b1e93d5a8ba6a47b40ed3879e75d89626c1686f406f63696c4001f504c32fa70"}


def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    result = {
        "method": "Paired patient-cluster percentile bootstrap, separately within each local validation split",
        "n_resamples": 10000, "analysis_seed": 20261112,
        "rng": "numpy.default_rng using SeedSequence(20261112).spawn(2), split order 2026 then 2027",
        "estimand": "MRI-case-weighted mean paired Dice difference; macro averages WT, TC and ET",
        "scope": "Conditional on each split's fitted and locally selected checkpoints. Not variation across retraining or across splits, and not an independent test of checkpoint-selection optimism.",
        "multiple_comparison_adjustment": "None; pointwise descriptive 95% intervals, no family-wise significance claims",
        "splits": {},
    }
    patient_sets, case_sets = [], []
    for split, child in zip((2026, 2027), np.random.SeedSequence(20261112).spawn(2)):
        path = ROOT / f"results/local/split_{split}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == EXPECTED_HASHES[split]
        rows = list(csv.DictReader(path.open()))
        values = {model: {} for model in MODELS}
        for row in rows:
            model, case = row["model"], row["case_id"]
            assert case not in values[model]
            roi = np.array([float(row[r]) for r in ("WT", "TC", "ET")])
            assert np.isfinite(roi).all() and np.all((roi >= 0) & (roi <= 1))
            values[model][case] = np.r_[roi.mean(), roi]
        ids = set((SOURCE / f"split_{split}/test.txt").read_text().split())
        assert len(ids) == 251 and all(set(v) == ids for v in values.values())
        patients = defaultdict(list)
        for case in sorted(ids):
            patients[case.rsplit("-", 1)[0]].append(case)
        assert len(patients) == 227
        counts = np.array([len(cases) for p, cases in sorted(patients.items())], dtype=float)
        assert np.count_nonzero(counts == 1) == 203 and np.count_nonzero(counts == 2) == 24
        cluster_sums = []
        for patient, cases in sorted(patients.items()):
            cluster_sums.append(np.concatenate([
                100 * np.stack([values[model][c] - values[MODELS[0]][c] for c in cases]).sum(axis=0)
                for model in MODELS[1:]
            ]))
        cluster_sums = np.stack(cluster_sums)
        rng = np.random.default_rng(child)
        weights = rng.multinomial(len(patients), np.full(len(patients), 1 / len(patients)), size=10000)
        assert np.all(weights.sum(axis=1) == len(patients))
        boot = (weights @ cluster_sums) / (weights @ counts)[:, None]
        point = cluster_sums.sum(axis=0) / len(ids)
        assert np.allclose((weights @ (0.02 * counts)) / (weights @ counts), .02)
        limits = np.percentile(boot, [2.5, 97.5], axis=0)
        comparisons = {}
        for model_index, model in enumerate(MODELS[1:]):
            comparisons[model] = {}
            for metric_index, metric in enumerate(("Mean_Dice", "WT", "TC", "ET")):
                i = model_index * 4 + metric_index
                expected = 100 * np.stack([values[model][c] - values[MODELS[0]][c] for c in sorted(ids)]).mean(axis=0)[metric_index]
                assert abs(point[i] - expected) < 1e-10
                comparisons[model][metric] = {"increase_pp": float(point[i]), "ci95_lower_pp": float(limits[0, i]), "ci95_upper_pp": float(limits[1, i])}
        result["splits"][str(split)] = {
            "source": str(path), "source_sha256": digest, "MRI_cases": len(ids), "patient_groups": len(patients),
            "means": {model: float(np.stack(list(v.values())).mean(axis=0)[0]) for model, v in values.items()},
            "comparisons_with_RFNet": comparisons,
        }
        patient_sets.append(set(patients))
        case_sets.append(ids)
    result["overlap_between_validation_splits"] = {"patients": len(patient_sets[0] & patient_sets[1]), "MRI_cases": len(case_sets[0] & case_sets[1]), "pooled_as_independent": False}
    assert result["overlap_between_validation_splits"]["patients"] == 53
    target = ROOT / "outputs/RFNet_Local_Resplits_Patient_Bootstrap_2026-11-12.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved", target)


if __name__ == "__main__":
    main()
