"""Paired patient-cluster confidence intervals for RFNet transfer.

Patients are resampled within each institution, retaining all their timepoints.
The estimand is the original case-weighted mean paired difference. Institutions
and trained model instances are fixed, not resampled or retrained.
"""

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "data/splits/brats2020_to_2023"
MODELS = ("RFNet", "RFNet + BWSA", "RFNet + TG-BWSA")
ENDPOINTS = (
    "BWSA_Dice_increase_pp", "BWSA_HD95_decrease_mm",
    "TG_BWSA_Dice_increase_pp", "TG_BWSA_HD95_decrease_mm",
)
SEED = 20261112
N_RESAMPLES = 10000


def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    source = ROOT / "results/transfer/per_case.csv"
    content = source.read_bytes()
    expected_sha256 = "59c4327a3c78edbde0a7a3fa376d56282d9410aaca5257dea080ca40f56d0644"
    assert hashlib.sha256(content).hexdigest() == expected_sha256
    records = list(csv.DictReader(content.decode("utf-8-sig").splitlines()))
    metadata = {r["case_id"]: r for r in json.loads((COHORT / "cohort_metadata.json").read_text())["records"]}
    values = {}
    for model in MODELS:
        selected = [r for r in records if r["model"] == model]
        assert len(selected) == 886 and len({r["case_id"] for r in selected}) == 886
        values[model] = {
            r["case_id"]: np.array([
                sum(float(r[k]) for k in ("WT", "TC", "ET")) / 3,
                sum(float(r[k]) for k in ("HD95_WT", "HD95_TC", "HD95_ET")) / 3,
            ]) for r in selected
        }
        assert set(values[model]) == set(metadata)
    patient_sites = defaultdict(set)
    for r in metadata.values():
        patient_sites[r["patient_id"]].add(r["site_no"])
    assert all(len(v) == 1 for v in patient_sites.values())
    scopes = {
        "all_886": sorted(metadata),
        "unseen_sites_466": sorted(c for c in metadata if metadata[c]["unseen_site_relative_to_2020_mapping"]),
    }
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "method": "Paired patient-cluster percentile bootstrap, stratified by institution",
        "estimand": "Case-weighted mean difference in macro Dice or macro HD95; macro averages WT, TC and ET",
        "n_resamples": N_RESAMPLES,
        "numpy_version": np.__version__,
        "rng": "numpy.default_rng with SeedSequence(20261112).spawn(2), cohort order all_886 then unseen_sites_466",
        "bootstrap_seed_not_training_seed": SEED,
        "confidence_level": 0.95,
        "quantiles": [2.5, 97.5],
        "multiple_comparison_adjustment": "None; pointwise nominal CIs, not simultaneous intervals or family-wise tests",
        "resampling": "Within each fixed institution, draw the original number of patient clusters with replacement; keep all timepoints together and use the same cluster weights for both methods",
        "interpretation": "Conditional on these fitted models, fixed represented institutions, and their observed patient allocation; not uncertainty across training seeds or a population of unseen institutions",
        "cohorts": {},
    }
    children = np.random.SeedSequence(SEED).spawn(len(scopes))
    for index, (name, ids) in enumerate(scopes.items()):
        patients = defaultdict(list)
        for case in ids:
            patients[(metadata[case]["site_no"], metadata[case]["patient_id"])].append(case)
        strata = defaultdict(list)
        for (site, patient), cases in sorted(patients.items()):
            numerator = []
            for model in MODELS[1:]:
                differences = np.stack([values[model][c] - values[MODELS[0]][c] for c in cases]).sum(axis=0)
                numerator.extend([100 * differences[0], -differences[1]])
            strata[site].append((len(cases), np.asarray(numerator)))
        boot_numerator = np.zeros((N_RESAMPLES, len(ENDPOINTS)))
        boot_denominator = np.zeros(N_RESAMPLES)
        point_numerator = np.zeros(len(ENDPOINTS))
        rng = np.random.default_rng(children[index])
        for site, clusters in sorted(strata.items()):
            n = len(clusters)
            counts = np.array([c[0] for c in clusters], dtype=float)
            sums = np.stack([c[1] for c in clusters])
            weights = rng.multinomial(n, np.full(n, 1 / n), size=N_RESAMPLES)
            assert np.all(weights.sum(axis=1) == n)
            boot_numerator += weights @ sums
            boot_denominator += weights @ counts
            point_numerator += sums.sum(axis=0)
            # A constant per-case effect must remain constant despite unequal visits.
            assert np.allclose((weights @ (0.02 * counts)) / (weights @ counts), 0.02)
        points = point_numerator / len(ids)
        intervals = np.percentile(boot_numerator / boot_denominator[:, None], [2.5, 97.5], axis=0)
        endpoints = {
            endpoint: {
                "estimate": float(points[j]),
                "ci95_lower": float(intervals[0, j]),
                "ci95_upper": float(intervals[1, j]),
            }
            for j, endpoint in enumerate(ENDPOINTS)
        }
        result["cohorts"][name] = {
            "MRI_cases": len(ids), "patient_groups": len(patients),
            "institution_count": len(strata), "endpoints": endpoints,
        }
        print(name, endpoints)

    unseen = scopes["unseen_sites_466"]
    sensitivity = {}
    for excluded_site in sorted({metadata[c]["site_no"] for c in unseen}):
        retained = [c for c in unseen if metadata[c]["site_no"] != excluded_site]
        means = {model: np.stack([values[model][c] for c in retained]).mean(axis=0) for model in MODELS}
        sensitivity[str(excluded_site)] = {
            "excluded_institution": excluded_site,
            "remaining_MRI_cases": len(retained),
            "remaining_patient_groups": len({metadata[c]["patient_id"] for c in retained}),
            "models": {model: {"Dice_macro": float(v[0]), "HD95_macro_mm": float(v[1])} for model, v in means.items()},
            "BWSA_Dice_increase_pp": float(100 * (means[MODELS[1]][0] - means[MODELS[0]][0])),
            "TG_BWSA_Dice_increase_pp": float(100 * (means[MODELS[2]][0] - means[MODELS[0]][0])),
        }
    result["institution_omission_sensitivity"] = {
        "method": "Omit each of the six institutions from aggregation once; no model retraining and not leave-one-site-out cross-validation",
        "results": sensitivity,
    }
    assert source.read_bytes() == content
    target = ROOT / "outputs/RFNet_Transfer_Patient_Bootstrap_2026-11-12.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Without institution 18:", sensitivity["18"])
    print("Saved", target)


if __name__ == "__main__":
    main()
