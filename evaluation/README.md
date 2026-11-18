# Evaluation

Run the statistical analyses from the repository root:

```bash
python evaluation/bootstrap_local.py
python evaluation/bootstrap_transfer.py
```

Results are written to `outputs/`. Both scripts use 10,000 paired patient-cluster bootstrap resamples and report pointwise 95% confidence intervals. All MRI examinations from a patient are kept together. The transfer analysis samples patients within each institution. The intervals describe uncertainty for the fitted checkpoints and evaluated cohorts.

The image-quality scripts are in `image_quality/`. Set dataset and checkpoint paths and confirm the modality order in the configuration before inference.
