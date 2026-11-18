# Results

- `local/`: per-case Dice scores for the two additional RFNet validation splits.
- `transfer/`: per-case Dice and HD95 for transfer from BraTS 2020 to BraTS 2023.
- `image_quality/`: per-case scores and condition means for the noise and blur experiments.
- `missing_modality/`: results across modality combinations.
- `lora/`: the LoRA comparison results.
- `online/`: official BraTS online evaluation reports and submission-level summaries.
- `statistics/`: paired patient-cluster bootstrap confidence intervals.
- `training/`: epoch-level validation metrics.

`computational_costs.csv` reports the model sizes and computational costs in the manuscript. Training-step times use the batch sizes listed in the table. Approximate values retain a `~` prefix. For pretrained MEFCT, trainable parameters are listed separately for the two fine-tuning phases.
