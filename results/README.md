# Results

- `local/`: per-case Dice scores for the two additional RFNet validation splits.
- `transfer/`: per-case Dice and HD95 for transfer from BraTS 2020 to BraTS 2023.
- [Image quality](image_quality/): condition-level Dice summaries for the noise and blur evaluation in Table 18.
- [Missing modalities](missing_modality/): complete-input and masking-trained results for Tables 16 and 17.
- `lora/`: the LoRA comparison results.
- [Online validation](online/): official BraTS reports, submission-level summaries, and the Table 9 submission manifest.
- `statistics/`: paired patient-cluster bootstrap confidence intervals.
- `training/`: epoch-level validation metrics.

[`computational_costs.csv`](computational_costs.csv) transcribes Table 13, including model sizes, trainable parameters, time per training step, training batch size, and inference costs. For pretrained MEFCT, trainable parameters are listed separately for each fine-tuning phase.
