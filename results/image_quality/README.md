# Image-quality results

[`degradation_summary.csv`](degradation_summary.csv) transcribes the manuscript WT, TC and ET Dice values from Table 19 for the three RFNet variants. The aggregate Dice and drops from clean inputs were calculated from the rounded table values. The experiment evaluates complete-input checkpoints on clean inputs and eight synthetic corruptions. Gaussian noise and three-dimensional Gaussian blur were applied to either T1ce or FLAIR, with all four modalities available to the model.

[`per_case.csv`](per_case.csv) and [`summary.csv`](summary.csv) are archived outputs from a preliminary run covering 251 BraTS 2023 cases and nine conditions. Their scores differ from Table 19. The case list is [the main local-evaluation split](../../data/splits/brats2023/test.txt).

The final per-case scores for Table 19 are pending. The [evaluation scripts](../../updated_code/image_quality/) and [configuration template](../../configs/image_quality_main_251.json) describe how to run the evaluation.
