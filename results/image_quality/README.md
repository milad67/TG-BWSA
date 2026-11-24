# Image-quality evaluation

[`degradation_summary.csv`](degradation_summary.csv) reports WT, TC and ET Dice for the three RFNet variants in Table 18. Scores use the 0–1 scale. The aggregate Dice and drops from clean inputs are calculated from the reported regional values.

The evaluation covers 251 BraTS 2023 local-evaluation cases, clean inputs and eight synthetic image-quality conditions. Gaussian noise or three-dimensional Gaussian blur is applied to T1ce or FLAIR while all four modalities remain available.

This directory provides condition-level summaries. The [case list](../../data/splits/brats2023/test.txt), [evaluation scripts](../../updated_code/image_quality/) and [configuration template](../../configs/image_quality_main_251.json) are available separately.
