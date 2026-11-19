# Data

Obtain the MRI images and segmentation labels from the BraTS challenge repositories under their access conditions.

## Main experiments

| Dataset | Training | Internal validation | Local evaluation |
|---|---:|---:|---:|
| BraTS 2020 | [219](splits/brats2020/train.txt) | [50](splits/brats2020/val.txt) | [100](splits/brats2020/test.txt) |
| BraTS 2023 | [875](splits/brats2023/train.txt) | [125](splits/brats2023/val.txt) | [251](splits/brats2023/test.txt) |

The partitions are subject-disjoint. All MRI examinations from a patient remain in one partition. Internal validation is used for checkpoint selection. The separate local-evaluation partition provides the local-validation scores reported in the manuscript. The official BraTS 2023 online validation set is separate from these lists.

## Additional RFNet experiments

`splits/brats2023/split_2026` and `split_2027` each contain `development.txt` (1,000 cases), `train.txt` (900), `val.txt` (100), and `test.txt` (251). The generated train/val subdivision is patient-disjoint and is provided for new runs. The two-examination patients are kept together. `split_manifest.json` records the generation settings and counts. These generated internal lists do not identify the internal lists used by previously trained checkpoints.

`splits/brats2023/validation.txt` now matches the confirmed main 251-case local-evaluation list. The former list associated with the existing image-quality results is preserved at `old_code/data/splits/brats2023/validation.txt` from the repository root. New scores for the replacement list require a new checkpoint evaluation.

`splits/brats2020_to_2023/test.txt` lists the 886 transfer cases. `test_unseen_sites.txt` lists the nested subset of 466 cases from six institutions absent from the BraTS 2020 source cohort. The cohort metadata contains patient and institution identifiers from the official BraTS mapping.

The local `test.txt` filenames follow the model loader convention. They identify local-evaluation cases and do not refer to the official BraTS challenge test set.
