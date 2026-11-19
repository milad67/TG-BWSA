# Data

Obtain the MRI images and segmentation labels from the BraTS challenge repositories under their access conditions.

## Main experiments

| Dataset | Training | Internal validation | Local evaluation |
|---|---:|---:|---:|
| BraTS 2020 | [219](splits/brats2020/train.txt) | [50](splits/brats2020/val.txt) | [100](splits/brats2020/test.txt) |
| BraTS 2023 | [875](splits/brats2023/train.txt) | [125](splits/brats2023/val.txt) | [251](splits/brats2023/test.txt) |

The partitions are subject-disjoint. All MRI examinations from a patient remain in one partition. Internal validation is used for checkpoint selection. The separate local-evaluation partition provides the local-validation scores reported in the manuscript. The official BraTS 2023 online validation set is separate from these lists.

## Additional RFNet experiments

`splits/brats2023/split_2026` and `split_2027` contain the development and local-evaluation lists for the two additional RFNet splits. Each `train.txt` lists 1,000 development cases, divided into 900 training and 100 internal-validation cases in the experiments. Each `test.txt` lists 251 local-evaluation cases. All timepoints from a patient remain in the same outer partition.

`splits/brats2020_to_2023/test.txt` lists the 886 transfer cases. `test_unseen_sites.txt` lists the nested subset of 466 cases from six institutions absent from the BraTS 2020 source cohort. The cohort metadata contains patient and institution identifiers from the official BraTS mapping.

The local `test.txt` filenames follow the model loader convention. They identify local-evaluation cases and do not refer to the official BraTS challenge test set.
