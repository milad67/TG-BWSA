# Data

Obtain the MRI images and segmentation labels from the BraTS challenge repositories under their access conditions.

`splits/brats2023/split_2026` and `split_2027` each contain 1,000 training and 251 validation cases, with all timepoints from a patient in the same partition. `validation.txt` lists the 251 cases used for the original RFNet local validation.

`splits/brats2020_to_2023/test.txt` lists the 886 transfer cases. `test_unseen_sites.txt` lists the nested subset of 466 cases from six institutions absent from the BraTS 2020 source cohort. The cohort metadata contains patient and institution identifiers from the official BraTS mapping.

The local `test.txt` filenames follow the model loader convention and serve as validation lists in these experiments.
