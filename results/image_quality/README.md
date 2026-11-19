# Image-quality results

The existing `per_case.csv` and `summary.csv` retain their original scores and case IDs. Their case list is preserved at `old_code/data/splits/brats2023/validation.txt` from the repository root.

The updated evaluation uses `data/splits/brats2023/test.txt`, the confirmed main 251-case local-evaluation list. Its run configuration is `configs/image_quality_main_251.json`. The previous scores have not been reassigned to the replacement cases. A new evaluation is required to produce the replacement scores.
