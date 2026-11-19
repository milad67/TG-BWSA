# Image-quality results

`per_case.csv` contains 6,777 rows for three RFNet variants, 251 BraTS 2023 cases and nine input conditions. `summary.csv` reports condition means and differences from clean inputs. Scores use the 0–1 Dice scale.

The evaluated cases match `data/splits/brats2023/test.txt`, the main local-evaluation partition. Clean inputs and all eight noise/blur conditions retain all four modalities. These results are reported in Table 19. The updated evaluation scripts are in `updated_code/image_quality/` and the configuration template is `configs/image_quality_main_251.json`.
