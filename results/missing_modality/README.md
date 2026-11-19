# Missing-modality evaluation

`results.xlsx` contains configuration-level scores for RFNet, RFNet+BWSA, and RFNet+TG-BWSA trained with random modality masking. Workbook scores are percentages.

`summary.csv` reports Dice on the 0–1 scale and separates the complete-input score, the mean over the 14 incomplete-input configurations, and the mean over all 15 configurations. Table 17 uses the 14-configuration mean; Table 18 reports individual configurations.

The workbook columns `Mean` and `Mean_recomputed` average all 15 configurations, including complete inputs. They therefore differ from the incomplete-input mean in Table 17.
