# Missing-modality evaluation

`complete_input_configurations.csv` reports Dice means and standard deviations over three runs for models trained with all four modalities. The 15 input configurations are reported in Table 18. Values use the 0–1 scale and retain the two-decimal precision of the reported configuration scores.

`results.xlsx` contains configuration-level scores for RFNet, RFNet+BWSA, and RFNet+TG-BWSA trained with random modality masking. Workbook scores are percentages. These configurations are reported in Table 19.

`summary.csv` reports masking-trained Dice on the 0–1 scale and separates the complete-input score, the mean over the 14 incomplete-input configurations, and the mean over all 15 configurations. Table 17 compares the 14-configuration means under both training strategies.

The workbook columns `Mean` and `Mean_recomputed` average all 15 configurations, including complete inputs. They therefore differ from the incomplete-input mean in Table 17.
