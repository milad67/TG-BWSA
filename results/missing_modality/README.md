# Missing-modality evaluation

`complete_input_configurations.csv` reports Dice means and standard deviations over three runs for models trained with all four modalities. The 15 input configurations are reported in Table 17. Values use the 0–1 scale and retain four decimal places for the reported means and standard deviations. Modality configurations follow the author-confirmed FLAIR, T1, T2, T1ce mask order.

`results.xlsx` contains configuration-level scores for RFNet, RFNet+BWSA, and RFNet+TG-BWSA trained with random modality masking. Workbook scores are percentages. These configurations are reported in Table 18.

`summary.csv` reports masking-trained Dice on the 0–1 scale and separates the complete-input score, the mean over the 14 incomplete-input configurations, and the mean over all 15 configurations. Tables 17 and 18 report the 14-configuration means for their respective training strategies.

The workbook columns `Mean` and `Mean_recomputed` average all 15 configurations, including complete inputs. They therefore differ from the incomplete-input mean in Table 18.
