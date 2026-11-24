# BraTS 2023 online validation

Official evaluation reports for selected submissions are grouped by submission identifier. Region-wise and lesion-wise metrics are stored separately. Each report contains 219 MRI cases.

`submissions.csv` lists the regional Dice and HD95 means, submission identifiers, report paths and Synapse sources. HD95 is reported in millimeters.

## Manuscript submission manifest

[`online_submissions_manifest.csv`](online_submissions_manifest.csv) lists the Table 9 models and links the available original reports. Report-linked entries are matched using the six region-wise metrics at manuscript precision. The `match_status` field distinguishes these entries from manuscript summaries. Region-wise and lesion-wise metrics are listed separately. Scores transcribed from manuscript summaries retain their reported precision. The manifest does not include training seed scores or checkpoint epochs.

For the RFNet adapter variants, the manifest links the Table 9 region-wise results. The manuscript is the reference for the named lesion-wise results in Table 10.

The authors identified `9751548` as the primary MEUNet + TG-BWSA submission and `9751572` as a repeat upload. The public region-wise and lesion-wise CSVs are identical for the two submissions.

For MEFCT + TG-BWSA, [the manuscript summary](mefct_tg_bwsa_manuscript_summary.csv) records the region-wise values reported in Table 9. The original Synapse report and submission identifier are unavailable for this entry. The manifest labels it `manuscript_summary_only`, lists the reported scores, and leaves the Synapse identifier and report fields empty.
