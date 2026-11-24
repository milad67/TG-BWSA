# BraTS 2023 online validation

Official evaluation reports for selected submissions are grouped by submission identifier. Region-wise and lesion-wise metrics are stored separately. Each report contains 219 MRI cases.

`submissions.csv` lists the regional Dice and HD95 means, submission identifiers, report paths and Synapse sources. HD95 is reported in millimeters.

## Manuscript submission manifest

[`online_submissions_manifest.csv`](online_submissions_manifest.csv) matches Table 10 models to the public reports using the six region-wise metrics at manuscript precision. It includes both metric types when a single submission can be assigned. The averages are arithmetic means of the three reported regions. The manifest does not include training seed scores or checkpoint epochs.

For the RFNet adapter rows, the author-provided mapping assigns `9752099` to RFNet + BWSA and `9752120` to RFNet + TG-BWSA. Each manifest row uses the region-wise and lesion-wise reports from that same submission. The lesion-wise scores therefore reflect the corrected model assignments for Table 11.

The authors identified `9751548` as the primary MEUNet + TG-BWSA submission and `9751572` as a repeat upload. The public region-wise and lesion-wise CSVs are identical for the two submissions.

[The MEFCT + TG-BWSA summary](mefct_tg_bwsa_manuscript_summary.csv) contains the region-wise values from Table 10. The submission ID and underlying Synapse report are pending. The manifest leaves the associated submission and report fields empty.
