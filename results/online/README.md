# BraTS 2023 online validation

Official evaluation reports for selected submissions are grouped by submission identifier. Region-wise and lesion-wise metrics are stored separately. Each report contains 219 MRI cases.

`submissions.csv` lists the regional Dice and HD95 means, submission identifiers, report paths and Synapse sources. HD95 is reported in millimeters.

## Manuscript submission manifest

[`online_submissions_manifest.csv`](online_submissions_manifest.csv) matches Table 10 models to the public reports using the six region-wise metrics at manuscript precision. It includes both metric types when a single submission can be assigned. The averages are arithmetic means of the three reported regions. The manifest does not include training seed scores or checkpoint epochs.

For the RFNet adapter rows, the author-provided mapping assigns `9752099` to RFNet + BWSA and `9752120` to RFNet + TG-BWSA. Each manifest row uses the region-wise and lesion-wise reports from that same submission. The lesion-wise scores therefore reflect the corrected model assignments for Table 11.

MEUNet + TG-BWSA has two candidate submissions with the same region-wise scores. The exact submission ID is pending confirmation. The MEFCT + TG-BWSA row is awaiting a matching public report.
