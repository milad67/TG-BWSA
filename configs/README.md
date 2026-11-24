# Configuration

`rfnet_splits.json` contains the settings for the two additional RFNet splits. The final 900/100 internal lists and their seeds are recorded in the split manifests. `image_quality_main_251.json` is a template for evaluation on the confirmed main split. Set the data, implementation and checkpoint paths before running.

`tensorflow.yml` specifies the TensorFlow/Keras environment. Model scripts contain dataset and checkpoint paths that must be set for the local installation. Trained weights and experiment-specific server configurations are available from the corresponding author upon reasonable request.

The author-supplied PyTorch environment export is in `pytorch/`; `tensorflow.yml` records the TensorFlow/Keras environment.

[`brats2023_manuscript_protocol.json`](brats2023_manuscript_protocol.json) records the BraTS 2023 split paths, adapter settings and training settings described in the manuscript. It serves as a protocol reference. The model scripts use their own arguments and configuration formats. Use the original run manifests to identify the script revisions, seeds and checkpoints for reported results.
