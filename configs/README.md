# Configuration

`rfnet_splits.json` contains the settings for the two additional RFNet splits. The final 900/100 internal lists and their seeds are recorded in the split manifests. `image_quality_main_251.json` configures a new evaluation on the confirmed main split. Set the data, implementation and checkpoint paths before running. The `image_quality/` files retain the settings recorded with the existing published image-quality results.

`tensorflow.yml` specifies the TensorFlow/Keras environment. Model scripts contain dataset and checkpoint paths that must be set for the local installation. Trained weights and experiment-specific server configurations are available from the corresponding author upon reasonable request.

The author-supplied PyTorch environment export is in `pytorch/`; `tensorflow.yml` records the TensorFlow/Keras environment.
