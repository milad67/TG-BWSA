# Updated training and evaluation code

RFNet and mmFormer training entry points accept separate training, internal-validation and local-evaluation lists. They check case and patient overlap before loading data. A patient's examinations must remain in one partition.

| Directory | Model |
|---|---|
| `rfnet/baseline` | RFNet |
| `rfnet/adapters` | RFNet adapter implementations |
| `mmformer/baseline` | mmFormer |
| `mmformer/bwsa` | mmFormer with BWSA |
| `mmformer/tg_bwsa` | mmFormer with TG-BWSA |
| `meunet` | Multi-encoder U-Net and adapter variants |
| `mefct` | Multi-encoder FCT, adapter and LoRA variants |
| `mefct/pretraining` | Encoder pretraining |

The validation changes apply to the five RFNet/mmFormer `train.py` entry points. Model definitions, losses and preprocessing remain as in the original implementations. The 2D scripts are retained without a new split interface.

## Training

Use absolute paths. For example, from the RFNet baseline directory:

```bash
python train.py --datapath /path/to/processed_brats2023 \
  --savepath /path/to/runs/rfnet_split2026 \
  --train-list /path/to/split_2026/train.txt \
  --val-list /path/to/split_2026/val.txt \
  --test-list /path/to/split_2026/test.txt \
  --num_epochs 300 --batch_size 2 --seed 1024
```

During training, only training and internal-validation data are loaded. The local-evaluation list is checked for overlap but is not used for optimization or checkpoint selection. Internal validation runs after each epoch by default; `--val-every` changes this interval. `model_best.pth` stores the highest internal-validation mean Dice and the split-file hashes. `split_manifest.json` records the lists and training seed.

`--validation-mode full` selects by complete-input Dice. `--validation-mode all-subsets` selects by mean Dice over the 15 nonempty modality subsets. `--dice-indices` defaults to `0,1,3`, corresponding to WT, TC and the evaluated ET output in these predictors. Check this setting when using a different predictor.

For final local evaluation, use the same split arguments and add:

```bash
--resume /path/to/runs/rfnet_split2026/model_best.pth --evaluate-test
```

Set each model's training settings explicitly; for example, supply `--num_epochs 1000` for the manuscript's mmFormer setting. These entry points are for new executions. Newly generated validation lists do not identify the settings of previously trained checkpoints.

## Image-quality evaluation

Copy `updated_code/image_quality/config.example.json` and enter the data path, actual implementation directory and matching complete-input checkpoint for each model. The configuration uses the main 875/125/251 split. Confirm the checkpoint training lists and channel order before running.

```bash
python updated_code/image_quality/run_degradation.py --config /path/to/config.json
python updated_code/image_quality/collect_results.py --config /path/to/config.json
```

Run from the repository root, which contains `data/splits`, or use absolute list paths. New per-case results and an archive are created; existing result files are not overwritten.

## Environment

Run this in the Python environment used for RFNet/mmFormer:

```bash
python updated_code/collect_environment.py --output-dir server_environment_rfnet
```

Send the resulting `server_environment_rfnet.zip` for the revision. No GPU or model execution is needed. Run it separately for mmFormer if it uses a different environment.
