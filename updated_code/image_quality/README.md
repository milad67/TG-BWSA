# Image-quality evaluation

The scripts evaluate clean inputs and eight noise/blur conditions on the confirmed main BraTS 2023 local-evaluation list. They check patient separation between the supplied train, internal-validation and local-evaluation lists.

Copy `config.example.json` and set the data path, checkpoint paths and matching implementation directories. Set `checkpoint_split_confirmed` and `channel_order_confirmed` only after checking those settings. Run from the repository root:

```bash
python updated_code/image_quality/run_degradation.py --config path/to/config.json
python updated_code/image_quality/collect_results.py --config path/to/config.json
```

Use complete-input checkpoints trained without the evaluation patients. A new split list does not change the training data of an existing checkpoint. Outputs include per-case scores, condition means, checkpoint hashes and hashes of all three split lists. Install `requirements.txt` in the RFNet environment before inference.

The existing published image-quality scores retain their original case list. Scores for the replacement list require a new evaluation and a separate output directory.
