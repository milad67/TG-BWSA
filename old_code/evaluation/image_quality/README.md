# Image-quality evaluation

The scripts apply Gaussian noise and three-dimensional Gaussian blur to preprocessed RFNet volumes, evaluate the checkpoints, and aggregate Dice scores. The clean input and eight degradation conditions are evaluated on the same cases.

Copy `config.example.json` and set the data path, checkpoint paths, model directories, and channel order. Run from the repository root:

```bash
python evaluation/image_quality/run_degradation.py --config path/to/config.json
python evaluation/image_quality/collect_results.py --config path/to/config.json
```

The output contains per-case scores, condition means, and run settings. Install `requirements.txt` in the RFNet environment before running the image transformations.
