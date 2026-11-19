"""Explicit subject-disjoint splits and internal-validation checkpoint selection."""
import hashlib
import json
import re
from pathlib import Path


def patient_id(case):
    if re.fullmatch(r'BraTS-GLI-\d+-\d+', case):
        return case.rsplit('-', 1)[0]
    match = re.fullmatch(r'(?:HG_|LG_)?(BraTS20_Training_\d+)', case)
    return match.group(1) if match else case


def read_cases(path):
    values = [s.strip() for s in Path(path).read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f'Empty list or duplicate case IDs: {path}')
    if any(Path(s).name != s or s in ('.', '..') for s in values):
        raise ValueError(f'Expected case IDs, not paths: {path}')
    return values


def check_partitions(paths):
    parts = {role: read_cases(path) for role, path in paths.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        shared_cases = set(parts[a]) & set(parts[b])
        shared_patients = {patient_id(c) for c in parts[a]} & {patient_id(c) for c in parts[b]}
        if shared_cases or shared_patients:
            raise ValueError(f'{a}/{b} overlap: {len(shared_cases)} cases, {len(shared_patients)} patients')
    return parts


def add_split_arguments(parser):
    parser.add_argument('--train-list', required=True, help='Training case IDs; absolute or relative to datapath.')
    parser.add_argument('--val-list', required=True, help='Internal validation for checkpoint selection.')
    parser.add_argument('--test-list', required=True, help='Held-out local evaluation; not used for checkpoint selection.')
    parser.add_argument('--evaluate-test', action='store_true', help='Evaluate the explicitly supplied --resume checkpoint on test-list.')
    parser.add_argument('--val-every', type=int, default=1, help='Internal-validation interval in epochs.')
    parser.add_argument('--validation-mode', choices=['full', 'all-subsets'], default='full')
    parser.add_argument('--dice-indices', default='0,1,3', help='WT, TC and evaluated ET positions in predictor output.')


def resolve_split_files(args):
    paths = {}
    for role in ['train', 'val', 'test']:
        path = Path(getattr(args, role+'_list')).expanduser()
        if not path.is_absolute():
            path = Path(args.datapath) / path
        paths[role] = str(path.resolve(strict=True))
    check_partitions(paths)
    if args.val_every < 1:
        raise ValueError('--val-every must be positive.')
    if args.evaluate_test and args.resume is None:
        raise ValueError('--evaluate-test requires an explicit --resume checkpoint.')
    return paths


def record_split_manifest(paths, output_dir, training_seed):
    parts = check_partitions(paths)
    record = {'training_seed': training_seed, 'checkpoint_selection_partition': 'val', 'files': {}}
    for role, path in paths.items():
        record['files'][role] = {'path': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                 'cases': len(parts[role]), 'patients': len({patient_id(c) for c in parts[role]})}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'split_manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def internal_validation(model, loader, evaluator, args, masks):
    import numpy as np
    import torch
    previous_mode = model.training
    module = getattr(model, 'module', model)
    previous_flag = getattr(module, 'is_training', None)
    selected = masks if args.validation_mode == 'all-subsets' else [[True, True, True, True]]
    indices = [int(i) for i in args.dice_indices.split(',')]
    if len(indices) != 3:
        raise ValueError('--dice-indices must identify WT, TC and ET.')
    scores = []
    try:
        model.eval()
        with torch.no_grad():
            for mask in selected:
                values = np.asarray(evaluator(loader, model, dataname=args.dataname, feature_mask=mask), dtype=float).reshape(-1)
                if max(indices) >= len(values) or min(indices) < 0:
                    raise ValueError('Check --dice-indices against this predictor output.')
                regional = values[indices]
                if not np.isfinite(regional).all() or not ((regional >= 0) & (regional <= 1)).all():
                    raise ValueError('Validation returned invalid Dice values.')
                scores.append(float(regional.mean()))
    finally:
        model.train(previous_mode)
        if previous_flag is not None:
            module.is_training = previous_flag
    return float(np.mean(scores))


def save_best_checkpoint(score, best, model, optimizer, epoch, output_dir, manifest):
    if score <= best:
        return best
    import torch
    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optim_dict': optimizer.state_dict(),
                'best_validation_dice': score, 'split_manifest': manifest}, Path(output_dir)/'model_best.pth')
    return score
