"""Evaluate saved Area 5 LSeg predictions once per occupied fixed-grid voxel.

This is a voxel-only re-evaluation, without inverse mapping to raw points.
The first original point represents each voxel, including its prediction, GT,
and feature-coverage status. No score pooling or semantic majority voting.
Grid quantization follows OpenScene Voxelizer with augmentation disabled.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from metrics import CLASSES, confusion_matrix, summarize

WORKSPACE = Path(__file__).resolve().parent.parent
OPENSCENE_COMMIT = '0f369bc73d0724ae24b5e46bbada193f8ee9d193'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def quantize(xyz, voxel_size):
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError('voxel_size must be positive and finite')
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Expected nonempty, finite N by 3 coordinates')
    # OpenScene multiplies float32 homogeneous coordinates by a float64
    # diagonal matrix before flooring. Subtract the integer grid minimum
    # AFTER flooring, rather than translating the continuous coordinates.
    grid = np.floor(xyz.astype(np.float64) * (1.0 / voxel_size)).astype(np.int64)
    minimum = grid.min(axis=0)
    grid -= minimum
    unsigned = grid.astype(np.uint64)
    key = np.full(len(grid), np.uint64(14695981039346656037), dtype=np.uint64)
    for dimension in range(3):
        key *= np.uint64(1099511628211)
        key ^= unsigned[:, dimension]
    _, indices, inverse, counts = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True)
    selected_grid = grid[indices]
    # Check every original point, including possible hash collisions.
    for begin in range(0, len(grid), 500_000):
        end = begin + 500_000
        if not np.array_equal(grid[begin:end], selected_grid[inverse[begin:end]]):
            raise AssertionError('Voxel hash collision or incorrect inverse mapping')
    if int(counts.sum()) != len(xyz):
        raise AssertionError('Voxel occupancy counts do not recover all input points')
    return selected_grid, indices, counts, minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=WORKSPACE / 'area5_manifest.json')
    parser.add_argument('--predictions', type=Path, default=WORKSPACE / 'results/area5_lseg')
    parser.add_argument('--output', type=Path, default=WORKSPACE / 'results/area5_lseg_voxel_004')
    parser.add_argument('--voxel-size', type=float, default=0.04)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source_summary_path = args.predictions / 'summary.json'
    source_summary = json.loads(source_summary_path.read_text())
    if manifest['area'] != 5 or len(manifest['rooms']) != 68:
        raise ValueError('Expected the complete 68-room Area 5 manifest')
    if not source_summary['complete'] or source_summary['evaluated_rooms'] != 68:
        raise ValueError('The original-point prediction run is incomplete')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(
        area=5, expected_rooms=68, voxel_size_m=args.voxel_size,
        grid='floor(float64(original_xyz) * (1 / voxel_size)); subtract integer room minimum',
        grid_axes='unaligned S3DIS world axes', augmentation=False,
        voxel_order='OpenScene FNV64 hash order',
        representative='first point in original file order within each occupied voxel',
        prediction_and_ground_truth='both taken from the same representative point',
        feature_aggregation='reuse representative point prediction; no voxel score averaging',
        inverse_mapping_to_all_points=False, weighting='one vote per occupied voxel',
        no_feature_policy='prediction -1 contributes a false negative in both metrics',
        metric='global 13-class confusion counts over all occupied voxels in all 68 rooms',
        classes=list(CLASSES), openscene_commit=OPENSCENE_COMMIT,
        openscene_note='quantizer with augmentation disabled; upstream evaluate.py normally uses eval_all=True',
        manifest_sha256=sha256(args.manifest),
        source_summary_sha256=sha256(source_summary_path),
        source_config_sha256=sha256(args.predictions / 'config.json'),
        evaluator_sha256=sha256(__file__), metrics_sha256=sha256(Path(__file__).with_name('metrics.py')),
        numpy_version=np.__version__, torch_version=torch.__version__)
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'summary.json', dict(complete=False, evaluated_rooms=0, expected_rooms=68))
    total = np.zeros((13, 14), dtype=np.int64)
    original_total = np.zeros_like(total)
    records = []
    start = time.monotonic()
    for room in manifest['rooms']:
        scene = room['scene']
        xyz, _, target, _ = torch.load(room['ground_truth'], map_location='cpu', weights_only=False)
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        path = args.predictions / f'{scene}_predictions.npz'
        with np.load(path) as saved:
            prediction = saved['prediction'].copy()
            view_count = saved['view_count'].copy()
            original_cm = saved['confusion'].copy()
        if len(xyz) != len(prediction) or len(target) != len(prediction):
            raise AssertionError(f'{scene}: original prediction/GT length mismatch')
        if not np.array_equal(confusion_matrix(prediction, target), original_cm):
            raise AssertionError(f'{scene}: original confusion matrix mismatch')
        if not np.array_equal(prediction == -1, view_count == 0):
            raise AssertionError(f'{scene}: inconsistent feature-coverage status')
        original_total += original_cm
        grid, indices, counts, minimum = quantize(xyz, args.voxel_size)
        voxel_pred = prediction[indices]
        voxel_gt = target[indices]
        cm = confusion_matrix(voxel_pred, voxel_gt)
        total += cm
        output_path = args.output / f'{scene}_voxels.npz'
        np.savez_compressed(output_path, grid_coordinates=grid.astype(np.int32),
                            grid_minimum=minimum, representative_index=indices,
                            occupancy=counts, prediction=voxel_pred, target=voxel_gt,
                            view_count=view_count[indices], confusion=cm)
        # Independently recompute the confusion counts from the saved voxel output.
        with np.load(output_path) as saved:
            if not np.array_equal(confusion_matrix(saved['prediction'], saved['target']), cm):
                raise AssertionError(f'{scene}: saved voxel confusion mismatch')
        record = dict(scene=scene, original_points=len(xyz), occupied_voxels=len(indices),
                      valid_voxels=int(cm.sum()), grid_minimum=minimum.tolist(),
                      original_prediction_sha256=sha256(path),
                      voxel_output_sha256=sha256(output_path),
                      quantization_verified_for_all_points=True, saved_confusion_verified=True,
                      metrics=summarize(cm), confusion_matrix=cm.tolist())
        records.append(record)
        print(f'{len(records):02d}/68 {scene}: {len(xyz):,} points -> {len(indices):,} voxels', flush=True)
        del xyz, target, prediction, view_count, grid, indices, counts
    if not np.array_equal(original_total, np.asarray(source_summary['metrics']['confusion_matrix'])):
        raise AssertionError('Saved original-point baseline does not match source summary')
    occupied_voxels = sum(r['occupied_voxels'] for r in records)
    if occupied_voxels != int(total.sum()):
        raise AssertionError('Expected a valid semantic target for every occupied voxel')
    np.save(args.output / 'confusion_matrix.npy', total)
    np.savetxt(args.output / 'confusion_matrix.csv', total, delimiter=',', fmt='%d')
    summary = dict(complete=True, evaluated_rooms=68, expected_rooms=68,
                   voxel_size_m=args.voxel_size, evaluation='one vote per occupied voxel; no inverse mapping',
                   original_points=sum(r['original_points'] for r in records),
                   occupied_voxels=occupied_voxels, metrics=summarize(total),
                   confusion_matrix=total.tolist(),
                   confusion_orientation='GT rows, prediction columns; column 13=no feature',
                   original_point_reference=source_summary['metrics']['all_points'],
                   source_prediction_confusion_verified=True, rooms=records,
                   elapsed_seconds=time.monotonic() - start)
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('complete', 'evaluated_rooms', 'occupied_voxels', 'metrics')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
