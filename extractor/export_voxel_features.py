"""Generate full LSeg embeddings for the published 0.04 m Area 5 representatives.

Keep original point chunks and projection calculations to match saved inference.
Only representative feature sums are retained. No GT labels enter inference.
Release files omit raw RGB-D, point coordinates, and point-level GT labels.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from reproduce import atomic_json, checksum

ROOT = Path(__file__).resolve().parent


def export_room(room, extractor, args):
    start = time.monotonic()
    scene = room['scene']
    voxel_file = args.voxels / f'{scene}_voxels.npz'
    with np.load(voxel_file) as data:
        indices = data['representative_index'].copy()
        expected_pred = data['prediction'].copy()
        expected_count = data['view_count'].copy()
    xyz = np.asarray(torch.load(room['ground_truth'], map_location='cpu', weights_only=False)[0], dtype=np.float32)
    n, nv = len(xyz), len(indices)
    coords = torch.from_numpy(xyz).cuda()
    lookup = torch.full((n,), -1, dtype=torch.int32, device='cuda')
    lookup[torch.from_numpy(indices).cuda()] = torch.arange(nv, dtype=torch.int32, device='cuda')
    feature_sum = torch.zeros((nv, 512), dtype=torch.float32, device='cuda')
    score_sum = torch.zeros((nv, 13), dtype=torch.float32, device='cuda')
    counts = torch.zeros(nv, dtype=torch.int32, device='cuda')
    for j, view in enumerate(room['views']):
        rgb = np.asarray(Image.open(view['rgb']).convert('RGB'))
        raw_depth = np.asarray(Image.open(view['depth']))
        features, (h, w) = extractor(rgb)
        features = features.half()
        native_h, native_w = raw_depth.shape
        depth = raw_depth.astype(np.float32) / 512.0
        depth[(raw_depth == 65535) | (raw_depth == 0)] = 0
        depth = np.asarray(Image.fromarray(depth).resize((w, h), Image.Resampling.NEAREST)).copy()
        depth_gpu = torch.from_numpy(depth).cuda()
        k = np.asarray(view['intrinsic'], dtype=np.float64).copy()
        sx, sy = w / native_w, h / native_h
        k[0, 0] *= sx; k[1, 1] *= sy
        k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
        k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
        rt = torch.tensor(view['world_to_camera'], dtype=torch.float32, device='cuda')
        for lo in range(0, n, args.point_chunk):
            hi = min(n, lo + args.point_chunk)
            camera = coords[lo:hi] @ rt[:3, :3].T + rt[:3, 3]
            z = camera[:, 2]
            u = torch.round(k[0, 0] * camera[:, 0] / z + k[0, 2]).long()
            v = torch.round(k[1, 1] * camera[:, 1] / z + k[1, 2]).long()
            inside = (z > 0) & torch.isfinite(camera).all(1) & (u >= 10) & (v >= 10) & (u < w - 10) & (v < h - 10)
            ids = torch.nonzero(inside).flatten()
            if not ids.numel():
                continue
            d = depth_gpu[v[ids], u[ids]]
            ids = ids[(d > 0) & ((d - z[ids]).abs() <= 0.25 * d)]
            if not ids.numel():
                continue
            values = features[:, v[ids], u[ids]].T.float()
            # Preserve the original score GEMM batch shape before selecting representatives.
            point_scores = values @ extractor.text.T
            local = lookup[ids + lo].long()
            selected = local >= 0
            representatives = local[selected]
            feature_sum[representatives] += values[selected]
            score_sum[representatives] += point_scores[selected]
            counts[representatives] += 1
        if (j + 1) % 25 == 0 or j + 1 == len(room['views']):
            print(f'{scene}: views {j+1}/{len(room["views"])}; elapsed {time.monotonic()-start:.0f}s', flush=True)
    count_cpu = counts.cpu().numpy()
    prediction = score_sum.argmax(1)
    prediction[counts == 0] = -1
    pred_cpu = prediction.cpu().numpy().astype(np.int16)
    if not np.array_equal(count_cpu, expected_count):
        raise AssertionError(f'{scene}: feature visibility differs from audited run')
    if not np.array_equal(pred_cpu, expected_pred):
        mismatch = int((pred_cpu != expected_pred).sum())
        raise AssertionError(f'{scene}: {mismatch} regenerated predictions differ from audited run')
    denominator = counts.clamp_min(1)[:, None]
    fused = (feature_sum / denominator).cpu().numpy()
    scores = (score_sum / denominator).cpu().numpy()
    # CPU feature reclassification is a secondary numerical check; the release
    # retains per-view-accumulated FP32 scores for exact original classification.
    feature_scores = fused @ extractor.text.cpu().numpy().T
    feature_prediction = feature_scores.argmax(1).astype(np.int16)
    feature_prediction[count_cpu == 0] = -1
    feature_mismatch = int((feature_prediction != expected_pred).sum())
    if not np.isfinite(fused).all() or not np.isfinite(scores).all():
        raise AssertionError(f'{scene}: nonfinite released features or scores')
    for folder in ('features', 'scores', 'predictions', 'room_records'):
        (args.output / folder).mkdir(exist_ok=True, parents=True)
    feature_path = args.output / 'features' / f'{scene}.npy'
    score_path = args.output / 'scores' / f'{scene}.npy'
    prediction_path = args.output / 'predictions' / f'{scene}.npz'
    np.save(feature_path, fused)
    np.save(score_path, scores)
    np.savez_compressed(prediction_path, representative_index=indices,
                        prediction=pred_cpu, view_count=count_cpu)
    record = dict(scene=scene, original_points=n, representatives=nv,
                  features_shape=list(fused.shape), features_dtype=str(fused.dtype),
                  scores_dtype=str(scores.dtype), views=len(room['views']),
                  predictions_match_audited_run=True, view_counts_match_audited_run=True,
                  feature_dot_product_prediction_mismatches=feature_mismatch,
                  max_feature_score_absolute_error=float(np.abs(feature_scores - scores).max()),
                  source_ground_truth_file_sha256=checksum(Path(room['ground_truth'])),
                  source_voxel_file_sha256=checksum(voxel_file),
                  feature_sha256=checksum(feature_path), score_sha256=checksum(score_path),
                  prediction_sha256=checksum(prediction_path), seconds=time.monotonic()-start)
    atomic_json(args.output / 'room_records' / f'{scene}.json', record)
    print(f'FEATURE ROOM COMPLETE {scene}: {nv:,} representatives; exact score predictions verified; '
          f'feature-dot discrepancies {feature_mismatch}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT.parent / 'area5_manifest.json')
    p.add_argument('--voxels', type=Path, default=ROOT.parent / 'results/area5_lseg_voxel_004')
    p.add_argument('--checkpoint', type=Path, default=ROOT.parent / 'checkpoints/demo_e200.ckpt')
    p.add_argument('--output', type=Path, default=ROOT.parent / 'hf_release/UTok3D-OpenScene-S3DIS-Area5')
    p.add_argument('--point-chunk', type=int, default=65536)
    p.add_argument('--room-limit', type=int)
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1457)
    manifest = json.loads(a.manifest.read_text())
    if len(manifest['rooms']) != 68:
        raise ValueError('Expected all 68 Area 5 rooms')
    config = dict(manifest_sha256=checksum(a.manifest), checkpoint_sha256=checksum(a.checkpoint),
                  exporter_sha256=checksum(Path(__file__)), model_loader_sha256=checksum(ROOT/'lseg_model.py'),
                  source_voxel_config_sha256=checksum(a.voxels/'config.json'), voxel_size_m=0.04,
                  point_chunk=a.point_chunk, long_side=320, feature_dimension=512,
                  feature_dtype='float32', score_dtype='float32', torch_version=torch.__version__,
                  seed=1457, no_gt_in_inference=True, no_raw_data_or_point_gt_in_release=True)
    a.output.mkdir(parents=True, exist_ok=True)
    config_path = a.output/'feature_generation_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Feature output contains a different configuration')
    atomic_json(config_path, config)
    from lseg_model import Extractor
    extractor = Extractor(a.checkpoint, long_side=320, prompt=True, flip=True)
    np.save(a.output/'text_embeddings.npy', extractor.text.cpu().numpy())
    todo = manifest['rooms'][:a.room_limit] if a.room_limit else manifest['rooms']
    with torch.inference_mode():
        for room in todo:
            if (a.output/'room_records'/f'{room["scene"]}.json').exists():
                print('RESUME skip', room['scene'], flush=True)
                continue
            export_room(room, extractor, a)
    records = [json.loads(p.read_text()) for p in sorted((a.output/'room_records').glob('Area_5_*.json'))]
    atomic_json(a.output/'feature_generation_summary.json', dict(
        complete=len(records)==68, expected_rooms=68, completed_rooms=len(records),
        representatives=sum(r['representatives'] for r in records),
        exact_predictions_match=all(r['predictions_match_audited_run'] for r in records),
        view_counts_match=all(r['view_counts_match_audited_run'] for r in records),
        feature_dot_product_prediction_mismatches=sum(r['feature_dot_product_prediction_mismatches'] for r in records),
        feature_bytes=sum(p.stat().st_size for p in (a.output/'features').glob('*.npy')),
        rooms=records))


if __name__ == '__main__':
    main()
