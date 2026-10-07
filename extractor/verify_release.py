"""Reproduce the released 0.04 m Area 5 metrics on CPU using local S3DIS GT."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_voxels import quantize
from metrics import confusion_matrix, summarize

REPO_ID = 'Ryan-2026/UTok3D-OpenScene-S3DIS-Area5'


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', type=Path, default=Path(__file__).resolve().parent.parent/'artifacts')
    p.add_argument('--gt-root', type=Path, help='Open3DIS preprocess_open3dis directory with Area_5_*.pth')
    p.add_argument('--mode', choices=['scores', 'features', 'predictions', 'counts'], default='scores')
    p.add_argument('--download', action='store_true', help='Download required artifacts from Hugging Face')
    p.add_argument('--repo-id', default=REPO_ID)
    p.add_argument('--revision', default='v1.0')
    p.add_argument('--anonymous', action='store_true', help='Do not send a Hugging Face access token')
    p.add_argument('--room-limit', type=int, help='Smoke test only; omits the global result assertion')
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    if a.download:
        from huggingface_hub import snapshot_download
        patterns = ['reports/primary_summary.json', 'reports/confusion_matrix.csv',
                    'predictions/*', 'room_records/*',
                    'text_embeddings.npy', 'feature_generation*.json', 'README.md']
        if a.mode in ('scores', 'features'):
            patterns += [f'{a.mode}/*']
        snapshot_download(a.repo_id, repo_type='dataset', revision=a.revision,
                          local_dir=a.release_root, allow_patterns=patterns,
                          token=False if a.anonymous else None)
    expected = json.loads((a.release_root/'reports/primary_summary.json').read_text())
    if not expected['complete'] or expected['evaluated_rooms'] != 68:
        raise ValueError('Released primary evaluation is incomplete')
    cm = np.zeros((13, 14), dtype=np.int64)
    mismatch_total = 0
    if a.mode == 'counts':
        cm = np.loadtxt(a.release_root/'reports/confusion_matrix.csv', delimiter=',', dtype=np.int64)
    else:
        if a.gt_root is None:
            p.error('--gt-root is required for independent point-label evaluation')
        text = np.load(a.release_root/'text_embeddings.npy') if a.mode == 'features' else None
        rooms = expected['rooms'][:a.room_limit] if a.room_limit else expected['rooms']
        for room in rooms:
            scene = room['scene']
            record = json.loads((a.release_root/'room_records'/f'{scene}.json').read_text())
            gt_path = a.gt_root/f'{scene}.pth'
            if checksum(gt_path) != record['source_ground_truth_file_sha256']:
                raise ValueError(f'{scene}: use the exact unaligned Open3DIS point file; hash differs')
            xyz, _, gt, _ = torch.load(gt_path, map_location='cpu', weights_only=False)
            target = np.asarray(gt, dtype=np.int64)
            path = a.release_root/'predictions'/f'{scene}.npz'
            if checksum(path) != record['prediction_sha256']:
                raise ValueError(f'{scene}: prediction artifact checksum mismatch')
            with np.load(path) as data:
                indices = data['representative_index'].copy()
                reference = data['prediction'].copy()
                view_count = data['view_count'].copy()
            _, recomputed_indices, _, _ = quantize(xyz, 0.04)
            if not np.array_equal(indices, recomputed_indices):
                raise ValueError(f'{scene}: voxel representatives differ from released ordering')
            if a.mode == 'predictions':
                prediction = reference
            else:
                array_path = a.release_root/a.mode/f'{scene}.npy'
                key = 'feature_sha256' if a.mode == 'features' else 'score_sha256'
                if checksum(array_path) != record[key]:
                    raise ValueError(f'{scene}: {a.mode} artifact checksum mismatch')
                values = np.load(array_path, mmap_mode='r')
                if len(values) != len(indices):
                    raise ValueError(f'{scene}: feature/score row count mismatch')
                prediction = np.empty(len(indices), dtype=np.int16)
                for lo in range(0, len(indices), 65536):
                    block = np.asarray(values[lo:lo+65536])
                    scores = block @ text.T if a.mode == 'features' else block
                    prediction[lo:lo+65536] = scores.argmax(axis=1)
                prediction[view_count == 0] = -1
            mismatch = int((prediction != reference).sum())
            mismatch_total += mismatch
            if a.mode != 'features' and mismatch:
                raise AssertionError(f'{scene}: released scores fail exact prediction reproduction')
            room_cm = confusion_matrix(prediction, target[indices])
            if a.mode != 'features' and not np.array_equal(room_cm, np.asarray(room['confusion_matrix'])):
                raise AssertionError(f'{scene}: confusion counts differ from published audit')
            cm += room_cm
            print(f'{scene}: {len(indices):,} representatives; prediction differences {mismatch}', flush=True)
    result = summarize(cm)
    full = a.mode == 'counts' or not a.room_limit
    if full:
        for metric in ('mIoU_percent', 'mAcc_percent'):
            if round(result[metric], 1) != round(expected['metrics'][metric], 1):
                raise AssertionError(f'{metric}: rounded result differs from Table II')
        if a.mode != 'features' and not np.array_equal(cm, np.asarray(expected['confusion_matrix'])):
            raise AssertionError('Global confusion counts differ from the recorded run')
    report = dict(mode=a.mode, full_68_room_evaluation=full, metrics=result,
                  prediction_differences=mismatch_total,
                  exact_global_confusion_match=bool(np.array_equal(cm, np.asarray(expected['confusion_matrix']))))
    print(json.dumps(report, indent=2), flush=True)
    if a.report:
        a.report.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
