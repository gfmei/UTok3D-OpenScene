"""OpenScene LSeg fusion on all 68 S3DIS Area 5 rooms, with auditable metrics.

Data: unaligned Open3DIS room point clouds and Stanford 2D-3D-S RGB/depth/poses.
No model training, superpoint smoothing, ground-truth-based view selection or label alignment.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from metrics import CLASSES, audit, confusion_matrix

ROOT = Path(__file__).resolve().parent
# Pointcept's Stanford2D3D-to-S3DIS Area 5b registration transform.
AREA5B_TO_S3DIS = np.array([[0, 1, 0, -4.09703582], [-1, 0, 0, 6.22617759],
                          [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
# Stanford's perspective JSON already stores OpenCV world-to-camera RT.
CAMERA_AXIS_CONVERSION = np.eye(4)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def prepare(args):
    views = defaultdict(list)
    for area in ('5a', '5b'):
        base = args.raw_root / f'area_{area}' / 'data'
        index=ROOT/f'pose_index_{area}.json'
        if index.is_file():
            cached=json.loads(index.read_text())
            entries=sorted([(base/'pose'/Path(r['archive_path']).name,r['metadata'])
                            for r in cached['records']],key=lambda r:str(r[0]))
        else:
            entries=[(p,None) for p in sorted((base/'pose').glob('*.json'))]
        paths=[p for p,_ in entries]
        if not paths:
            raise FileNotFoundError(f'No perspective camera poses in {base / "pose"}')
        expected={'5a':6261,'5b':11332}[area]
        if len(paths)!=expected:
            raise ValueError(f'Incomplete camera manifest for {area}: {len(paths)} of {expected}')
        for p,meta in entries:
            if meta is None:
                meta = json.loads(p.read_text())
            room = meta['room'].rsplit('_', 1)[0]
            rgb = base/'rgb'/p.name.replace('pose.json', 'rgb.png')
            depth = base/'depth'/p.name.replace('pose.json', 'depth.png')
            if not args.defer_images and (not rgb.is_file() or not depth.is_file()):
                raise FileNotFoundError(f'Incomplete RGB-D view: {p}')
            rt = np.eye(4)
            rt[:3] = np.asarray(meta['camera_rt_matrix'], dtype=np.float64)
            # Perspective JSON is OpenCV RT in each scan area's world frame.
            # S3DIS Area 5 uses the Area 5a frame; register Area 5b first.
            if area == '5b':
                rt = rt @ np.linalg.inv(AREA5B_TO_S3DIS)
            views[f'Area_5_{room}'].append(dict(rgb=str(rgb), depth=str(depth), pose=str(p),
                intrinsic=meta['camera_k_matrix'], world_to_camera=rt.tolist(), source_area=area))
    rooms = []
    gt_paths = sorted(args.gt_root.glob('Area_5_*.pth'))
    if len(gt_paths) != 68:
        raise ValueError(f'Expected 68 Area 5 rooms, found {len(gt_paths)}')
    for p in gt_paths:
        room_views = sorted(views[p.stem], key=lambda v:(v['source_area'], v['rgb']))
        if not room_views:
            raise FileNotFoundError(f'No RGB-D views for {p.stem}')
        rooms.append(dict(scene=p.stem, ground_truth=str(p), all_view_count=len(room_views),
                          views=room_views[::args.frame_stride]))
    manifest = dict(area=5, classes=list(CLASSES), gt_root=str(args.gt_root),
        raw_root=str(args.raw_root), frame_stride=args.frame_stride, rooms=rooms,
        depth_scale=512.0, depth_invalid=65535, camera_convention='OpenCV world-to-camera',
        camera_axis_conversion=CAMERA_AXIS_CONVERSION.tolist(),
        area5b_registration='RT_s3dis = RT_5b @ inverse(T_5b_to_s3dis)',
        area5b_to_s3dis=AREA5B_TO_S3DIS.tolist(),
        view_selection='sorted perspective views within each room, every frame_stride-th view')
    atomic_json(args.manifest, manifest)
    print('MANIFEST',len(rooms),'rooms',sum(len(r['views']) for r in rooms),'selected views',flush=True)


def fuse_room(room, extractor, args):
    start = time.monotonic()
    scene = room['scene']
    xyz, _, target, _ = torch.load(room['ground_truth'], map_location='cpu', weights_only=False)
    xyz = np.asarray(xyz, dtype=np.float32)
    target = np.asarray(target, dtype=np.int64)
    n = len(xyz)
    coords = torch.from_numpy(xyz).cuda()
    score_sum = torch.zeros((n, 13), dtype=torch.float32, device='cuda')
    counts = torch.zeros(n, dtype=torch.int32, device='cuda')
    scratch = args.output/'scratch'/f'{scene}_sum.npy'
    feature_sum = None
    if args.save_features:
        scratch.parent.mkdir(parents=True, exist_ok=True)
        feature_sum = np.lib.format.open_memmap(scratch, mode='w+', dtype=np.float32, shape=(n,512))
        feature_sum[:] = 0
    view_stats = []
    for j, view in enumerate(room['views']):
        rgb = np.asarray(Image.open(view['rgb']).convert('RGB'))
        raw_depth = np.asarray(Image.open(view['depth']))
        if rgb.shape[:2] != raw_depth.shape or raw_depth.ndim != 2:
            raise ValueError(f'RGB and depth dimensions differ: {view["rgb"]}')
        feats, (h,w) = extractor(rgb)
        # Official OpenScene persists each image's features in float16.
        feats = feats.half()
        native_h, native_w = raw_depth.shape
        depth = raw_depth.astype(np.float32)/512.0
        depth[(raw_depth == 65535) | (raw_depth == 0)] = 0
        depth = np.asarray(Image.fromarray(depth).resize((w,h), Image.Resampling.NEAREST)).copy()
        depth_gpu = torch.from_numpy(depth).cuda()
        k = np.asarray(view['intrinsic'], dtype=np.float64).copy()
        sx, sy = w/native_w, h/native_h
        k[0,0] *= sx; k[1,1] *= sy
        k[0,2] = (k[0,2]+0.5)*sx-0.5
        k[1,2] = (k[1,2]+0.5)*sy-0.5
        rt = torch.tensor(view['world_to_camera'], dtype=torch.float32, device='cuda')
        visible_count = 0
        residual_sum = 0.
        for lo in range(0,n,args.point_chunk):
            hi = min(n,lo+args.point_chunk)
            camera = coords[lo:hi] @ rt[:3,:3].T + rt[:3,3]
            z = camera[:,2]
            u = torch.round(k[0,0]*camera[:,0]/z + k[0,2]).long()
            v = torch.round(k[1,1]*camera[:,1]/z + k[1,2]).long()
            inside = (z>0) & torch.isfinite(camera).all(1) & (u>=args.cut_bound) & (v>=args.cut_bound) \
                & (u<w-args.cut_bound) & (v<h-args.cut_bound)
            ids = torch.nonzero(inside).flatten()
            if not ids.numel():
                continue
            d = depth_gpu[v[ids],u[ids]]
            residual = (d-z[ids]).abs()
            ids = ids[(d>0) & (residual <= args.visibility_threshold*d)]
            if not ids.numel():
                continue
            values = feats[:,v[ids],u[ids]].T.float()
            global_ids = ids + lo
            score_sum[global_ids] += values @ extractor.text.T
            counts[global_ids] += 1
            if feature_sum is not None:
                feature_sum[global_ids.cpu().numpy()] += values.cpu().numpy()
            visible_count += ids.numel()
            residual_sum += (depth_gpu[v[ids],u[ids]]-z[ids]).abs().sum().item()
        view_stats.append(dict(rgb=view['rgb'], visible_points=visible_count,
                               mean_depth_residual_m=residual_sum/max(visible_count,1)))
        if (j+1)%10 == 0 or j+1==len(room['views']):
            print(f'{scene}: views {j+1}/{len(room["views"])}; '
                  f'coverage {100*(counts>0).float().mean().item():.1f}%; '
                  f'elapsed {time.monotonic()-start:.0f}s',flush=True)
    covered = counts>0
    if not covered.any():
        raise ValueError(f'No point is visible in {scene}; check camera registration')
    prediction = score_sum.argmax(1)
    prediction[~covered] = -1
    pred_cpu = prediction.cpu().numpy().astype(np.int16)
    count_cpu = counts.cpu().numpy()
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output/f'{scene}_predictions.npz',prediction=pred_cpu,
                        view_count=count_cpu, confusion=confusion_matrix(pred_cpu,target))
    # The dot product of mean features and text is the mean of these dot products.
    scores = (score_sum/counts.clamp_min(1)[:,None]).cpu().numpy().astype(np.float16)
    np.save(args.output/f'{scene}_scores.npy', scores)
    if feature_sum is not None:
        path = args.output/f'{scene}_features.npy'
        fused = np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(n,512))
        for lo in range(0,n,args.point_chunk):
            hi=min(n,lo+args.point_chunk)
            fused[lo:hi] = feature_sum[lo:hi]/np.maximum(count_cpu[lo:hi,None],1)
        fused.flush()
        del fused,feature_sum
        scratch.unlink()
    result = dict(scene=scene, views=len(room['views']), seconds=time.monotonic()-start,
                  metrics=audit(confusion_matrix(pred_cpu,target)),view_stats=view_stats)
    atomic_json(args.output/f'{scene}_metrics.json',result)
    print('ROOM COMPLETE',scene,json.dumps(result['metrics']['all_points']),flush=True)


def evaluate(args, manifest):
    cm=np.zeros((13,14),dtype=np.int64)
    rooms=[]
    missing=[]
    for room in manifest['rooms']:
        p=args.output/f'{room["scene"]}_predictions.npz'
        if not p.exists():
            missing.append(room['scene']);continue
        with np.load(p) as a:
            target=torch.load(room['ground_truth'],map_location='cpu',weights_only=False)[2]
            room_cm=confusion_matrix(a['prediction'],target)
            if not np.array_equal(room_cm,a['confusion']):
                raise ValueError(f'Saved confusion does not match predictions: {p}')
            cm+=room_cm
            rooms.append(dict(scene=room['scene'],metrics=audit(room_cm)))
    result=dict(complete=len(rooms)==68 and not missing, evaluated_rooms=len(rooms),
                expected_rooms=68,missing_rooms=missing,metrics=audit(cm),rooms=rooms)
    np.save(args.output/'confusion_matrix.npy',cm)
    atomic_json(args.output/'summary.json',result)
    print('EVALUATION',json.dumps({k:v for k,v in result.items() if k!='rooms'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','infer','evaluate'])
    p.add_argument('--raw-root',type=Path,default=ROOT.parent/'data/stanford2d3d_no_xyz')
    p.add_argument('--gt-root',type=Path,default=ROOT.parent/'data/preprocess_open3dis')
    p.add_argument('--manifest',type=Path,default=ROOT.parent/'area5_manifest.json')
    p.add_argument('--output',type=Path,default=ROOT.parent/'results/area5_lseg')
    p.add_argument('--checkpoint',type=Path,default=ROOT.parent/'checkpoints/demo_e200.ckpt')
    p.add_argument('--frame-stride',type=int,default=5)
    p.add_argument('--long-side',type=int,default=320)
    p.add_argument('--point-chunk',type=int,default=65536)
    p.add_argument('--visibility-threshold',type=float,default=0.25)
    p.add_argument('--cut-bound',type=int,default=10)
    p.add_argument('--save-features',action='store_true')
    p.add_argument('--no-prompt',action='store_true')
    p.add_argument('--no-flip',action='store_true')
    p.add_argument('--room-limit',type=int)
    p.add_argument('--defer-images',action='store_true',
                   help='Build the complete camera manifest while image extraction finishes')
    p.add_argument('--scenes',nargs='+',help='Run selected scenes from the complete manifest')
    a=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1457)
    if a.stage=='prepare':
        prepare(a);return
    manifest=json.loads(a.manifest.read_text())
    if a.stage=='evaluate':
        evaluate(a,manifest);return
    from lseg_model import Extractor
    config=dict(manifest_sha256=checksum(a.manifest),checkpoint_sha256=checksum(a.checkpoint),
        adapter_source_sha256=checksum(Path(__file__)),model_loader_sha256=checksum(ROOT/'lseg_model.py'),
        long_side=a.long_side, frame_stride=manifest['frame_stride'],
        visibility_threshold=a.visibility_threshold,cut_bound=a.cut_bound,
        prompt=not a.no_prompt,flip=not a.no_flip,save_features=a.save_features,
        torch_version=torch.__version__,mixed_precision='float16 CUDA autocast',
        attention_kernel='torch scaled_dot_product_attention; validated against upstream',seed=1457,
        all_points=True,zero_feature_prediction=-1,classes=list(CLASSES))
    config_path=a.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Output directory contains a different run configuration')
    atomic_json(config_path,config)
    extractor=Extractor(a.checkpoint,a.long_side,not a.no_prompt,not a.no_flip)
    todo=manifest['rooms'][:a.room_limit] if a.room_limit else manifest['rooms']
    if a.scenes:
        unknown=set(a.scenes)-{r['scene'] for r in manifest['rooms']}
        if unknown:
            raise ValueError(f'Unknown scenes: {unknown}')
        todo=[r for r in todo if r['scene'] in a.scenes]
    for room in todo:
        if (a.output/f'{room["scene"]}_metrics.json').exists():
            print('RESUME skip',room['scene'],flush=True);continue
        fuse_room(room,extractor,a)
    evaluate(a,manifest)


if __name__=='__main__':
    main()
