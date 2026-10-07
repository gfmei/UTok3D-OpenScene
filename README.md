# UTok3D: OpenScene/LSeg reproduction on S3DIS Area 5

Reproduction code for the OpenScene projection baseline in the revised UTok3D
manuscript. All S3DIS experiments use a **0.04 m voxel size**. Evaluation covers
**all 68 Area 5 rooms** and gives **40.2% mIoU / 49.5% mAcc**.

| Evaluation | Value |
| --- | ---: |
| Voxel size | 0.04 m |
| Evaluated voxel representatives | 5,340,122 |
| Class-mean mIoU | 40.16114727433952% |
| Mean class accuracy | 49.47965993714357% |
| Representatives with image features | 4,794,824 (89.788660259%) |
| Semantic classes | 13 |

The full **512-dimensional fused LSeg features**, class scores, text embeddings,
predictions and row-alignment records are published on
[Hugging Face, release v1.0](https://huggingface.co/datasets/Ryan-2026/UTok3D-OpenScene-S3DIS-Area5/tree/v1.0).
The feature package is approximately 11.23 GB; code and compact evaluation
records are in this GitHub repository.

This is a fresh, documented adaptation of OpenScene's LSeg projection and fusion
pipeline. It supplies measured replacement metrics for the manuscript's
inconsistent original 40.3% mIoU / 37.5% mAcc entry. The original predictions
were unavailable, so this release does not claim to recover that run.

## Evaluate the published artifacts on CPU

Clone the code and install Python 3.10, NumPy, PyTorch and Hugging Face Hub:

```bash
git clone https://github.com/gfmei/UTok3D-OpenScene.git
cd UTok3D-OpenScene
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.6 huggingface_hub==1.32.0
```

Obtain the unaligned S3DIS point files from the S3DIS preprocessing archive linked
in [Open3DIS DATA.md](https://github.com/VinAIResearch/Open3DIS/blob/bc2a3b3ee7161f5754ad7a7fe77202e9a66adc8e/docs/DATA.md).
Use its `preprocess_open3dis/Area_5_*.pth` files. The evaluator checks their SHA256
hashes and point ordering against the release records. Ground-truth point labels
are read from your local files.

```bash
python extractor/verify_release.py --download --anonymous \
  --gt-root /path/to/preprocess_open3dis --mode scores
```

This anonymously downloads scores and predictions into `artifacts/` and
independently recomputes voxel representatives and the global 13-class confusion
matrix. It requires neither a GPU nor the LSeg checkpoint. The full 68-room
score evaluation reproduces every audited prediction and the exact counts.

To classify the full published features using the supplied text embeddings:

```bash
python extractor/verify_release.py --download --anonymous \
  --gt-root /path/to/preprocess_open3dis --mode features
```

This downloads all 512-dimensional feature arrays and computes feature/text dot
products on CPU. Full validation found **one prediction difference among
5,340,122 representatives** due to floating-point operation order, giving
40.161155177% mIoU / 49.479665235% mAcc, which still round to **40.2% / 49.5%**.
Saved per-view-accumulated class scores preserve the original classification
arithmetic for exact reproduction. Reports are in `reports/score_validation.json`
and `reports/feature_validation.json`.

Other evaluation options:

- `--mode predictions` independently evaluates the saved predictions against local GT.
- `--release-root /path/to/release` uses a different artifact directory.
- `--room-limit 1` runs a smoke test and does not claim the full-split result.
- `--report /path/to/validation.json` saves the verification outcome.

For a quick arithmetic check using the compact reports already in this clone:

```bash
python extractor/verify_release.py --release-root . --mode counts
```

The counts mode recomputes metrics from the supplied aggregate confusion matrix;
it does not independently verify point labels.

## Evaluation protocol

Quantize the original unaligned S3DIS world coordinates with
`floor(float64(xyz) * 25)` and subtract the integer room minimum **after flooring**.
Use OpenScene's FNV64 ordering, with augmentation disabled. The first point in
original file order represents each occupied voxel; its prediction and its GT
label are selected together. Each occupied voxel contributes one vote.

There is no semantic majority vote, pooling across points, or inverse mapping
of predictions to all original points. An uncovered representative has prediction
`-1` and contributes a false negative in both metrics. The global confusion matrix
has 13 GT rows and 14 prediction columns; the last column denotes no image feature.
All 13 classes occur in the evaluated GT.

Class order: ceiling, floor, wall, beam, column, window, door, table, chair, sofa,
bookcase, board, clutter. **mIoU is the mean of the 13 semantic-class IoUs from
global confusion counts**, and mAcc is the mean of class recalls from the same
counts. This scene metric differs from ShapeNetPart's instance-averaged mIoU_I.
The upstream OpenScene evaluator normally uses `eval_all=True` to map voxel
predictions back to original points; this adapter explicitly evaluates voxel
representatives.

## Regenerate from RGB-D

The pinned environment is recorded in `reports/environment_freeze.txt`. Install
the CUDA inference dependencies and fetch the pinned upstream LSeg source:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r extractor/requirements.txt
python extractor/setup_upstream.py
```

Obtain both Stanford 2D-3D-S RGB-D archives, `area_5a_no_xyz.tar` and
`area_5b_no_xyz.tar`, from their original source. These archive names supply views
for the same **S3DIS Area 5 test split**. Open3DIS's Area 4 RGB-D example is not the
Area 5 image source. The archive MD5s are:

- `area_5a_no_xyz.tar`: `961a490072093fca726a483b74169b48`
- `area_5b_no_xyz.tar`: `85fd6de1a7c7247d8563ed5976dbd06d`

Extract RGB, depth and perspective camera poses:

```bash
python extractor/extract_data.py /path/to/area_5a_no_xyz.tar \
  --output /path/to/stanford2d3d_no_xyz --verify-md5
python extractor/extract_data.py /path/to/area_5b_no_xyz.tar \
  --output /path/to/stanford2d3d_no_xyz --verify-md5
```

Obtain the official `demo_e200.ckpt` from the
[LSeg feature-extraction repository](https://github.com/pengsongyou/lseg_feature_extraction).
Its SHA256 is
`f21aeb5532f53a5109d453d7d0d10320701ea11ece9a002291f84f95bda73af4`.

```bash
python extractor/reproduce.py prepare --gt-root /path/to/preprocess_open3dis \
  --raw-root /path/to/stanford2d3d_no_xyz --manifest /path/to/manifest.json
python extractor/reproduce.py infer --manifest /path/to/manifest.json \
  --checkpoint /path/to/demo_e200.ckpt --output /path/to/point_predictions
python extractor/evaluate_voxels.py --manifest /path/to/manifest.json \
  --predictions /path/to/point_predictions --voxel-size 0.04 \
  --output /path/to/voxel_predictions
python extractor/export_voxel_features.py --manifest /path/to/manifest.json \
  --voxels /path/to/voxel_predictions --checkpoint /path/to/demo_e200.ckpt \
  --output /path/to/regenerated_features
```

The inference stage fuses point-independent image predictions before selecting
voxel representatives for the reported evaluation. Feature export regenerates
and saves only the selected representatives' full 512-dimensional fused features,
class scores, predictions and view counts. Export verifies their predictions and
coverage against the voxel evaluation. New runs record their own source hashes;
the published reports retain the source hashes of the completed run.

The complete manifest contains 3,547 perspective views, retaining every fifth
sorted view per room. Inference uses a 320-pixel image long side, LSeg base/crop
640, a single scale with horizontal flip, prompt `a [class] in a scene`, relative
depth tolerance 0.25 and a 10-pixel border. Area 5b camera poses are registered
into the S3DIS world frame. No semantic GT enters feature generation or inference.
Checkpoint loading is strict. CUDA float16 image inference and validated SDPA
attention are recorded in the provenance; feature and score sums are float32.

## Sources and attribution

This reproduction uses [OpenScene](https://github.com/pengsongyou/openscene),
[LSeg](https://github.com/pengsongyou/lseg_feature_extraction),
[Open3DIS](https://github.com/VinAIResearch/Open3DIS), and
[Stanford 2D-3D-S](https://github.com/alexsax/2D-3D-Semantics).
Pinned source commits and the completed run's configuration are in `reports/`.
Original RGB-D scans, point coordinates, point-level GT and model checkpoints
must be obtained from their upstream sources under their source terms.
Cite OpenScene (Peng et al., CVPR 2023), LSeg (Li et al., ICLR 2022), S3DIS
(Armeni et al., CVPR 2016) and Stanford 2D-3D-S (Armeni et al., 2017) when using
this reproduction.
