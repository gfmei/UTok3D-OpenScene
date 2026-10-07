"""S3DIS metrics from one confusion matrix, retaining uncovered points as misses."""
import numpy as np

CLASSES = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
           'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')


def confusion_matrix(pred, target):
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError('Prediction and target shapes differ')
    valid = target != 255
    if np.any((target[valid] < 0) | (target[valid] >= 13)):
        raise ValueError('Invalid semantic label')
    if np.any((pred[valid] < -1) | (pred[valid] >= 13)):
        raise ValueError('Invalid prediction; use -1 for uncovered points')
    pred = np.where(pred == -1, 13, pred)
    # Rows are ground truth, columns predictions; last column is no feature.
    return np.bincount(target[valid] * 14 + pred[valid],
                       minlength=13*14).reshape(13, 14)


def summarize(cm):
    cm = np.asarray(cm, dtype=np.int64)
    if cm.shape != (13, 14) or np.any(cm < 0):
        raise ValueError('Expected a nonnegative 13 by 14 confusion matrix')
    tp = np.diag(cm[:, :13]).astype(np.float64)
    gt = cm.sum(axis=1)
    predicted = cm[:, :13].sum(axis=0)
    union = gt + predicted - tp
    iou = np.divide(tp, union, out=np.zeros(13), where=union > 0)
    acc = np.divide(tp, gt, out=np.zeros(13), where=gt > 0)
    if np.any(iou > acc + 1e-12):
        raise AssertionError('IoU exceeds accuracy on the shared class set')
    return dict(mIoU_percent=float(100*iou.mean()), mAcc_percent=float(100*acc.mean()),
                overall_accuracy_percent=float(100*tp.sum()/max(gt.sum(), 1)),
                points=int(gt.sum()), covered_points=int(cm[:, :13].sum()),
                coverage_percent=float(100*cm[:, :13].sum()/max(gt.sum(), 1)),
                classes=list(CLASSES), class_IoU_percent=(100*iou).tolist(),
                class_accuracy_percent=(100*acc).tolist(), class_points=gt.tolist(),
                absent_ground_truth_classes=[CLASSES[i] for i in np.flatnonzero(gt == 0)],
                absent_class_policy='fixed 13-class mean; zero when denominator is zero')


def audit(cm):
    all_points = summarize(cm)
    covered = cm.copy()
    covered[:, 13] = 0
    covered_points = summarize(covered)
    # Reproduce OpenScene util/metric.py's mixed denominators for diagnosis only.
    legacy = dict(mIoU_percent=covered_points['mIoU_percent'],
                  mAcc_percent=all_points['mAcc_percent'],
                  warning='IoU excludes uncovered points while accuracy includes them')
    return dict(all_points=all_points, covered_points=covered_points,
                legacy_mixed_denominators=legacy, confusion_matrix=cm.tolist(),
                confusion_orientation='ground-truth rows, prediction columns; column 13=no feature')
