import numpy as np


iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def get_iou(mask1, mask2):

    intersection = mask1 * mask2
    intersection_count = np.sum(intersection)

    union = mask1 + mask2
    union[union > 0] = 1
    union_count = np.sum(union)

    if union_count == 0:
        return 1
    else:
        return intersection_count * 1.0 / union_count


def get_precision(predict_mask, target_mask):
    ave_precision = 0
    iou = get_iou(predict_mask, target_mask)

    for iou_threshold in iou_thresholds:

        tp = 0
        fp = 0
        fn = 0

        if iou > iou_threshold:
            tp += 1
        else:
            if np.count_nonzero(predict_mask) != 0:
                fp += 1
            if np.count_nonzero(target_mask) != 0:
                fn += 1

        precision = tp * 1.0 / (tp + fp + fn)
        ave_precision += precision

    return ave_precision / len(iou_thresholds)


def get_ave_precision(predict_masks, target_masks):
    ave_precision = 0
    for predict_mask, target_mask in zip(predict_masks, target_masks):
        predict_mask[predict_mask > 0] = 1
        target_mask[target_mask > 0] = 1
        predict_mask = predict_mask.astype(np.int32)
        target_mask = target_mask.astype(np.int32)
        precision = get_precision(predict_mask, target_mask)
        ave_precision += precision
    return ave_precision * 1.0 / len(predict_masks)

