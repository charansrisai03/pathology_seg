import cv2
import numpy as np
import sys

def compute_metrics(gt, pred):
    """Compute Pixel Accuracy, IoU, and Dice between two binary masks."""
    # Ensure same size
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # Convert to binary (0 or 1)
    _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)

    # Compute intersection, union
    intersection = np.logical_and(gt_bin, pred_bin)
    union = np.logical_or(gt_bin, pred_bin)

    # Metrics
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    dice = 2 * np.sum(intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-6)
    accuracy = np.sum(gt_bin == pred_bin) / gt_bin.size

    return accuracy, iou, dice


if __name__ == "__main__":

    gt_path = "gt (3).png"
    pred_path = "prd (3).png"

    # Read grayscale images
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    if gt is None or pred is None:
        print("❌ Error reading one of the images. Check file paths.")
        sys.exit(1)

    # Compute metrics
    acc, iou, dice = compute_metrics(gt, pred)

    # Print results
    print("\n✅ Metrics for given image pair:")
    print(f"Pixel Accuracy: {acc * 100:.2f}%")
    print(f"IoU (Jaccard Index): {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")
