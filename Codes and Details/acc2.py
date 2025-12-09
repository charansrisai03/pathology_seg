import zipfile
import os
import cv2
import numpy as np
from tqdm import tqdm
import tempfile
import shutil

def compute_metrics(gt, pred):
    """Compute Pixel Accuracy, IoU, and Dice between two binary masks."""
    # Ensure same size
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    # Convert to binary (0 or 1)
    _, gt_bin = cv2.threshold(gt, 127, 1, cv2.THRESH_BINARY)
    _, pred_bin = cv2.threshold(pred, 127, 1, cv2.THRESH_BINARY)

    intersection = np.logical_and(gt_bin, pred_bin)
    union = np.logical_or(gt_bin, pred_bin)

    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    dice = 2 * np.sum(intersection) / (np.sum(gt_bin) + np.sum(pred_bin) + 1e-6)
    accuracy = np.sum(gt_bin == pred_bin) / gt_bin.size
    return accuracy, iou, dice


def evaluate_from_zip(gt_zip, pred_zip):
    """Evaluate metrics for all _final images in GT and predicted zips (same order)."""
    temp_dir = tempfile.mkdtemp()

    gt_dir = os.path.join(temp_dir, "gt")
    pred_dir = os.path.join(temp_dir, "pred")

    # Extract both zips
    with zipfile.ZipFile(gt_zip, 'r') as zip_ref:
        zip_ref.extractall(gt_dir)
    with zipfile.ZipFile(pred_zip, 'r') as zip_ref:
        zip_ref.extractall(pred_dir)

    # Extract image directories
    gt_image_dir = os.path.join(gt_dir)
    pred_image_dir = os.path.join(pred_dir,'output_unetpp')

    # ✅ Filter only "_final" images from GT
    gt_files = [os.path.join(gt_image_dir, f) for f in os.listdir(gt_image_dir)
                if f.lower().endswith(('_final.png', '_final.jpg', '_final.jpeg'))]

    # ✅ Get all predicted images
    pred_files = [os.path.join(pred_image_dir, f) for f in os.listdir(pred_image_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Pair sequentially (in same order, not sorted)
    count = min(len(gt_files), len(pred_files))
    if count == 0:
        print("❌ No valid _final image pairs found.")
        shutil.rmtree(temp_dir)
        return

    total_acc, total_iou, total_dice = 0, 0, 0

    print(f"\nEvaluating {count} _final image pairs sequentially...\n")

    for i in tqdm(range(count)):
        gt = cv2.imread(gt_files[i], cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_files[i], cv2.IMREAD_GRAYSCALE)

        if gt is None or pred is None:
            print(f"⚠️ Error reading image pair index {i}")
            continue

        acc, iou, dice = compute_metrics(gt, pred)
        total_acc += acc
        total_iou += iou
        total_dice += dice

    print("\n✅ Final Average Metrics:")
    print(f"Pixel Accuracy: {(total_acc / count) * 100:.2f}%")
    print(f"IoU (Jaccard Index): {total_iou / count:.4f}")
    print(f"Dice Coefficient: {total_dice / count:.4f}")

    shutil.rmtree(temp_dir)


# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    gt_zip = "output_images.zip"             # Ground truth zip (contains overlay + final)
    pred_zip = "output_unetpp.zip"   # Predicted zip (final only)
    evaluate_from_zip(gt_zip, pred_zip)
