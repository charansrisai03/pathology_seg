import os
import zipfile
import numpy as np
import cv2
import torch
import imageio
from tqdm import tqdm
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects, closing, disk

from stardist.models import StarDist2D
from csbdeep.utils import normalize

# ---------- SAM IMPORT ----------
from segment_anything import SamPredictor, sam_model_registry


# ---------- CONFIG ----------
INPUT_ROOT   = "/home/llmPathoUser/pathologyStudentsAug25/pathologyStudentsAug25/necrosisData"       # contains necrosis.zip and non_necrosis.zip
EXTRACT_ROOT = "unzipped_data"      # folder to extract zip files
OUTPUT_ROOT  = "segmentation_output"

os.makedirs(EXTRACT_ROOT, exist_ok=True)
os.makedirs(OUTPUT_ROOT, exist_ok=True)

PROB_THRESH   = 0.35
NMS_THRESH    = 0.4
MIN_OBJ_SIZE  = 40

SAM_WEIGHTS   = "sam_vit_b_01ec64.pth"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- LOAD MODELS ----------
print("Loading StarDist...")
stardist = StarDist2D.from_pretrained("2D_versatile_fluo")

print("Loading SAM...")
sam = sam_model_registry["vit_b"](checkpoint=SAM_WEIGHTS)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# ---------- ZIP EXTRACTION ----------
def extract_zip_files():
    for file in os.listdir(INPUT_ROOT):
        if file.endswith(".zip"):
            zip_path = os.path.join(INPUT_ROOT, file)
            extract_folder = os.path.join(EXTRACT_ROOT, file.replace(".zip", ""))

            print(f"Extracting {zip_path} → {extract_folder}")
            os.makedirs(extract_folder, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)


# ---------- SAM MASK ----------
def run_sam(image):
    sam_predictor.set_image(image)
    
    h, w = image.shape[:2]
    points = np.array([[w//2, h//2]])
    labels = np.array([1])

    masks, _, _ = sam_predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True
    )
    
    return masks.max(0)


# ---------- StarDist INSTANCE ----------
def run_stardist(img):
    img_norm = normalize(img, 1, 99.8)

    labels, _ = stardist.predict_instances(
        img_norm,
        prob_thresh=PROB_THRESH,
        nms_thresh=NMS_THRESH
    )
    return labels


# ---------- FUSION ----------
def fuse_segments(image):
    sam_mask = run_sam(image)
    stardist_labels = run_stardist(image)

    markers = stardist_labels.astype(np.int32)
    dist = ndi.distance_transform_edt(sam_mask > 0)

    refined = watershed(-dist, markers, mask=sam_mask > 0)

    refined = remove_small_objects(refined, MIN_OBJ_SIZE)
    refined = closing(refined, disk(2))

    return refined.astype(np.uint16)


# ---------- RECURSIVE PROCESSING ----------
def process_all_images(root_dir):

    for subdir, _, files in os.walk(root_dir):

        # Create matching output folder
        relative_path = os.path.relpath(subdir, EXTRACT_ROOT)
        save_path = os.path.join(OUTPUT_ROOT, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for fname in tqdm(files):
            if not fname.lower().endswith((".png", ".tif", ".jpg", ".jpeg")):
                continue

            input_path = os.path.join(subdir, fname)
            out_path   = os.path.join(save_path, fname)

            try:
                img = imageio.imread(input_path)

                if img.ndim == 3:
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                else:
                    img_gray = img

                mask = fuse_segments(img_gray)

                imageio.imwrite(out_path, mask)

                print("Saved →", out_path)

            except Exception as e:
                print(f"ERROR processing {input_path}: {e}")


# ---------- MAIN ----------
if __name__ == "__main__":
    extract_zip_files()
    process_all_images(EXTRACT_ROOT)
