#!/usr/bin/env python3
"""
pipeline_stardist.py
Loads stardist_inference.py from a file path and runs a full pipeline:
1) extract zip
2) run StarDist segmentation (using stardist_inference.py)
3) extract features (Xception) from original + segmented images
4) SVM prediction, save predictions & confusion chart
"""

import os
import sys
import zipfile
import argparse
import glob
import importlib.util
from pathlib import Path
from datetime import datetime
import time
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    import joblib
except Exception:
    joblib = None

# Keras/Xception imports for feature extractor
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.applications.xception import preprocess_input
except Exception:
    load_model = None

# ---------- Utilities ----------
def load_module_from_path(path, module_name="stardist_inference_local"):
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def extract_zip(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    patterns = ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.tif', '**/*.tiff']
    files = []
    for p in patterns:
        files.extend(Path(out_dir).glob(p))
    return [str(p) for p in sorted(files)]

def batch_predict_features(model, image_paths, batch_size=32, target_size=(256,256)):
    X = []
    names = []
    labels = []
    features = []
    for i, p in enumerate(image_paths):
        try:
            img = load_img(p, target_size=target_size)
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            X.append(arr)
            names.append(os.path.basename(p))
            labels.append(0 if 'non_necrosis' in p.lower() else 1)
        except Exception:
            continue

        if len(X) == batch_size or i == len(image_paths) - 1:
            Xb = np.stack(X, axis=0)
            feats = model.predict(Xb, verbose=0)
            features.append(feats)
            X = []

    if features:
        features = np.vstack(features)
    else:
        try:
            out_dim = model.output_shape[-1]
        except Exception:
            out_dim = 0
        features = np.zeros((0, out_dim))
    return features, names, labels

def find_best_svm_model(root_dir):
    candidates = []
    for pattern in [
        'Results/Final_Model_nulite*/**/svm_model.joblib',
        'Results/Final_Model*/**/svm_model.joblib',
        'Results/**/svm_model.joblib',
    ]:
        candidates.extend(glob.glob(os.path.join(root_dir, pattern), recursive=True))
    candidates = sorted(candidates)
    return candidates[0] if candidates else None

def load_optional_joblib(path):
    if joblib is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None

def generate_confusion_matrix_chart(predictions_df, output_dir):
    if predictions_df.empty:
        print("No predictions to visualize.")
        return
    actual = predictions_df['Actual_Class'].values
    predicted = predictions_df['Predicted'].values
    cm = confusion_matrix(actual, predicted, labels=[0,1])
    TN, FP = cm[0]
    FN, TP = cm[1]
    print("\nConfusion Matrix:")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    fig, axes = plt.subplots(1,2, figsize=(14,5))
    ax1 = axes[0]
    im = ax1.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, cm[i,j], ha="center", va="center", fontsize=16, fontweight='bold')
    ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
    ax1.set_xticklabels(['Non-Necrosis (0)', 'Necrosis (1)'])
    ax1.set_yticklabels(['Non-Necrosis (0)', 'Necrosis (1)'])
    ax1.set_xlabel('Predicted'); ax1.set_ylabel('Actual'); ax1.set_title('Confusion Matrix Heatmap')

    ax2 = axes[1]
    categories = ['TP','TN','FP','FN']
    values = [TP, TN, FP, FN]
    colors = ['#2ecc71','#2ecc71','#f39c12', '#e74c3c']
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x()+bar.get_width()/2., value, str(int(value)), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count'); ax2.set_title('Classification Results')
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'confusion_matrix_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix chart to: {chart_path}")
    metrics_csv = os.path.join(output_dir, 'confusion_matrix_metrics.csv')
    pd.DataFrame({
        'Metric': ['TP','TN','FP','FN'],
        'Count': [TP, TN, FP, FN]
    }).to_csv(metrics_csv, index=False)
    print(f"Saved metrics to: {metrics_csv}")

# ---------- Model-dir auto-detection ----------
def autodetect_stardist_model_dir(module_file_path):
    mpath = Path(module_file_path).resolve()
    parent = mpath.parent
    candidates = []
    # immediate siblings
    for entry in parent.iterdir():
        if entry.is_dir():
            name = entry.name.lower()
            if '2d' in name and 'versatile' in name and 'he' in name:
                candidates.append(str(entry))
            elif name.startswith('python_2d') or name.startswith('2d_'):
                candidates.append(str(entry))
    # also search deeper (one level) for common patterns
    for entry in parent.glob('*'):
        if entry.is_dir():
            for sub in entry.iterdir():
                if sub.is_dir():
                    lname = sub.name.lower()
                    if ('2d' in lname and 'versatile' in lname and 'he' in lname) or lname.startswith('python_2d'):
                        candidates.append(str(sub))
    # last resort: any dir under parent that matches patterns
    for entry in parent.iterdir():
        if entry.is_dir():
            if fnmatch.fnmatch(entry.name.lower(), '*2d*versatile*he*') or fnmatch.fnmatch(entry.name.lower(), 'python_*'):
                candidates.append(str(entry))
    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)
    return seen[0] if seen else None

# ---------- Main ----------
def main():
    script_start_time = time.time()
    print("\nPIPELINE STARTED:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, help='Path to input zip containing images')
    parser.add_argument('--output', default=None, help='Output base dir (default: script folder)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--svm-path', default=None, help='Optional: path to svm joblib to use for prediction')
    parser.add_argument('--stardist-module-file', default=r'/home/llmPathoUser/pathologyStudentsAug25/pathologyStudentsAug25/Codes and Details/segmentation_models/stardist/stardist_inference.py', help='Path to stardist_inference.py module file (local)')
    parser.add_argument('--stardist-model-dir', default=None, help='Path to the StarDist model folder (e.g. .../python_2D_versatile_he). If omitted, pipeline will attempt auto-detection.')
    args = parser.parse_args()

    zip_path = os.path.abspath(args.zip)
    if not os.path.exists(zip_path):
        print('Zip path not found:', zip_path); return

    script_dir = str(Path(__file__).resolve().parent)
    base_out = args.output or script_dir
    base_out = os.path.abspath(base_out)
    os.makedirs(base_out, exist_ok=True)

    # 1) extract
    original_dir = os.path.join(base_out, 'original_input_images')
    print(f'Extracting zip to {original_dir} ...')
    imgs = extract_zip(zip_path, original_dir)
    print(f'Found {len(imgs)} images')

    # 2) segmentation using StarDist module from file
    segmented_dir = os.path.join(base_out, 'segmented_input_images')
    print('Running StarDist segmentation...')

    st_module = None
    mod_path = args.stardist_module_file
    try:
        st_module = load_module_from_path(mod_path, module_name="stardist_inference_user")
        print(f"Loaded stardist module from: {mod_path}")
    except Exception as e:
        print(f"Could not load module from {mod_path}: {e}")
        try:
            import infer_stardist as st_local
            st_module = st_local
            print("Imported 'infer_stardist' from PYTHONPATH.")
        except Exception as e2:
            print("Fallback import failed:", e2)
            st_module = None

    if st_module is None:
        print("StarDist inference module not available. Skipping segmentation. You can still run feature extraction on original images only.")
    else:
        # Find model_dir to pass into load_model(model_dir)
        model_dir = args.stardist_model_dir
        if not model_dir:
            model_dir = getattr(st_module, "DEFAULT_MODEL_DIR", None)
        if not model_dir:
            try:
                detected = autodetect_stardist_model_dir(mod_path)
                if detected:
                    print("Auto-detected StarDist model directory:", detected)
                    model_dir = detected
            except Exception:
                model_dir = None

        if not model_dir:
            parent = Path(mod_path).resolve().parent
            fallback = None
            for p in parent.rglob('python_*2d*versatile*he*'):
                if p.is_dir():
                    fallback = str(p)
                    break
            if fallback:
                print("Found fallback model dir:", fallback)
                model_dir = fallback

        if not model_dir:
            print("\nERROR: StarDist model directory not found automatically.")
            print("Please pass --stardist-model-dir pointing to the model folder (example):")
            print("  --stardist-model-dir '/home/llmPathoUser/.../python_2D_versatile_he'\n")
        else:
            try:
                if hasattr(st_module, "load_model"):
                    st_model = st_module.load_model(model_dir)
                else:
                    raise AttributeError("stardist module does not expose 'load_model(model_dir)'")
                if hasattr(st_module, "infer_images"):
                    st_module.infer_images(st_model, original_dir, segmented_dir, size=(256,256))
                else:
                    raise AttributeError("stardist module does not expose 'infer_images(...)'")
            except Exception as e:
                print("StarDist segmentation failed with error:", e)
                print("You can still run feature extraction on original images only.")

    # 3) feature extraction
    extractor_path_seg = os.path.join(Path(__file__).resolve().parents[1], 'Codes and Details', 'saved_models', 'Xception_feature_extractor.h5')
    extractor_path = os.path.join(Path(__file__).resolve().parents[1], 'Codes and Details', 'saved_models', 'Xception_feature_extractor_original.h5')

    if not os.path.exists(extractor_path_seg) or not os.path.exists(extractor_path):
        print("Feature extractor model(s) missing. Expected at:")
        print(" ", extractor_path_seg)
        print(" ", extractor_path)
        return

    if load_model is None:
        print('TensorFlow/Keras not available in this environment. Install tensorflow.'); return

    print('Loading feature extractor models...')
    feat_model_seg = load_model(extractor_path_seg)
    feat_model = load_model(extractor_path)

    # original features
    original_images = sorted([p for p in glob.glob(os.path.join(original_dir, '**', '*.*'), recursive=True)
                              if p.lower().endswith(('.png', '.jpg', '.jpeg'))])
    feats_orig, names_orig, labels_orig = batch_predict_features(feat_model, original_images, batch_size=args.batch_size)
    cols = [f'feature_{i}' for i in range(feats_orig.shape[1])] if feats_orig.size else []
    df_orig = pd.DataFrame(feats_orig, columns=cols) if feats_orig.size else pd.DataFrame(columns=cols)
    df_orig['Image_Name'] = names_orig
    df_orig['label'] = labels_orig
    orig_csv = os.path.join(base_out, 'features_original.csv')
    df_orig.to_csv(orig_csv, index=False)
    print('Saved original features to', orig_csv)

    # segmented features
    segmented_images = sorted([p for p in glob.glob(os.path.join(segmented_dir, '**', '*.*'), recursive=True)
                               if p.lower().endswith(('.png', '.jpg', '.jpeg'))])
    feats_seg, names_seg, labels_seg = batch_predict_features(feat_model_seg, segmented_images, batch_size=args.batch_size)
    df_seg = pd.DataFrame(feats_seg, columns=cols) if feats_seg.size else pd.DataFrame(columns=cols)
    df_seg['Image_Name'] = names_seg
    df_seg['label'] = labels_seg
    seg_csv = os.path.join(base_out, 'features_segmented.csv')
    df_seg.to_csv(seg_csv, index=False)
    print('Saved segmented features to', seg_csv)

    # 4) prediction
    root_dir = str(Path(__file__).resolve().parents[1])
    svm_path = args.svm_path
    preferred_svm = os.path.join(root_dir, 'Results', 'Final_Model_nulite', 'Batch_1', 'svm_model.joblib')
    if svm_path is None:
        if os.path.exists(preferred_svm):
            svm_path = preferred_svm
            print(f"Using preferred SVM model: {svm_path}")
        else:
            svm_path = find_best_svm_model(root_dir)

    predictions = []
    if svm_path and joblib is not None:
        print('Loading SVM model from', svm_path)
        svm = load_optional_joblib(svm_path)
        def predict_df(df, source_label, threshold=0.5):
            if df.empty:
                return []
            X = df.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore').values
            probs = None
            try:
                probs = svm.predict_proba(X)[:,1]
            except Exception:
                probs = np.zeros(len(X))
            y_pred_custom = (probs >= threshold).astype(int)
            rows = []
            for idx, (name, pr) in enumerate(zip(df['Image_Name'].tolist(), probs.tolist())):
                actual_label = int(df['label'].iloc[idx]) if 'label' in df.columns else -1
                predicted_label = int(y_pred_custom[idx])
                rows.append({
                    'Image_Name': name,
                    'Source': source_label,
                    'Predicted': predicted_label,
                    'Actual_Class': actual_label,
                    'Probability': float(pr)
                })
            return rows

        df_seg_ren = df_seg.rename(columns=lambda c: f"seg_{c}" if c.startswith("feature_") else c)
        df_comb = pd.merge(df_orig, df_seg_ren, on='Image_Name', how='outer', suffixes=('','_seg'))
        df_comb = df_comb.fillna(0)
        feature_cols = [c for c in df_comb.columns if c.startswith('feature_') or c.startswith('seg_feature_')]
        if not feature_cols:
            feature_cols = [c for c in df_comb.columns if c not in ('Image_Name','label','seg_label')]
        df_for_pred = df_comb[ feature_cols + ['Image_Name'] ] if feature_cols else df_comb[['Image_Name']].copy()
        cols_order = [c for c in df_for_pred.columns if c != 'Image_Name'] + ['Image_Name']
        df_for_pred = df_for_pred[cols_order]
        df_for_pred['label'] = df_comb.get('label', df_comb.get('seg_label', -1)).values

        predictions.extend(predict_df(df_for_pred, "combined", threshold=0.3))
    else:
        print('No SVM model found or joblib unavailable. Skipping prediction. Searched path:', svm_path)

    out_excel = os.path.join(base_out, 'predictions.xlsx')
    if predictions:
        pdf = pd.DataFrame(predictions)
        pdf.to_excel(out_excel, index=False)
        print('Saved predictions to', out_excel)
        generate_confusion_matrix_chart(pdf, base_out)
    else:
        pdf = pd.DataFrame(columns=['Image_Name','Source','Predicted','Probability','Actual_Class'])
        pdf.to_excel(out_excel, index=False)
        print('Wrote empty predictions template to', out_excel)

    total_time = time.time() - script_start_time
    print("\nPIPELINE COMPLETED. Total time: {:.1f}s. Output folder: {}".format(total_time, base_out))

if __name__ == '__main__':
    main()
