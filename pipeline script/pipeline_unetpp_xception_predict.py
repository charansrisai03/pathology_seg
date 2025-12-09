#!/usr/bin/env python3

import os
import sys
import zipfile
import argparse
import shutil
import glob
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    import joblib
except Exception:
    joblib = None

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.xception import preprocess_input
    from tensorflow.keras.preprocessing.image import load_img
except Exception:
    load_model = None


def organize_images_by_classification(predictions_df, original_dir, segmented_dir, output_dir):
    """
    Organize images into folder structure:
    output_dir/
    ├── original_images_by_class/
    │   ├── TP/
    │   ├── TN/
    │   ├── FP/
    │   └── FN/
    └── segmented_images_by_class/
        ├── TP/
        ├── TN/
        ├── FP/
        └── FN/
    """
    if predictions_df.empty:
        print("No predictions to organize.")
        return
    
    print("\n" + "="*60)
    print("📁 ORGANIZING IMAGES BY CLASSIFICATION")
    print("="*60)
    
    # Create folder structure
    original_class_dir = os.path.join(output_dir, 'original_images_by_class')
    segmented_class_dir = os.path.join(output_dir, 'segmented_images_by_class')
    
    for base_dir in [original_class_dir, segmented_class_dir]:
        for classification in ['TP', 'TN', 'FP', 'FN']:
            os.makedirs(os.path.join(base_dir, classification), exist_ok=True)
    
    # Count files organized
    tp_count = tn_count = fp_count = fn_count = 0
    
    # Build dictionaries of all images in original and segmented directories (recursive search)
    original_images_map = {}
    segmented_images_map = {}
    
    # Search for images recursively in original_dir
    for root, dirs, files in os.walk(original_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                full_path = os.path.join(root, f)
                original_images_map[f] = full_path
    
    # Search for images recursively in segmented_dir
    for root, dirs, files in os.walk(segmented_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                full_path = os.path.join(root, f)
                segmented_images_map[f] = full_path
    
    print(f"  Found {len(original_images_map)} original images")
    print(f"  Found {len(segmented_images_map)} segmented images")
    
    # Process each prediction
    for _, row in predictions_df.iterrows():
        img_name = row['Image_Name']
        predicted = row['Predicted']
        actual = row['Actual_Class']
        
        # Determine classification
        if predicted == 1 and actual == 1:
            classification = 'TP'
            tp_count += 1
        elif predicted == 0 and actual == 0:
            classification = 'TN'
            tn_count += 1
        elif predicted == 1 and actual == 0:
            classification = 'FP'
            fp_count += 1
        elif predicted == 0 and actual == 1:
            classification = 'FN'
            fn_count += 1
        else:
            continue
        
        # Copy original image (search in map first)
        if img_name in original_images_map:
            original_src = original_images_map[img_name]
            original_dst = os.path.join(original_class_dir, classification, img_name)
            try:
                shutil.copy2(original_src, original_dst)
            except Exception as e:
                print(f"  Warning: Could not copy original image {img_name}: {e}")
        
        # Copy segmented image (search in map first)
        if img_name in segmented_images_map:
            segmented_src = segmented_images_map[img_name]
            segmented_dst = os.path.join(segmented_class_dir, classification, img_name)
            try:
                shutil.copy2(segmented_src, segmented_dst)
            except Exception as e:
                print(f"  Warning: Could not copy segmented image {img_name}: {e}")
    
    print(f"\n✅ Image Organization Complete:")
    print(f"  TP (True Positives): {tp_count} images")
    print(f"  TN (True Negatives): {tn_count} images")
    print(f"  FP (False Positives): {fp_count} images")
    print(f"  FN (False Negatives): {fn_count} images")
    print(f"\n📂 Organized images saved to:")
    print(f"   {original_class_dir}")
    print(f"   {segmented_class_dir}")
    print("="*60)

def extract_zip(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    # gather images
    patterns = ['**/*.png', '**/*.jpg', '**/*.jpeg']
    files = []
    for p in patterns:
        files.extend(Path(out_dir).glob(p))
    return [str(p) for p in sorted(files)]

def extract_zip2(zip_path, out_dir):
    import zipfile, os
    imgs = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)

    # walk through all extracted files
    for root, dirs, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
                imgs.append(os.path.join(root, f))

    return imgs


def import_unetpp_infer_module():
    # Add nulite infer script directory to path
    base = Path(__file__).resolve().parents[1]
    infer_path = base / 'Codes and Details' / 'segmentation_models' / 'unetpp'
    if not infer_path.exists():
        raise FileNotFoundError(f"unetpp infer folder not found: {infer_path}")
    sys.path.insert(0, str(infer_path))
    try:
        import infer_unetpp as infer_mod
    except Exception as e:
        raise ImportError(f"Failed to import infer_unetpp.py: {e}")
    return infer_mod


def run_segmentation(input_dir, output_dir, model_path):
    infer_mod = import_unetpp_infer_module()
    model = infer_mod.load_model(model_path)
    infer_mod.infer_images(model, input_dir, output_dir)


def batch_predict_features(model, image_paths, batch_size=32, target_size=(256, 256)):
    # model: keras model that takes preprocessed images and returns feature vector
    X = []
    names = []
    features = []
    labels = []
    for i, p in enumerate(image_paths):
        try:
            img = load_img(p, target_size=target_size)
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            X.append(arr)
            names.append(os.path.basename(p))
            
            # Extract label from folder structure
            # Check if 'necrosis' is in the path (case-insensitive)
            # print(p)
            if 'non_necrosis' in p.lower():
                labels.append(0)
            else:
                labels.append(1)
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
        # fallback to empty shape with output dim if possible
        print('No features extracted, returning empty array')
        try:
            out_dim = model.output_shape[-1]
        except Exception:
            out_dim = 0
        features = np.zeros((0, out_dim))
    return features, names, labels


def find_best_svm_model(root_dir):
    # Search for svm_model.joblib under Results/Final_Model_nulite* then Results/Final_Model*
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
    """
    Generate confusion matrix chart from predictions.
    TP: Predicted=1, Actual=1 (Necrosis correctly identified)
    TN: Predicted=0, Actual=0 (Non-Necrosis correctly identified)
    FP: Predicted=1, Actual=0 (False alarm - predicted necrosis but wasn't)
    FN: Predicted=0, Actual=1 (Missed necrosis)
    """
    if predictions_df.empty:
        print("No predictions to visualize.")
        return
    
    actual = predictions_df['Actual_Class'].values
    predicted = predictions_df['Predicted'].values
    
    # Calculate confusion matrix
    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    # Print metrics
    print("\n" + "="*50)
    print(" CONFUSION MATRIX RESULTS")
    print("="*50)
    print(f"True Positives (TP):  {TP}  - Correctly identified Necrosis")
    print(f"True Negatives (TN):  {TN}  - Correctly identified Non-Necrosis")
    print(f"False Positives (FP): {FP}  - Incorrectly predicted as Necrosis")
    print(f"False Negatives (FN): {FN}  - Missed Necrosis cases")
    print("="*50 + "\n")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Chart 1: Confusion Matrix Heatmap ===
    ax1 = axes[0]
    im = ax1.imshow(cm, cmap='Blues', aspect='auto')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, cm[i, j], ha="center", va="center", 
                           color="white" if cm[i, j] > cm.max() / 2 else "black", 
                           fontsize=16, fontweight='bold')
    
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Non-Necrosis (0)', 'Necrosis (1)'], fontsize=11)
    ax1.set_yticklabels(['Non-Necrosis (0)', 'Necrosis (1)'], fontsize=11)
    ax1.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Class', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
    
    # === Chart 2: Bar Chart ===
    ax2 = axes[1]
    categories = ['True\nPositives\n(TP)', 'True\nNegatives\n(TN)', 
                  'False\nPositives\n(FP)', 'False\nNegatives\n(FN)']
    values = [TP, TN, FP, FN]
    colors = ['#2ecc71','#2ecc71','#f39c12', '#e74c3c']
    
    bars = ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Classification Results', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(values) * 1.15)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(output_dir, 'confusion_matrix_chart_unetpp.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f' Confusion matrix chart saved to: {chart_path}')
    plt.close()
    
    # === Save metrics summary to CSV ===
    metrics_summary = pd.DataFrame({
        'Metric': ['TP (True Positives)', 'TN (True Negatives)', 'FP (False Positives)', 'FN (False Negatives)'],
        'Count': [TP, TN, FP, FN],
        'Description': [
            'Correctly identified Necrosis',
            'Correctly identified Non-Necrosis',
            'Incorrectly predicted as Necrosis',
            'Missed Necrosis cases'
        ]
    })
    
    metrics_csv = os.path.join(output_dir, 'confusion_matrix_metrics_unetpp.csv')
    metrics_summary.to_csv(metrics_csv, index=False)
    print(f' Metrics summary saved to: {metrics_csv}')


def main():
    # ========== START TIMER ==========
    script_start_time = time.time()
    print("\n" + "="*60)
    print(" PIPELINE STARTED")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True, help='Path to input zip containing images')
    parser.add_argument('--output', default=None, help='Output base dir (default: the script folder)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--svm-path', default=None, help='Optional: path to svm joblib to use for prediction')
    args = parser.parse_args()

    zip_path = os.path.abspath(args.zip)
    if not os.path.exists(zip_path):
        print('Zip path not found:', zip_path)
        return

    # Default output dir is the folder containing this script (so segmented images
    # will be saved alongside the script if --output is not provided).
    script_dir = str(Path(__file__).resolve().parent)
    base_out = args.output or script_dir
    base_out = os.path.abspath(base_out)
    os.makedirs(base_out, exist_ok=True)

    # 1) extract
    original_dir = os.path.join(base_out, 'original_input_images')
    print(f'Extracting zip to {original_dir} ...')
    imgs = extract_zip(zip_path, original_dir)
    print(f'Found {len(imgs)} images')
    
    # 2) segmentation
    segmented_dir = os.path.join(base_out, 'segmented_input_images_unetpp')
    print('Running unetpp segmentation...')
    unetpp_model_path = os.path.join(
        Path(__file__).resolve().parents[1], 'Codes and Details', 'segmentation_models', 'unetpp', 'best_unetpp.pth'
    )
    if not os.path.exists(unetpp_model_path):
        print('Warning: unetpp model not found at', unetpp_model_path)
    try:
        run_segmentation(original_dir, segmented_dir, unetpp_model_path)
    except Exception as e:
        print('Segmentation failed:', e)
        print('You can still run feature extraction on original images only')

    # 3) feature extraction
    extractor_path_seg = os.path.join(
        Path(__file__).resolve().parents[1], 'Codes and Details', 'saved_models', 'Xception_feature_extractor_unetpp.h5'
    )
    if not os.path.exists(extractor_path_seg):
        print('Feature extractor model not found:', extractor_path_seg)
        return

    if load_model is None:
        print('TensorFlow/Keras not available in this environment. Install tensorflow.')
        return

    print('Loading feature extractor model...')
    feat_model_seg = load_model(extractor_path_seg)

    extractor_path = os.path.join(
        Path(__file__).resolve().parents[1], 'Codes and Details', 'saved_models', 'Xception_feature_extractor_original.h5'
    )
    if not os.path.exists(extractor_path):
        print('Feature extractor model not found:', extractor_path)
        return

    if load_model is None:
        print('TensorFlow/Keras not available in this environment. Install tensorflow.')
        return

    print('Loading feature extractor model...')
    feat_model = load_model(extractor_path)

    # original features
    original_images = sorted(
        [
            p
            for p in glob.glob(os.path.join(original_dir, '**', '*.*'), recursive=True)
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    )
    feats_orig, names_orig, labels_orig = batch_predict_features(feat_model, original_images, batch_size=args.batch_size)
    cols = [f'feature_{i}' for i in range(feats_orig.shape[1])] if feats_orig.size else []
    df_orig = pd.DataFrame(feats_orig, columns=cols) if feats_orig.size else pd.DataFrame(columns=cols)
    df_orig['Image_Name'] = names_orig
    df_orig['label'] = labels_orig
    orig_csv = os.path.join(base_out, 'features_original.csv')
    df_orig.to_csv(orig_csv, index=False)
    print('Saved original features to', orig_csv)

    # segmented features
    segmented_images = sorted(
        [
            p
            for p in glob.glob(os.path.join(segmented_dir, '**', '*.*'), recursive=True)
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    )
    feats_seg, names_seg, labels_seg = batch_predict_features(feat_model_seg, segmented_images, batch_size=args.batch_size)
    df_seg = pd.DataFrame(feats_seg, columns=cols) if feats_seg.size else pd.DataFrame(columns=cols)
    df_seg['Image_Name'] = names_seg
    df_seg['label'] = labels_seg
    seg_csv = os.path.join(base_out, 'features_segmented_unetpp.csv')
    df_seg.to_csv(seg_csv, index=False)
    print('Saved segmented features to', seg_csv)

    # 4) prediction
    root_dir = str(Path(__file__).resolve().parents[1])
    svm_path = args.svm_path
    # If user didn't pass --svm-path, prefer the specific Batch_12 model requested by user
    preferred_svm = os.path.join(root_dir, 'Results', 'Final_Model', 'Batch_16', 'svm_model.joblib')
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
        # optional scaler/pca in same folder
        svm_dir = os.path.dirname(svm_path)
        def predict_df(df, source_label, threshold=0.5):
            """
            Predict class with custom probability threshold.
            If probability >= threshold, predict necrosis (1), else non_necrosis (0)
            """
            if df.empty:
                return []
            X = df.iloc[:,:-2].values  # Exclude Image_Name and label columns
            probs = None
            try:
                probs = svm.predict_proba(X)[:, 1]
            except Exception:
                probs = np.zeros(len(X))
            
            # Apply custom threshold: if prob >= 0.3, predict 1 (necrosis), else 0
            y_pred_custom = (probs >= threshold).astype(int)
            
            rows = []
            for idx, (name, pr) in enumerate(zip(df['Image_Name'].tolist(), probs.tolist())):
                actual_label = int(df['label'].iloc[idx]) if 'label' in df.columns else -1
                predicted_label = y_pred_custom[idx]
                rows.append({
                    'Image_Name': name, 
                    'Source': source_label, 
                    'Predicted': int(predicted_label), 
                    'Actual_Class': actual_label,
                    'Probability': float(pr)
                })
            return rows
         
        df_seg = df_seg.rename(columns=lambda c: f"seg_{c}" if c.startswith("feature_") else c)
        df=pd.concat([df_orig,df_seg],axis=1)

        df = df.loc[:, ~df.columns.duplicated(keep='last')]

        # print(df.columns.tolist())
        predictions.extend(predict_df(df, "combined"))
    else:
        print('No SVM model found or joblib unavailable. Skipping prediction. Searched path:', svm_path)

    # write predictions excel
    out_excel = os.path.join(base_out, 'predictions_unetpp.xlsx')
    if predictions:
        pdf = pd.DataFrame(predictions)
        pdf.to_excel(out_excel, index=False)
        print('Saved predictions to', out_excel)
        
        organize_images_by_classification(pdf, original_dir, segmented_dir, base_out)
        # Generate confusion matrix visualization
        generate_confusion_matrix_chart(pdf, base_out)
    else:
        # create empty template
        pdf = pd.DataFrame(columns=['Image_Name', 'Source', 'Predicted', 'Probability', 'Actual_Class'])
        pdf.to_excel(out_excel, index=False)
        print('Wrote empty predictions template to', out_excel)

    # ========== END TIMER & SUMMARY ==========
    script_end_time = time.time()
    total_time = script_end_time - script_start_time
    
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*60)
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Execution Time: {hours}h {minutes}m {seconds}s")
    print("="*60)
    print(f'Output folder: {base_out}\n')

if __name__ == '__main__':
    main()
