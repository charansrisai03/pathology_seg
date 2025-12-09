# debug_svm_diagnostics.py
import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# adjust if needed
BASE_OUT = os.path.abspath('.')   # run from pipeline output folder (where features_*.csv are)
SVM_PATH = os.path.expanduser("/home/llmPathoUser/pathologyStudentsAug25/pathologyStudentsAug25/Results/Final_Model/Batch_2/svm_model.joblib")

print("Working folder:", BASE_OUT)
print("SVM path:", SVM_PATH)
print()

if not os.path.exists(SVM_PATH):
    raise SystemExit("SVM not found at path")

svm = joblib.load(SVM_PATH)
print("Loaded SVM:", type(svm))
print("svm.classes_:", getattr(svm, "classes_", None))
print("svm.n_features_in_:", getattr(svm, "n_features_in_", None))

# load features
def load_feats(path):
    if not os.path.exists(path):
        print("Missing:", path); return None
    df = pd.read_csv(path)
    feat_cols = sorted([c for c in df.columns if c.startswith('feature_')])
    print(f"Loaded {os.path.basename(path)} shape={df.shape} feat_cols={len(feat_cols)}")
    return df, feat_cols

orig = load_feats(os.path.join(BASE_OUT, 'features_original.csv'))
seg = load_feats(os.path.join(BASE_OUT, 'features_segmented.csv'))

# helper to predict and show stats
def predict_and_stats(df_tuple, label):
    if df_tuple is None:
        print(f"No {label} df"); return None
    df, feat_cols = df_tuple
    if not feat_cols:
        print(f"No feature columns in {label}")
        return None
    X = df[feat_cols].values.astype(np.float32)
    # align/truncate/pad if needed
    n_expected = getattr(svm, "n_features_in_", None)
    if n_expected is not None:
        if X.shape[1] < n_expected:
            pad = np.zeros((X.shape[0], n_expected - X.shape[1]), dtype=np.float32)
            X = np.concatenate([X, pad], axis=1)
        elif X.shape[1] > n_expected:
            X = X[:, :n_expected]
    print(f"{label} X shape after align: {X.shape}")
    # predictions
    try:
        y_pred = svm.predict(X)
    except Exception as e:
        print("Predict error:", e); return None
    print(f"{label} predictions unique counts:\n", pd.Series(y_pred).value_counts())
    # decision scores / probs
    try:
        if hasattr(svm, "decision_function"):
            scores = svm.decision_function(X)
            print(f"{label} decision_function stats: mean={scores.mean():.4f} std={scores.std():.4f} min={scores.min():.4f} max={scores.max():.4f}")
        if hasattr(svm, "predict_proba"):
            probs = svm.predict_proba(X)[:, 1]
            print(f"{label} predict_proba stats: mean={probs.mean():.4f} std={probs.std():.4f} min={probs.min():.4f} max={probs.max():.4f}")
        else:
            probs = None
    except Exception as e:
        print("Prob/score error:", e); probs = None

    # if label column exists, compare
    if 'label' in df.columns:
        y_true = df['label'].values
        print(f"{label} true label counts:\n", pd.Series(y_true).value_counts())
        try:
            cm = confusion_matrix(y_true[:len(y_pred)], y_pred)
            print(f"{label} confusion matrix:\n", cm)
            print(classification_report(y_true[:len(y_pred)], y_pred, zero_division=0))
            # show up to 5 mismatches where true!=pred
            mism = (y_true[:len(y_pred)] != y_pred)
            if mism.sum() > 0:
                print(f"Found {mism.sum()} mismatches; showing up to 5 examples:")
                idxs = np.where(mism)[0][:5]
                for i in idxs:
                    print(f" idx={i} name={df.iloc[i].get('Image_Name','<noname>')} true={y_true[i]} pred={y_pred[i]}")
                    print("   sample feature[0:10]:", X[i,:10].tolist())
        except Exception as e:
            print("Confusion/error:", e)
    return {'y_pred': y_pred, 'probs': probs, 'X': X, 'df': df}

res_seg = predict_and_stats(seg, 'SEGMENTED')
res_ori = predict_and_stats(orig, 'ORIGINAL')

# If both predict all 1s, print some sample rows for manual inspection
def dump_samples(res, tag):
    if res is None: return
    pp = res['probs'] if res['probs'] is not None else None
    y = res['y_pred']
    X = res['X']
    print(f"\n{tag} sample predictions (first 10):", y[:10].tolist())
    if pp is not None:
        print(f"{tag} sample probs (first 10):", pp[:10].tolist())
    print(f"{tag} sample X[0][:20]:", X[0,:20].tolist())

dump_samples(res_seg, 'SEGMENTED')
dump_samples(res_ori, 'ORIGINAL')
