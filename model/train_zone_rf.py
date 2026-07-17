import os
import sys
import glob
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

# Import thermal_utils
sys.path.insert(0, r"C:\Users\HP\Downloads\files")
from thermal_utils import process_image, feature_vector

DATA_DIR = r"C:\Projects\sleeping-monitor\labeled_zone_dataset\train"
TEST_DIR = r"C:\Projects\sleeping-monitor\labeled_zone_dataset\test"
MODEL_OUT_PATH = r"C:\Projects\sleeping-monitor\zone_model_rf.joblib"
RANDOM_STATE = 42

def load_dataset(data_dir):
    labels = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    X, y, groups = [], [], []
    group_id = 0

    for label in labels:
        paths = glob.glob(os.path.join(data_dir, label, "*.png"))
        print(f"Loading {label}: {len(paths)} images")
        
        # Grouping by original sequence
        seq_dict = {}
        for path in paths:
            seq = os.path.basename(path).split("_")[0]
            if seq not in seq_dict:
                seq_dict[seq] = group_id
                group_id += 1
                
            feats, mask, thresh, plausible = process_image(path)
            if feats is None or not plausible:
                continue
                
            X.append(feature_vector(feats))
            y.append(label)
            groups.append(seq_dict[seq])

    return np.array(X), np.array(y), np.array(groups), labels

def main():
    print("=== Training Random Forest Zone Classifier ===")
    print(f"Loading training data from {DATA_DIR}...")
    X_train, y_train, groups_train, labels = load_dataset(DATA_DIR)
    
    print(f"Loading test data from {TEST_DIR}...")
    X_test, y_test, _, _ = load_dataset(TEST_DIR)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # We will use HistGradientBoostingClassifier as it handles tabular data excellently
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, random_state=RANDOM_STATE)
    
    print("\nTraining HistGradientBoostingClassifier...")
    clf.fit(X_train, y_train_enc)
    
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    print("\n=== Evaluation on Test Set ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))
    
    # Evaluate predicted probabilities
    probas = clf.predict_proba(X_test)
    print("\nSample Probability Output (First 5 test frames):")
    for i in range(5):
        print(f"True: {y_test[i]:<15} Pred: {y_pred[i]:<15}")
        for j, label in enumerate(labels):
            print(f"  P({label}): {probas[i][j]*100:.1f}%")
            
    joblib.dump({"model": clf, "labels": labels}, MODEL_OUT_PATH)
    print(f"\nModel saved to {MODEL_OUT_PATH}")

if __name__ == "__main__":
    main()
