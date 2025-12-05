import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)

def get_args():
    parser = argparse.ArgumentParser(description='Calibrate Threshold, Evaluate, and Plot Curves')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to predictions.csv (with labels)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--save_plot', type=str, default='results/final_evaluation_curves.png', help='Path to save plot')
    parser.add_argument('--target_precision', type=float, default=None, help='Target Precision for threshold finding')
    return parser.parse_args()

def calculate_metrics(y_true, probs, threshold):
    y_pred = (probs > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cm': cm
    }

def plot_curves(y_true, y_scores, save_path, title_suffix=""):
    """
    Plots ROC and PR curves for the given data.
    """
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(12, 5))

    # 1. ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {title_suffix}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 2. PR
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AUPRC = {auprc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve {title_suffix}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Curves saved to {save_path}")

def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    
    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)
    
    # Clean data
    valid_df = df[df['label'] != '?'].copy()
    print(f"Total valid sites: {len(valid_df)}")

    # --- 1. Split Data (Grouped by Protein ID) ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(valid_df, groups=valid_df['uniprot_id']))
    
    calib_df = valid_df.iloc[train_idx]
    eval_df = valid_df.iloc[test_idx]
    
    print("-" * 40)
    print(f"Calibration Set : {len(calib_df)} sites ({calib_df['uniprot_id'].nunique()} proteins)")
    print(f"Evaluation Set  : {len(eval_df)} sites ({eval_df['uniprot_id'].nunique()} proteins)")
    print("-" * 40)

    def get_y_p(d):
        y = d['label'].apply(lambda x: 1 if str(x).strip() in ['P','1','1.0'] else 0).values
        p = d['probability'].values
        return y, p

    y_calib, p_calib = get_y_p(calib_df)
    y_eval, p_eval = get_y_p(eval_df)

    # --- 2. Find Best Threshold on Calibration Set ---
    print("\nFinding optimal threshold on Calibration Set...")
    best_th = 0.5
    best_score = 0.0
    
    if args.target_precision:
        precisions, recalls, thresholds = precision_recall_curve(y_calib, p_calib)
        valid_indices = np.where(precisions[:-1] >= args.target_precision)[0]
        if len(valid_indices) > 0:
            best_th = thresholds[valid_indices[0]]
            print(f"-> Selected Threshold {best_th:.4f} (Target Precision >= {args.target_precision})")
        else:
            print(f"-> Warning: Target Precision not reached. Optimizing F1 instead.")
            args.target_precision = None

    if args.target_precision is None:
        # Search for Max F1
        for th in np.arange(0.1, 0.99, 0.01):
            m = calculate_metrics(y_calib, p_calib, th)
            if m['f1'] > best_score:
                best_score = m['f1']
                best_th = th
        print(f"-> Selected Threshold {best_th:.4f} (Max F1 on Calibration Set)")

    # --- 3. Final Evaluation on Test Set ---
    print("\n" + "="*40)
    print("FINAL EVALUATION (on Unseen Data)")
    print("="*40)
    
    final_metrics = calculate_metrics(y_eval, p_eval, best_th)
    
    print(f"Threshold Used : {best_th:.4f}")
    print(f"Accuracy       : {final_metrics['acc']:.4f}")
    print(f"Precision      : {final_metrics['prec']:.4f}")
    print(f"Recall         : {final_metrics['rec']:.4f}")
    print(f"F1 Score       : {final_metrics['f1']:.4f}")
    print(f"MCC            : {final_metrics['mcc']:.4f}")
    print("-" * 40)
    print("Confusion Matrix (Evaluation Set):")
    cm = final_metrics['cm']
    print(f" TN: {cm[0,0]}  FP: {cm[0,1]}")
    print(f" FN: {cm[1,0]}  TP: {cm[1,1]}")
    print("="*40)

    # --- 4. Plot Curves for Evaluation Set ---
    plot_curves(y_eval, p_eval, args.save_plot, title_suffix="(Evaluation Set)")

if __name__ == "__main__":
    main()