import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sampv2.metrics import evaluate_basic_classifier
from sampv2.stacking import get_oof_with_selected_models


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate SAMPv2.")
    parser.add_argument("--train", required=True, help="Training CSV file.")
    parser.add_argument("--test", required=True, help="Independent test CSV file.")
    parser.add_argument(
        "--output",
        default="results/test_predictions.csv",
        help="Path for independent test predictions.",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/metrics.csv",
        help="Path for summary metrics.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of repeated first-layer OOF runs.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--knn-neighbors",
        type=int,
        default=50,
        help="Number of neighbors for KNN classifiers.",
    )
    return parser.parse_args()


def load_feature_matrix(path):
    data = pd.read_csv(path, index_col=0)
    if "labels" not in data.columns:
        raise ValueError(f"{path} must contain a 'labels' column.")
    y = data.pop("labels").to_numpy()
    return data, y


def build_first_layer_models(knn_neighbors):
    return [
        SVC(kernel="rbf", C=1, probability=True),
        KNeighborsClassifier(n_neighbors=knn_neighbors),
    ], ["SVC", "KNN"]


def build_second_layer_models(knn_neighbors):
    return [
        KNeighborsClassifier(n_neighbors=knn_neighbors),
    ], ["KNN"]


def run_training(args):
    xtrain, ytrain = load_feature_matrix(args.train)
    xtest, ytest = load_feature_matrix(args.test)

    all_models, model_names = build_first_layer_models(args.knn_neighbors)
    second_layer_models, second_layer_names = build_second_layer_models(args.knn_neighbors)
    combinations = list(itertools.combinations(range(len(all_models)), 2))

    output_path = Path(args.output)
    metrics_output_path = Path(args.metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)

    best_accuracy = 0
    best_first_layer = None
    best_second_layer = None
    results = []

    for combo_idx, combo in enumerate(combinations):
        selected_models = [all_models[i] for i in combo]
        selected_model_names = [model_names[i] for i in combo]
        combo_name = "-".join(name[:3] for name in selected_model_names)

        print(f"\n===== Testing combination {combo_idx + 1}/{len(combinations)}: {combo_name} =====")
        start_time = time.time()
        proba_matrix, proba_ind_mean = get_oof_with_selected_models(
            xtrain,
            ytrain,
            xtest,
            selected_models,
            selected_model_names,
            n_repeats=args.n_repeats,
            n_splits=args.n_splits,
            output_dir=metrics_output_path.parent,
        )
        elapsed_minutes = (time.time() - start_time) / 60
        print(f"First-layer training time: {elapsed_minutes:.2f} minutes")

        train_meta_columns = [f"Meta_{i + 1}" for i in range(proba_matrix.shape[1])]
        train_meta = pd.DataFrame(proba_matrix, columns=train_meta_columns, index=xtrain.index)
        xtrain_enhanced = pd.concat([xtrain, train_meta], axis=1)

        test_meta_columns = [f"Meta_{i + 1}" for i in range(proba_ind_mean.shape[1])]
        test_meta = pd.DataFrame(proba_ind_mean, columns=test_meta_columns, index=xtest.index)
        xtest_enhanced = pd.concat([xtest, test_meta], axis=1)

        cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

        for model, second_name in zip(second_layer_models, second_layer_names):
            print(f"\nTesting {second_name} as second-layer classifier")
            oof_preds = np.zeros_like(ytrain, dtype=float)
            test_preds = []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(xtrain_enhanced, ytrain)):
                print(f"Fold {fold_idx + 1}/{args.n_splits}")
                x_tr, y_tr = xtrain_enhanced.iloc[train_idx], ytrain[train_idx]
                x_val = xtrain_enhanced.iloc[val_idx]

                model_fold = clone(model)
                model_fold.fit(x_tr, y_tr)

                if hasattr(model_fold, "predict_proba"):
                    val_prob = model_fold.predict_proba(x_val)[:, 1]
                    test_prob = model_fold.predict_proba(xtest_enhanced)[:, 1]
                else:
                    val_prob = model_fold.predict(x_val)
                    test_prob = model_fold.predict(xtest_enhanced)

                oof_preds[val_idx] = val_prob
                test_preds.append(test_prob)

            test_preds = np.array(test_preds)
            test_score = test_preds.mean(axis=0)
            oof_final = (oof_preds >= 0.5).astype(int)
            test_final = (test_score >= 0.5).astype(int)

            print("Training set metrics")
            train_metrics = evaluate_basic_classifier(ytrain, oof_final, y_score=oof_preds)
            print(train_metrics.to_string(index=False))

            print("Independent test set metrics")
            test_metrics = evaluate_basic_classifier(ytest, test_final, y_score=test_score)
            print(test_metrics.to_string(index=False))

            pd.DataFrame(
                {
                    "y_true": ytest,
                    "y_pred": test_final,
                    "y_score": test_score,
                },
                index=xtest.index,
            ).to_csv(output_path)

            train_accuracy = train_metrics.loc[
                train_metrics["Metric"] == "Accuracy",
                "Value",
            ].values[0]
            test_accuracy = test_metrics.loc[
                test_metrics["Metric"] == "Accuracy",
                "Value",
            ].values[0]

            result_row = {
                "First_Layer_Models": combo_name,
                "Second_Layer_Model": second_name,
                "Train_Accuracy": train_accuracy,
                "Test_Accuracy": test_accuracy,
            }

            for metric, value in zip(train_metrics["Metric"], train_metrics["Value"]):
                result_row[f"Train_{metric}"] = value
            for metric, value in zip(test_metrics["Metric"], test_metrics["Value"]):
                result_row[f"Test_{metric}"] = value

            results.append(result_row)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_first_layer = combo_name
                best_second_layer = second_name
                print(
                    "New best: "
                    f"{best_first_layer} + {best_second_layer} | "
                    f"Accuracy = {best_accuracy:.4f}"
                )

    pd.DataFrame(results).to_csv(metrics_output_path, index=False)

    print("\n===== BEST COMBINATION =====")
    print(f"First layer models: {best_first_layer}")
    print(f"Second layer model: {best_second_layer}")
    print(f"Best test accuracy: {best_accuracy:.4f}")


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    args = parse_args()
    start_time = time.time()
    run_training(args)
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"Whole time: {elapsed_minutes:.2f} minutes")


if __name__ == "__main__":
    main()
