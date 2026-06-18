import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold


def get_oof_with_selected_models(
    x_train,
    y_train,
    x_test,
    selected_models,
    model_names,
    n_repeats=5,
    n_splits=5,
    random_state=23,
    output_dir=None,
):
    """Generate out-of-fold train predictions and averaged test predictions."""
    n_selected = len(selected_models)
    proba_matrix = np.zeros((x_train.shape[0], n_selected * 2))
    all_test_predictions = np.zeros((n_repeats, x_test.shape[0], n_selected * 2))

    for iteration in range(n_repeats):
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state + iteration,
        )
        iter_proba_matrix = np.zeros((x_train.shape[0], n_selected * 2))
        iter_test_predictions = np.zeros((n_splits, x_test.shape[0], n_selected * 2))

        print(f"Starting iteration {iteration + 1}/{n_repeats}")

        for fold_idx, (train_index, test_index) in enumerate(kf.split(x_train)):
            kf_x_train = x_train.iloc[train_index]
            kf_y_train = y_train[train_index]
            kf_x_test = x_train.iloc[test_index]

            for idx, model in enumerate(selected_models):
                current_model = clone(model)
                current_model.fit(kf_x_train, kf_y_train)

                if hasattr(current_model, "predict_proba"):
                    proba_e = current_model.predict_proba(kf_x_test)
                    proba_ind = current_model.predict_proba(x_test)
                else:
                    pred_e = current_model.predict(kf_x_test).reshape(-1, 1)
                    pred_ind = current_model.predict(x_test).reshape(-1, 1)
                    proba_e = np.hstack([1 - pred_e, pred_e])
                    proba_ind = np.hstack([1 - pred_ind, pred_ind])

                start = idx * 2
                end = (idx + 1) * 2
                iter_proba_matrix[test_index, start:end] = proba_e
                iter_test_predictions[fold_idx, :, start:end] = proba_ind

            print(f"  Iteration {iteration + 1}, fold {fold_idx + 1} completed")

        proba_matrix += iter_proba_matrix
        all_test_predictions[iteration] = iter_test_predictions.mean(axis=0)

    proba_matrix = proba_matrix / n_repeats
    proba_ind_mean = all_test_predictions.mean(axis=0)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        combo_name = "-".join(name[:3] for name in model_names)
        pd.DataFrame(proba_matrix).to_csv(
            output_dir / f"proba_matrix_train_{combo_name}.csv",
            index=False,
        )
        pd.DataFrame(proba_ind_mean).to_csv(
            output_dir / f"proba_matrix_test_{combo_name}.csv",
            index=False,
        )

    return proba_matrix, proba_ind_mean
