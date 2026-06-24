import numpy as np
import transformers
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error


def calculate_metric_for_regression(logits: np.ndarray, labels: np.ndarray, label_names=None):
    """Per-label and mean metrics for single- or multi-label regression.

    r2 is defined as pearson**2 (not sklearn's coefficient of determination)
    to match RiboNN's evaluation convention.
    """
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])

    predictions = logits.squeeze()
    labels = labels.squeeze()

    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
        labels = labels[:, np.newaxis]

    n_labels = predictions.shape[1]
    metrics = {}

    all_valid_preds = []
    all_valid_labels = []
    per_label = {"pearson": [], "spearman": [], "r2": [], "cell-type": []}

    for i in range(n_labels):
        preds_i = predictions[:, i]
        labels_i = labels[:, i]
        valid = ~np.isnan(labels_i)
        name = label_names[i] if label_names else str(i)
        if valid.sum() < 2:
            print(f"[Metrics] label '{name}' skipped (fewer than 2 valid samples)")
            continue
        preds_i = preds_i[valid]
        labels_i = labels_i[valid]

        all_valid_preds.append(preds_i)
        all_valid_labels.append(labels_i)

        pearson_i, _ = pearsonr(labels_i, preds_i)
        spearman_i, _ = spearmanr(labels_i, preds_i)
        r2_i = pearson_i ** 2
        per_label["pearson"].append(pearson_i)
        per_label["spearman"].append(spearman_i)
        per_label["r2"].append(r2_i)
        per_label["cell-type"].append(name)

    metrics["mse_loss_mean"] = mean_squared_error(
        np.concatenate(all_valid_labels), np.concatenate(all_valid_preds)
    )
    metrics["pearson_corr_mean"] = np.mean(per_label["pearson"])
    metrics["spearman_corr_mean"] = np.mean(per_label["spearman"])
    metrics["r2_score_mean"] = np.mean(per_label["r2"])

    # Mean TE per sequence: average across cell-types (NaN-safe), then correlate.
    # Mask predictions by label NaN so both means cover the same cell-types per sequence.
    predictions_masked = np.where(np.isnan(labels), np.nan, predictions)
    mean_pred_TE = np.nanmean(predictions_masked, axis=1)
    mean_label_TE = np.nanmean(labels, axis=1)
    valid_TE = ~(np.isnan(mean_pred_TE) | np.isnan(mean_label_TE))
    if valid_TE.sum() >= 2:
        pearson_mean_TE, _ = pearsonr(mean_label_TE[valid_TE], mean_pred_TE[valid_TE])
        r2_mean_TE = pearson_mean_TE ** 2
    else:
        pearson_mean_TE = float("nan")
        r2_mean_TE = float("nan")

    metrics["pearson_mean_TE"] = pearson_mean_TE
    metrics["r2_mean_TE"] = r2_mean_TE

    for i, name in enumerate(per_label["cell-type"]):
        metrics[f"pearson_{name}"] = per_label["pearson"][i]
        metrics[f"spearman_{name}"] = per_label["spearman"][i]
        metrics[f"r2_{name}"] = per_label["r2"][i]

    return metrics


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Save model state dict to disk, moving tensors to CPU first."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
