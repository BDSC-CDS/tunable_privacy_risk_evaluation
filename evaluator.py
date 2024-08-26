import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


def get_thresholds(distribution, alpha=list(np.linspace(0, 1, 1000))):
    thresholds = np.quantile(distribution, q=alpha[1:-1], method="linear")
    thresholds = np.concatenate(
        [
            np.array(distribution.min() - 1e-4).reshape(-1),
            thresholds,
            np.array(distribution.max() + 1e-4).reshape(-1),
        ],
        axis=0,
    )

    return thresholds[:, None]


def get_predictions(member_signals, non_member_signals, thresholds):
    member_preds = np.less(member_signals, thresholds)
    non_member_preds = np.less(non_member_signals, thresholds)
    predictions = np.concatenate([member_preds, non_member_preds], axis=1)

    return predictions


def calculate_metrics(true_labels, predicted_labels, show_roc=True, log_axis=True):
    accuracy = np.mean(predicted_labels == true_labels, axis=1)
    tn = np.sum(true_labels == 0) - np.sum(
        predicted_labels[:, true_labels == 0], axis=1
    )
    tp = np.sum(predicted_labels[:, true_labels == 1], axis=1)
    fp = np.sum(predicted_labels[:, true_labels == 0], axis=1)
    fn = np.sum(true_labels == 1) - np.sum(
        predicted_labels[:, true_labels == 1], axis=1
    )
    roc_auc = metrics.auc(
        fp / (np.sum(true_labels == 0)), tp / (np.sum(true_labels == 1))
    )

    if show_roc:
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)

        fig, ax = plt.subplots(nrows=1, ncols=1)

        if log_axis:
            ax.loglog(fpr, tpr, label="Population Attack (area = %0.4f)" % roc_auc)
            ax.loglog(
                [0, 1], [0, 1], label="Random guess", linestyle="--", color="black"
            )
        else:
            ax.plot(fpr, tpr, label="Population Attack (area = %0.4f)" % roc_auc)
            ax.plot([0, 1], [0, 1], label="Random guess", linestyle="--", color="black")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        # fig.savefig(
        #     "figs_test/{}_seed_{}.png".format(args.dataset, seed),
        #     format="png",
        # )
        plt.show()

    results = {
        "accuracy": accuracy,
        "tn": tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "auc": roc_auc,
        "true labels": true_labels,
        "predicted labels": predicted_labels,
    }

    # fpr = fp / (fp + tn)
    # tpr = tp / (tp + fn)
    # roc_auc = np.trapz(x=fpr, y=tpr)

    return results


def estimate_privacy_risk(
    population_signals,
    member_signals,
    non_member_signals,
    alpha=list(np.linspace(0, 1, 1000)),
    show_roc=True,
    log_axis=True,
):
    thresholds = get_thresholds(population_signals, alpha=alpha)

    true_labels = np.concatenate(
        [np.ones(len(member_signals)), np.zeros(len(non_member_signals))]
    )

    num_threshold = len(alpha)
    member_signals = member_signals.reshape(-1, 1).repeat(num_threshold, 1).T
    non_member_signals = non_member_signals.reshape(-1, 1).repeat(num_threshold, 1).T
    predictions = get_predictions(member_signals, non_member_signals, thresholds)

    results = calculate_metrics(
        true_labels, predictions, show_roc=show_roc, log_axis=log_axis
    )

    return results
