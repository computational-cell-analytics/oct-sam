import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from util import get_flatline_handle, export_legend

png_dpi = 300

FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"


def plot_model_acc(mode="precision", plot=False, save_path=None):
    value_dict = {
        "nnU-Net_all": {
            "label": "nnU-Net",
            "precision": 0.963,
            "recall": 0.963,
            "f1-score": 0.963,
            "marker": "^",
            "symm_dice": 0.881,
        },
#        "v7-no-prompts-pp": {
#            "label": "octSAM*",
#            "precision": 0.588,
#            "recall": 0.494,
#            "f1-score": 0.53,
#            "marker": "s",
#            "symm_dice": 0.623,
#        },
        "v7-no-prompts": {
            "label": "octSAM*",
            "precision": 0.349,
            "recall": 0.47,
            "f1-score": 0.396,
            "marker": "s",
            "symm_dice": 0.533,
        },
        "v7-post": {
            "label": "octSAM",
            "precision": 0.928,
            "recall": 0.798,
            "f1-score": 0.851,
            "marker": "P",
            "symm_dice": 0.766,
        },
    }

    # Convert setting labels to numerical x positions
    offset = 0.08  # horizontal shift for scatter separation

    # Plot
    tick_rotation = 45

    main_label_size = 20
    main_tick_size = 16
    marker_size = 200

    labels = [value_dict[key]["label"] for key in value_dict.keys()]
    fig_width = len(value_dict) * 2

    if mode == "precision":
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for num, key in enumerate(list(value_dict.keys())):
            precision = [value_dict[key]["precision"]]
            recall = [value_dict[key]["recall"]]
            f1score = [value_dict[key]["f1-score"]]
            marker = value_dict[key]["marker"]
            x_pos = num + 1

            plt.scatter([x_pos - offset], precision,
                        color=COLOR_P, marker=marker, s=marker_size)
            plt.scatter([x_pos],         recall,
                        color=COLOR_R, marker=marker, s=marker_size)
            plt.scatter([x_pos + offset], f1score,
                        color=COLOR_F, marker=marker, s=marker_size)

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=main_tick_size, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        plt.ylim(0, 1)
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif mode == "dice":
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        if "Spiner" in labels:
            labels.remove("Spiner")

        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        x_pos = 1
        for num, key in enumerate(list(value_dict.keys())):
            runtime = [value_dict[key]["symm_dice"]]
            if runtime[0] is None:
                continue
            marker = value_dict[key]["marker"]
            plt.scatter([x_pos], runtime, label="Dice", color=COLOR_T, marker=marker, s=marker_size)
            x_pos = x_pos + 1

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=16, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Dice's coefficient", fontsize=main_label_size)
        plt.ylim(0, 1)
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif "both":
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for num, key in enumerate(list(value_dict.keys())):
            precision = [value_dict[key]["precision"]]
            recall = [value_dict[key]["recall"]]
            f1score = [value_dict[key]["f1-score"]]
            dicescore = [value_dict[key]["symm_dice"]]
            marker = value_dict[key]["marker"]
            x_pos = num + 1

            plt.scatter([x_pos - 1.5 * offset], precision,
                        color=COLOR_P, marker=marker, s=marker_size)
            plt.scatter([x_pos - 0.5 * offset], recall,
                        color=COLOR_R, marker=marker, s=marker_size)
            plt.scatter([x_pos + 0.5 * offset], f1score,
                        color=COLOR_F, marker=marker, s=marker_size)
            plt.scatter([x_pos + 1.5 * offset], dicescore,
                        color=COLOR_T, marker=marker, s=marker_size)

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=main_tick_size, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        plt.ylim(0, 1)
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

        color = [COLOR_P, COLOR_R, COLOR_F, COLOR_T]
        label = ["Precision", "Recall", "F1-score", "Dice's coefficient"]

        handles = [get_flatline_handle(c) for c in color]
        plt.legend(handles, label, loc=(0, 1), ncol=len(label), framealpha=1, frameon=False)

    else:
        raise ValueError("Unsupported mode for plotting.")

    plt.tight_layout()

    if save_path is not None:
        if ".png" in save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
        else:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def plot_legend_acc(save_path):
    """Plot common legend for figure 2c.

    Args:
        save_path: save path to save legend.
    """
    # Colors
    color = [COLOR_P, COLOR_R, COLOR_F]  # , COLOR_T
    label = ["Precision", "Recall", "F1-score"]

    handles = [get_flatline_handle(c) for c in color]
    legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for the figures for the OCT paper which show accuracy and Dice coefficients."
    )
    parser.add_argument("-f", "--figure_dir", type=str, required=True,
                        help="Output directory for plots.")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)
    plot_legend_acc(save_path=os.path.join(args.figure_dir, f"legend_acc.{FILE_EXTENSION}"))

    plot_model_acc(mode="precision", save_path=os.path.join(args.figure_dir, f"model_acc.{FILE_EXTENSION}"))
    plot_model_acc(mode="dice", save_path=os.path.join(args.figure_dir, f"model_dice.{FILE_EXTENSION}"))
    plot_model_acc(mode="both", save_path=os.path.join(args.figure_dir, f"model_performance.{FILE_EXTENSION}"))


if __name__ == "__main__":
    main()
