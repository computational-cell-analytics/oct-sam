import argparse
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from util import get_marker_handle, get_flatline_handle, export_legend

png_dpi = 300

FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"


def plot_model_acc(
    mode: str = "precision",
    plot: bool = False,
    save_path: Optional[str] = None,
    dict_set: str = "nnunet",
    show_legend: bool = False,
):
    value_dict_nnunet = {
        "nnunet-public": {
            "label": "nnU-Net public",
            "precision": 0.479,
            "recall": 0.5,
            "f1-score": 0.488,
            "marker": "o",
            "symm_dice": 0.631,
        },
        "nnunet-n001": {
            "label": "nnU-Net n1",
            "precision": 0.675,
            "recall": 0.631,
            "f1-score": 0.642,
            "marker": "o",
            "symm_dice": 0.673,
        },
        "nnunet-n005": {
            "label": "nnU-Net n5",
            "precision": 0.814,
            "recall": 0.827,
            "f1-score": 0.82,
            "marker": "o",
            "symm_dice": 0.783,
        },
        "nnunet-n010": {
            "label": "nnU-Net n10",
            "precision": 0.843,
            "recall": 0.875,
            "f1-score": 0.858,
            "marker": "o",
            "symm_dice": 0.786,
        },
        "nnunet-n025": {
            "label": "nnU-Net n25",
            "precision": 0.9,
            "recall": 0.914,
            "f1-score": 0.907,
            "marker": "o",
            "symm_dice": 0.824,
        },
        "nnunet-n050": {
            "label": "nnU-Net n50",
            "precision": 0.933,
            "recall": 0.936,
            "f1-score": 0.934,
            "marker": "o",
            "symm_dice": 0.847,
        },
        "nnunet-n100": {
            "label": "nnU-Net n100",
            "precision": 0.964,
            "recall": 0.964,
            "f1-score": 0.964,
            "marker": "o",
            "symm_dice": 0.871,
        },
        # "nnunet-n179": {
        #     "label": "nnU-Net n179",
        #     "precision": 0.949,
        #     "recall": 0.949,
        #     "f1-score": 0.949,
        #     "marker": "o",
        #     "symm_dice": 0.878,
        # },
        "nnunet-all": {
            "label": "nnU-Net all",
            "precision": 0.97,
            "recall": 0.97,
            "f1-score": 0.97,
            "marker": "o",
            "symm_dice": 0.88,
        },
    }

    value_dict_sam = {
        "sam-public": {
            "label": "octSAM public",
            "precision": 0.384,
            "recall": 0.383,
            "f1-score": 0.378,
            "marker": "v",
            "symm_dice": 0.525,
        },
        "sam-n001": {
            "label": "octSAM n1",
            "precision": 0.61,
            "recall": 0.532,
            "f1-score": 0.561,
            "marker": "v",
            "symm_dice": 0.656,
        },
        "sam-n005": {
            "label": "octSAM n5",
            "precision": 0.655,
            "recall": 0.56,
            "f1-score": 0.594,
            "marker": "v",
            "symm_dice": 0.644,
        },
        "sam-n010": {
            "label": "octSAM n10",
            "precision": 0.556,
            "recall": 0.489,
            "f1-score": 0.512,
            "marker": "v",
            "symm_dice": 0.62,
        },
        "sam-n025": {
            "label": "octSAM n25",
            "precision": 0.602,
            "recall": 0.531,
            "f1-score": 0.552,
            "marker": "v",
            "symm_dice": 0.63,
        },
        "sam-n050": {
            "label": "octSAM n50",
            "precision": 0.59,
            "recall": 0.54,
            "f1-score": 0.55,
            "marker": "v",
            "symm_dice": 0.628,
        },
        "sam-n100": {
            "label": "octSAM n100",
            "precision": 0.633,
            "recall": 0.589,
            "f1-score": 0.594,
            "marker": "v",
            "symm_dice": 0.63,
        },
        # "nnunet-n179": {
        #     "label": "nnU-Net n179",
        #     "precision": 0.949,
        #     "recall": 0.949,
        #     "f1-score": 0.949,
        #     "marker": "o",
        #     "symm_dice": 0.878,
        # },
        "sam-all": {
            "label": "octSAM all",
            "precision": 0.588,
            "recall": 0.494,
            "f1-score": 0.53,
            "marker": "v",
            "symm_dice": 0.623,
        },
    }

    # Convert setting labels to numerical x positions
    offset = 0.08  # horizontal shift for scatter separation

    # Plot
    tick_rotation = 45

    main_label_size = 20
    main_tick_size = 16
    marker_size = 200

    if dict_set == "nnunet":
        value_dicts = [value_dict_nnunet]
        labels = [item["label"] for _, item in value_dicts[0].items()]
    elif dict_set == "sam":
        value_dicts = [value_dict_sam]
        labels = [item["label"] for _, item in value_dicts[0].items()]
    else:
        value_dicts = [value_dict_nnunet, value_dict_sam]
        labels = [item["label"].split(" ")[1] for _, item in value_dicts[0].items()]

    fig_width = len(value_dicts[0]) * 2

    if mode == "precision":
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for value_dict in value_dicts:
            for num, key in enumerate(list(value_dict.keys())):
                precision = [value_dict[key]["precision"]]
                recall = [value_dict[key]["recall"]]
                f1score = [value_dict[key]["f1-score"]]
                marker = value_dict[key]["marker"]
                x_pos = num + 1

                plt.scatter([x_pos - offset], precision, label="Precision",
                            color=COLOR_P, marker=marker, s=marker_size)
                plt.scatter([x_pos],         recall, label="Recall",
                            color=COLOR_R, marker=marker, s=marker_size)
                plt.scatter([x_pos + offset], f1score, label="F1-score",
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
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for value_dict in value_dicts:
            for num, key in enumerate(list(value_dict.keys())):
                runtime = [value_dict[key]["symm_dice"]]
                x_pos = num + 1
                if runtime[0] is None:
                    continue
                marker = value_dict[key]["marker"]
                plt.scatter([x_pos], runtime, label="Dice", color=COLOR_T, marker=marker, s=marker_size)

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=16, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Dice's coefficient", fontsize=main_label_size)
        plt.ylim(0.4, 1)
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif mode == "both":
        fig, ax = plt.subplots(figsize=(fig_width, 5))
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for value_dict in value_dicts:
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

    else:
        raise ValueError("Unsupported mode for plotting.")

    if show_legend:
        linewidth = 9
        if mode == "dice":
            color = [COLOR_T]
            label = ["Dice's coefficient"]
            handles = [get_flatline_handle(c, linewidth=linewidth) for c in color]
            ncols = len(label)

        elif mode == "precision":
            color = [COLOR_P, COLOR_R, COLOR_F]
            label = ["Precision", "Recall", "F1-score"]
            handles = [get_flatline_handle(c, linewidth=linewidth) for c in color]
            ncols = len(label)

        elif mode == "both":
            color = [COLOR_P, COLOR_R, COLOR_F, COLOR_T]
            label1 = ["Precision", "Recall", "F1-score", "Dice's coefficient"]
            handles1 = [get_flatline_handle(c, linewidth=linewidth) for c in color]

            label2 = ["nnU-Net", "octSAM"]
            color = ["black" for _ in label2]
            marker = ["o", "v"]
            handles2 = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]

            handles = handles1 + handles2
            label = label1 + label2
            # variant for legend format
            # ncols = 4
            ncols = len(label)

        plt.legend(handles, label, loc=(0, 1), ncols=ncols, framealpha=1, frameon=False,
                   fontsize=14,
                   markerscale=2.5,
                   handlelength=4, handletextpad=0.5, columnspacing=2)

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
    parser = argparse.ArgumentParser(description="Generate plots for comparing retrain accuracy of OCT.")
    parser.add_argument("-f", "--figure_dir", type=str, help="Output directory for plots.",
                        default="./panels/supp_fig2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)
    plot_legend_acc(save_path=os.path.join(args.figure_dir, f"legend_acc.{FILE_EXTENSION}"))

    show_legend = True
    dict_set = "nnunet"
    plot_model_acc(
        mode="precision",
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_acc.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )
    plot_model_acc(
        mode="dice",
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_dice.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )

    dict_set = "sam"
    plot_model_acc(
        mode="precision",
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_acc.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )
    plot_model_acc(
        mode="dice",
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_dice.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )

    dict_set = "both"
    mode = "precision"
    plot_model_acc(
        mode=mode,
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_{mode}.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )
    mode = "dice"
    plot_model_acc(
        mode=mode,
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_{mode}.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )
    mode = "both"
    plot_model_acc(
        mode=mode,
        save_path=os.path.join(args.figure_dir, f"retrain_{dict_set}_{mode}.{FILE_EXTENSION}"),
        dict_set=dict_set,
        show_legend=show_legend,
    )


if __name__ == "__main__":
    main()
