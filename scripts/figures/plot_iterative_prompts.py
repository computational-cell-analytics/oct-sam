import argparse
import json
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from oct_tools.analysis.eval_iterative_prompting import eval_iter_prompts_networks_multi
from oct_tools.analysis.eval_thickness_measurement import LAYER_MAPPING
from util import get_marker_handle, get_flatline_handle, export_legend

png_dpi = 300

FILE_EXTENSION = "png"

ITERATIVE_COLORS = {
    "pretrained-public": "#23CF72",
    "n001": "#00DDFA",
    "n005": "#23BBCF",
    "n010": "#3798A5",
    "n025": "#3D737A",
    "n050": "#354D50",
    "n100": "#2B3233",
}

LAYER_MARKER = {
    "RNFL": "o",
    "GCIPL": "v",
    "INL": "x",
    "OPL": "*",
    "ONL": "<",
    "EZ": "^",
    "RPE": "s",
}


def plot_legend_iter_prompts(
    save_path: str,
    plot_mode: str = "shapes",
):
    """Plot common legend for Figure 2c.

    Args:.
        save_path: save path to save legend.
        plot_mode: Plot either 'shapes' or 'colors' of data points.
    """
    if plot_mode == "shapes":
        # Shapes
        label = [key for key in list(LAYER_MARKER.keys())]
        marker = [item for _, item in LAYER_MARKER.items()]
        color = ["black" for _ in marker]

        handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]
        legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
        export_legend(legend, save_path)
        legend.remove()
        plt.close()

    elif plot_mode == "colors":
        # Colors
        label = [key for key in list(ITERATIVE_COLORS.keys())]
        color = [item for _, item in ITERATIVE_COLORS.items()]

        handles = [get_flatline_handle(c) for c in color]
        legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
        export_legend(legend, save_path)
        legend.remove()
        plt.close()

    else:
        raise ValueError("Choose either 'shapes' or 'colors' as plot_mode.")


def plot_iter_prompts(
    figure_dir: str,
    eval_dir: Optional[str] = None,
    networks: List[str] = [
        "oct-sam-pretrained-v2",
        "oct-sam-pre-v2-n001",
        "oct-sam-pre-v2-n005",
        "oct-sam-pre-v2-n010",
        "oct-sam-pre-v2-n025",
        "oct-sam-pre-v2-n050",
        "oct-sam-pre-v2-n100",
    ],
    eval_type: str = "box",
    plot_modes: List[str] = ["mean", "individual"],
    eval_dict: Optional[dict] = None,
    eval_json: Optional[str] = None,
):
    if eval_dict is None:
        if eval_dir is None:
            raise ValueError("Provide either eval_dict or eval_dir.")
        eval_dict = eval_iter_prompts_networks_multi(eval_dir, networks=networks)

    for plot_mode in plot_modes:
        save_path = os.path.join(figure_dir, f"iter_prompts_{eval_type}_{plot_mode}.{FILE_EXTENSION}")
        acc_type = "f1-score"

        colors = [item for _, item in ITERATIVE_COLORS.items()]
        markers = [item for _, item in LAYER_MARKER.items()]

        offset = 0.08
        plt.grid(axis="y", linestyle="solid", alpha=0.5, zorder=0)

        mean_values = {}
        for num, (key, items) in enumerate(eval_dict.items()):
            layer_indexes = list(items.keys())
            layer_index = 1

            iterations = [0, 1, 2, 3]
            mean_values[key] = {}
            for iit in iterations:
                data_y = [items[layer_index][eval_type][iit][acc_type] for layer_index in layer_indexes]
                mean_values[key][iit] = np.mean(data_y)

            for enum, layer_index in enumerate(layer_indexes):
                iterations = list(items[layer_index][eval_type].keys())
                data_x = [i + 1 for i in list(items[layer_index][eval_type].keys())]
                data_x_offset = [x_pos - len(colors) // 2 * offset + offset * num for x_pos in data_x]
                data_y = [items[layer_index][eval_type][i][acc_type] for i in iterations]
                if plot_mode == "individual":
                    plt.scatter(data_x_offset, data_y, color=colors[num], marker=markers[enum], zorder=2)

            mean_values_network = [mean_values[key][i] for i in iterations]
            if plot_mode == "mean":
                ylim_low = 0.4
                ylim_up = 0.7
                interval = 0.05
                y_ticks = list(np.arange(ylim_low, ylim_up + interval, interval))
                plt.ylim(ylim_low, ylim_up)
                plt.yticks(y_ticks)
                plt.scatter(data_x, mean_values_network, color=colors[num], marker="P", s=80, zorder=2)
            else:
                plt.ylim(-0.05, 1.05)

        label_size = 18
        plt.xticks(data_x)
        plt.xlabel("Iterations", fontsize=label_size)
        plt.ylabel("F1-Score", fontsize=label_size)

        if eval_json is not None:
            with open(eval_json, "w") as f:
                json.dump(eval_dict, f, indent='\t', separators=(',', ': '))

        if ".png" in save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
        else:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        plt.close()


def _export_df_from_json(
    input_json: str,
    output_folder: str,
    iterations: List[int] = [0, 1, 2, 3],
):
    """Utility function for the evaluation of iterative prompting of OCT-SAM.
    The parameters are read from a JSON dictionary and exported to TSV and Excel format.

    Args:
        input_json: JSON dictionary with parameters of the iterative prompting of OCT-SAM.
        output_folder: Output folder.
        iterations: Iterations of iterative prompting.
    """
    with open(input_json, 'r') as myfile:
        data = myfile.read()
    params = json.loads(data)

    for table_type in ["box", "point"]:
        dict_list = []
        for key, item in params.items():
            for value_type in ["precision", "recall", "f1-score"]:
                for iteration in iterations:
                    dict_tmp = {}
                    dict_tmp["Model"] = key
                    dict_tmp["Iteration"] = iteration + 1
                    dict_tmp["Value"] = value_type
                    for layer_key, layer in item.items():
                        dict_tmp[LAYER_MAPPING[int(layer_key)]] = layer[table_type][str(iteration)][value_type]
                    dict_list.append(dict_tmp)

        df = pd.DataFrame(dict_list)
        df = df.round(3)

        output_df = os.path.join(output_folder, f"OCT-SAM_iterative-prompts_{table_type}.tsv")
        df.to_csv(output_df, sep="\t", index=False)
        output_excel = os.path.join(output_folder, f"OCT-SAM_iterative-prompts_{table_type}.xlsx")
        df.to_excel(output_excel, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots for the figures for the OCT paper which show accuracy and Dice coefficients."
    )
    parser.add_argument("-f", "--figure_dir", type=str, required=True,
                        help="Output directory for plots.")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    plot_mode = "shapes"
    save_path = os.path.join(args.figure_dir, f"legend_iter_prompts_{plot_mode}.{FILE_EXTENSION}")
    plot_legend_iter_prompts(save_path, plot_mode=plot_mode)

    plot_mode = "colors"
    save_path = os.path.join(args.figure_dir, f"legend_iter_prompts_{plot_mode}.{FILE_EXTENSION}")
    plot_legend_iter_prompts(save_path, plot_mode=plot_mode)

    eval_dir = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/eval_interactive"
    networks = [
        "oct-sam-pretrained-v2",
        "oct-sam-pre-v2-n001",
        "oct-sam-pre-v2-n005",
        "oct-sam-pre-v2-n010",
        "oct-sam-pre-v2-n025",
        "oct-sam-pre-v2-n050",
        "oct-sam-pre-v2-n100",
    ]
    eval_dict = eval_iter_prompts_networks_multi(eval_dir, networks=networks)
    eval_json = os.path.join(args.figure_dir, "OCT-SAM_iterative-prompts.json")
    plot_iter_prompts(figure_dir=args.figure_dir, eval_dict=eval_dict, eval_type="box", eval_json=eval_json)
    plot_iter_prompts(figure_dir=args.figure_dir, eval_dict=eval_dict, eval_type="point")


if __name__ == "__main__":
    main()
