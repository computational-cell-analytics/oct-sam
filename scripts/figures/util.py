import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

png_dpi = 300


def export_legend(legend, filename="legend.png"):
    legend.axes.axis("off")
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, bbox_inches=bbox, dpi=png_dpi)


def get_flatline_handle(color, linewidth=3):
    return Line2D([], [], lw=linewidth, color=color)


def get_marker_handle(color, marker, edgecolors=None):
    """Get function handle for plotting external legend without plot.
    """
    if edgecolors is None:
        return plt.plot([], [], marker=marker, color=color, ls="none")[0]
    else:
        return plt.plot([], [], marker=marker, markerfacecolor='none', markeredgecolor=edgecolors, ls="none")[0]
