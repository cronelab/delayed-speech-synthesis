import argparse
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from scipy.io import loadmat
from typing import Optional
from pathlib import Path


logger = logging.getLogger("gen_contamination_report.py")


def main(agg_dir: Path, analysis_dir: Path, output_dir: Optional[Path] = None):

    agg_files = sorted(agg_dir.glob("*.mat"))
    nb_days = len(agg_files)
    days = [re.match("[0-9]*_[0-9]*_[0-9]*", agg.name).group(0) for agg in agg_files]
    day_labels = [f"$D_{{{a}}}$" for a in range(nb_days)]

    surrogate_data = np.zeros(shape=(nb_days, 10_000), dtype=np.float32)
    dataset_measure = np.zeros(nb_days, dtype=np.float32)
    p_criterion = np.zeros(nb_days, dtype=np.float32)

    # Populate data
    for i, day in enumerate(days):
        mat = loadmat((analysis_dir / f"{day}_contamination_result.mat").as_posix(), simplify_cells=True)
        surrogate_data[i, :] = mat["out"]["surrogate_measures"]
        dataset_measure[i] = mat["out"]["dataset_measure"]
        p_criterion[i] = mat["out"]["criterion_value"]

    gs = grid_spec.GridSpec(nb_days, 1)
    fig = plt.figure(figsize=(11, 4))

    i = 0
    ax_objs = []
    for j, day in enumerate(days):
        # creating new axes object
        ax_objs.append(fig.add_subplot(gs[i:i + 1, 0:]))

        counts, bins = np.histogram(surrogate_data[j, :], bins=50)
        ax_objs[-1].stairs(counts, bins, color="#f0f0f0", lw=1.5)
        ax_objs[-1].stairs(counts, bins, fill=True, alpha=1, color="black")

        measure_color = "red" if p_criterion[j] < 0.05 else "limegreen"
        ax_objs[-1].axvline(x=dataset_measure[j], color=measure_color, lw=2)
        ax_objs[-1].axhline(y=0, color="black")

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(0.07, 0.185)
        ax_objs[-1].yaxis.set_tick_params(length=0)

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticks([0])
        ax_objs[-1].set_yticklabels([day_labels[j]])
        ax_objs[-1].tick_params(axis='y', colors='black')

        if i == nb_days-1:
            ax_objs[-1].set_xlabel("Mean diagonal value")
        else:
            ax_objs[-1].set_xticklabels([])
            ax_objs[-1].set_xticks([])

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        i += 1

    h1 = ax_objs[0].axhline(xmin=0, xmax=0, y=0, color="limegreen")
    h2 = ax_objs[0].axhline(xmin=0, xmax=0, y=0, color="red")
    h3 = ax_objs[0].axhline(xmin=0, xmax=0, y=0, color="black", lw=5)
    handles = [h1, h2, h3]

    ax_objs[0].legend(handles, ["p > 0.05", "p ≤ 0.05", "Permutation distribution"], ncol=3, frameon=False,
                      loc="upper right")
    gs.update(hspace=-0.7)
    plt.subplots_adjust(left=0.036, right=0.983, top=0.974, bottom=0.12)

    if output_dir is not None:
        plt.savefig(output_dir / "contamination_report.png", dpi=150)
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render contamination report supplementary figure")
    parser.add_argument("cont", help="Path to the contamination directory.")
    parser.add_argument("-o", "--out", help="Path to the output folder.")
    args = parser.parse_args()

    # initialize logging handler
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)-30s] [%(levelname)8s]: %(message)s',
                        datefmt='%d.%m.%y %H:%M:%S')

    agg_dir = Path(args.cont) / "aggregated_by_day"
    analysis_dir = Path(args.cont) / "analysis"

    logger.info(f"python gen_contamination_report.py {args.cont}" + args.out if args.out else "")
    main(agg_dir, analysis_dir, output_dir=Path(args.out) if args.out else None)
