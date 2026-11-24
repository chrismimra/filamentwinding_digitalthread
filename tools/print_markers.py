"""A4 print sheet of ArUco markers for the position calibration.

Usage: `uv run python tools/print_markers.py demoData/marker_locations.csv
[--size 30] [--out demoData/markers_A4.pdf]`

The marker IDs are the point numbers of the marker file. Each marker is drawn
at its exact size (black square including the 1-bit border) with its ID and an
arrow at its top edge. Print at 100 % (no "fit to page") and check the 100 mm
scale bar with a ruler.
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

A4_MM = (210.0, 297.0)
MARGIN_MM = 10.0
DICTIONARY = cv2.aruco.DICT_4X4_50
MM_PER_INCH = 25.4


def marker_ids(marker_file):
    ids = []
    for line in Path(marker_file).read_text(encoding="utf-8").splitlines()[1:]:
        first = line.split(",")[0].strip()
        if first:
            ids.append(int(float(first)))
    return ids


def print_sheet(ids, size_mm, out):
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY)
    width, height = A4_MM
    cell_w, cell_h = size_mm + 8.0, size_mm + 14.0
    cols = int((width - 2 * MARGIN_MM) // cell_w)
    rows = int((height - 2 * MARGIN_MM - 20.0) // cell_h)
    if len(ids) > cols * rows:
        raise SystemExit(f"{len(ids)} markers do not fit on one page at {size_mm} mm")

    fig = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    for n, marker_id in enumerate(ids):
        r, c = divmod(n, cols)
        x = MARGIN_MM + c * cell_w + (cell_w - size_mm) / 2
        y = MARGIN_MM + 8.0 + r * cell_h
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, 600, borderBits=1)
        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
            interpolation="nearest",
            extent=(x, x + size_mm, y + size_mm, y),
        )
        ax.annotate(
            "",
            xy=(x + size_mm / 2, y - 6.0),
            xytext=(x + size_mm / 2, y - 1.0),
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "0.4"},
        )
        ax.text(
            x + size_mm / 2,
            y + size_mm + 4.0,
            f"ID {marker_id}",
            ha="center",
            va="center",
            fontsize=8,
        )
    y_bar = height - MARGIN_MM - 6.0
    ax.plot([MARGIN_MM, MARGIN_MM + 100.0], [y_bar, y_bar], color="black", lw=1.2)
    for x_tick in (MARGIN_MM, MARGIN_MM + 100.0):
        ax.plot([x_tick, x_tick], [y_bar - 2, y_bar + 2], color="black", lw=1.2)
    ax.text(
        MARGIN_MM,
        y_bar + 6.0,
        f"100 mm - print at 100 %. ArUco DICT_4X4_50, marker {size_mm:g} mm incl. "
        "border. Arrow = top edge: points along +Y (mandrel axis).",
        fontsize=6.5,
        va="center",
    )
    fig.savefig(out, dpi=600)
    plt.close(fig)
    return np.array(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("marker_file")
    parser.add_argument("--size", type=float, default=30.0, help="marker size in mm")
    parser.add_argument("--out", default="demoData/markers_A4.pdf")
    args = parser.parse_args()
    ids = print_sheet(marker_ids(args.marker_file), args.size, args.out)
    print(f"{len(ids)} markers ({ids.min()}-{ids.max()}) written to {args.out}")


if __name__ == "__main__":
    main()
