"""A3 print sheet of the checkerboard for the intrinsic calibration.

Usage: `uv run python tools/print_board.py [--out demoData/board_A3.pdf]`

10 x 7 tiles of 40 mm (9 x 6 inner corners, as the Calibration tab expects),
A3 landscape. Print at 100 % (no "fit to page"), check that 10 tiles measure
400 mm, and glue the sheet flat onto a rigid board.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digital_thread.calibration import intrinsic

A3_MM = (420.0, 297.0)
MM_PER_INCH = 25.4
TILES = (10, 7)
TILE_MM = intrinsic.__defaults__[1]  # 40 mm, as the calibration assumes


def print_sheet(out):
    width, height = A3_MM
    cols, rows = TILES
    x0 = (width - cols * TILE_MM) / 2
    y0 = (height - rows * TILE_MM) / 2
    fig = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (x0 + c * TILE_MM, y0 + r * TILE_MM),
                        TILE_MM,
                        TILE_MM,
                        color="black",
                        lw=0,
                    )
                )
    ax.text(
        x0,
        y0 + rows * TILE_MM + 4.0,
        f"{cols} x {rows} tiles of {TILE_MM:g} mm: print at 100 %, "
        f"{cols} tiles = {cols * TILE_MM:g} mm.",
        fontsize=6.5,
        va="center",
    )
    fig.savefig(out)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demoData/board_A3.pdf")
    args = parser.parse_args()
    print_sheet(args.out)
    print(f"Checkerboard written to {args.out}")


if __name__ == "__main__":
    main()
