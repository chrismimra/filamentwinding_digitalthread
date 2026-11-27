"""A3 print sheet of the ChArUco board for the intrinsic calibration.

Usage: `uv run python tools/print_board.py [--out demoData/board_A3.pdf]`

10 x 7 tiles of 40 mm with 30 mm ArUco markers (DICT_5X5_50) in the white
tiles, as the Calibration tab expects; A3 landscape. Print at 100 % (no "fit to
page"), check that 10 tiles measure 400 mm, and glue the sheet flat onto a
rigid board.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from digital_thread import calibration

A3_MM = (420.0, 297.0)
MM_PER_INCH = 25.4
PX_PER_MM = 20  # resolution of the board image (about 500 dpi)


def print_sheet(out):
    width, height = A3_MM
    cols, rows = calibration.BOARD_TILES
    tile = calibration.BOARD_TILE_MM
    board_w, board_h = cols * tile, rows * tile
    image = calibration.BOARD.generateImage(
        (int(board_w * PX_PER_MM), int(board_h * PX_PER_MM)), marginSize=0
    )
    x0, y0 = (width - board_w) / 2, (height - board_h) / 2
    fig = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.axis("off")
    ax.imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=255,
        interpolation="nearest",
        extent=(x0, x0 + board_w, y0 + board_h, y0),
    )
    ax.text(
        x0,
        y0 + board_h + 4.0,
        f"ChArUco {cols} x {rows} tiles of {tile:g} mm, markers "
        f"{calibration.BOARD_MARKER_MM:g} mm (DICT_5X5_50): print at 100 %, "
        f"{cols} tiles = {board_w:g} mm.",
        fontsize=6.5,
        va="center",
    )
    fig.savefig(out, dpi=500)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="demoData/board_A3.pdf")
    args = parser.parse_args()
    print_sheet(args.out)
    print(f"ChArUco board written to {args.out}")


if __name__ == "__main__":
    main()
