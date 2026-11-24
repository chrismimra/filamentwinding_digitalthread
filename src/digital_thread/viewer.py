"""Tow viewer: click through the tows of a processed run (production display code).

Reads only the export package and the run's output folder (polarisation maps). Per revolution: the overlay
in the flattened image, the camera image of the frame with the crossing point
marked, and the revolution's row of the tow table.

Command: `uv run python -m digital_thread.viewer <run_folder> [<out_folder>]`
"""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons

from digital_thread import dataset, geometry


class TowViewer:
    def __init__(self, run_folder, out_folder=None, fig=None):
        """`fig`: an existing matplotlib figure to draw into (e.g. in a Qt canvas)."""
        self.run = dataset.read_run(run_folder)
        self.cal = dataset.read_calibration(self.run.calibration_folder)
        out = Path(out_folder) if out_folder is not None else self.run.folder
        self.out = out  # holds the polarisation maps written by processing
        self.export = out / dataset.EXPORT_FOLDER
        self.rows, _ = dataset.read_export(self.export)
        self.text = ""

        self.fig = fig if fig is not None else plt.figure(figsize=(14, 8))
        self.ax_overlay = self.fig.add_axes((0.18, 0.55, 0.8, 0.4))
        self.ax_camera = self.fig.add_axes((0.18, 0.03, 0.45, 0.42))
        self.ax_text = self.fig.add_axes((0.66, 0.05, 0.32, 0.45))
        self.ax_text.axis("off")
        labels = [f"rev {r['revolution']:03d}" for r in self.rows]
        self.list = RadioButtons(self.fig.add_axes((0.01, 0.3, 0.14, 0.6)), labels)
        self.list.on_clicked(lambda label: self.show(labels.index(label)))
        self.prev_button = Button(
            self.fig.add_axes((0.01, 0.15, 0.065, 0.08)), "Previous"
        )
        self.next_button = Button(self.fig.add_axes((0.085, 0.15, 0.065, 0.08)), "Next")
        self.prev_button.on_clicked(lambda _: self.previous())
        self.next_button.on_clicked(lambda _: self.next())
        self.show(0)

    def next(self):
        self.show((self.index + 1) % len(self.rows))

    def previous(self):
        self.show((self.index - 1) % len(self.rows))

    def show(self, index):
        """Show the tow table row `index` (its revolution may skip a missed one)."""
        self.index = index
        row = self.rows[index]
        k = self.revolution = row["revolution"]
        flat = dataset.read_flat(self.export, k)
        gh, gw = flat.tow.shape
        extent = (
            flat.origin[0],
            flat.origin[0] + flat.spacing * gw,
            flat.origin[1] + flat.spacing * gh,
            flat.origin[1],
        )
        self.ax_overlay.clear()
        self.ax_overlay.imshow(
            mpimg.imread(self.export / dataset.overlay_file(k)), extent=extent
        )
        self.ax_overlay.set(
            xlabel="axial position [mm]",
            ylabel=f"arc length at r = {flat.r_ref:.1f} mm [mm]",
            title=f"Revolution {k}: measured tow (orange), planned tow edges (blue)",
        )

        s0 = dataset.read_polarisation(self.out, f"rev{k:03d}").s0  # from Conversion
        self.ax_camera.clear()
        self.ax_camera.imshow(s0, cmap="gray")
        point = np.array([[row["x_mm"], row["y_mm"], row["z_mm"]]])
        if row["visible"]:  # a not-visible crossing point lies outside the view
            u, v = geometry.project(self.cal, point)
            self.ax_camera.plot(u, v, "o", mfc="none", mec="red", ms=14, mew=2)
        self.ax_camera.set(xlim=(0, s0.shape[1]), ylim=(s0.shape[0], 0))
        self.ax_camera.set_title("Camera image (S0) with the crossing point")
        self.ax_camera.axis("off")

        xyz = ", ".join(f"{row[c]:.1f}" for c in ("x_mm", "y_mm", "z_mm"))
        lines = [
            f"Revolution {k}",
            f"Frame time: {row['timestamp_s']:.1f} s",
            f"Crossing point: ({xyz}) mm",
            f"Planned angle: {row['planned_angle_deg']:.2f} deg",
        ]
        if not row["visible"]:
            lines.append("Crossing point not visible")
        elif np.isnan(row["delta_position_mm"]):
            lines.append("No measured tow at the crossing point")
        else:
            lines += [
                f"Delta position: {row['delta_position_mm']:+.2f} mm",
                f"Delta angle: {row['delta_angle_deg']:+.2f} deg",
                f"Tow cells used: {row['tow_cells']}",
            ]
        self.text = "\n".join(lines)
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(0, 1, self.text, va="top", family="monospace", fontsize=11)
        if self.list.value_selected != self.list.labels[index].get_text():
            self.list.set_active(index)
        self.fig.canvas.draw_idle()

    def close(self):
        plt.close(self.fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Click through the tows of a run.")
    parser.add_argument("run_folder")
    parser.add_argument("out_folder", nargs="?", default=None)
    args = parser.parse_args(argv)
    TowViewer(args.run_folder, args.out_folder)
    plt.show()


if __name__ == "__main__":
    main()
