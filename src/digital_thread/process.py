"""Process a finished run (slices 4-7) into an export package.

Command: `uv run python -m digital_thread.process <run_folder> [<out_folder>]`
"""

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from digital_thread import conversion, dataset, geometry, mapping, overlay, tow_matching

CONVENTIONS = {
    "frame": "STEP frame of the mandrel model (origin and axes of the STEP file)",
    "units": "mm and degrees",
    "fibre_angle": "angle between fibre and mandrel axis in the local surface plane, "
    "positive towards the surface direction of mandrel rotation, in (-90, 90]",
    "crossing_point": "where the revolution's planned path crosses the plane through "
    "the camera centre and the mandrel axis, on the camera side",
    "delta_position": "signed distance of the measured tow centre line from the planned "
    "crossing point, oriented along n x t (n outward normal, t planned direction of "
    "travel)",
    "delta_angle": "direction of the measured tow centre line minus planned fibre angle",
    "flat_grid": "flattened image: rows = arc length r_ref * body angle, columns = axial "
    "position, both from the given origin with the given spacing",
}


def run_geometry(cal, out):
    """Run geometry from the output folder if present, else computed and written."""
    if (out / dataset.GEOMETRY_FILE).exists():
        return dataset.read_geometry(out)
    geo = geometry.compute(cal)
    dataset.write_geometry(out, geo)
    return geo


def process_run(run_folder, out_folder=None, progress=None) -> Path:
    """Conversion, Mapping, Tow matching and export of a finished run.
    All derived files go to `out_folder` (default: the run folder).
    `progress(done, total)` is called after each frame is mapped."""
    run = dataset.read_run(run_folder)
    cal = dataset.read_calibration(run.calibration_folder)
    out = Path(out_folder) if out_folder is not None else run.folder
    out.mkdir(parents=True, exist_ok=True)
    geo = run_geometry(cal, out)

    fibre = {}
    for done, frame in enumerate(run.frames, start=1):
        maps = conversion.convert(
            dataset.read_frame(run, frame.name), run.polar_pattern
        )
        dataset.write_polarisation(out, frame.name, maps)
        fibre[frame.name] = mapping.fibre_angles(maps, geo, cal, run.processing)
        dataset.write_fibre(out, frame.name, fibre[frame.name])
        if progress is not None:
            progress(done, len(run.frames))

    planned = geometry.planned_path(
        cal, dataset.read_path(run.folder / run.tow_file), run.tow_width_mm
    )
    rows = tow_matching.tow_table(run, cal, geo, planned, fibre)
    dataset.write_tow_table(out, rows)

    flats, overlays = {}, {}
    for row, before, frame in zip(rows, run.frames[:-1], run.frames[1:], strict=True):
        k = row["revolution"]
        before = before.name
        tow = tow_matching.measured_tow(fibre[before], fibre[frame.name], geo)
        flats[k] = dataset.FlatRevolution(
            fibre_deg=mapping.flatten(fibre[frame.name], geo),
            tow=tow,
            origin=geo.flat_origin,
            spacing=geo.flat_spacing,
            r_ref=geo.r_ref,
        )
        overlays[k] = overlay.overlay_image(
            fibre[frame.name], tow, planned, k, cal, geo, run.tow_width_mm
        )
        row["timestamp_s"] = frame.timestamp_s

    meta = {
        "run_id": run.run_id,
        # The STEP file name is not stored in the bundle; the part is named by
        # the folder holding the calibration bundle.
        "part": run.calibration_folder.parent.name,
        "calibration": run.calibration,
        "tow_file": run.tow_file,
        "tow_width_mm": run.tow_width_mm,
        "processing": asdict(run.processing),
        "camera": {
            "R": np.asarray(cal.R).tolist(),
            "t": np.asarray(cal.t).tolist(),
            "centre": geometry.camera_centre(cal).tolist(),
        },
        "exported": datetime.now(UTC).isoformat(timespec="seconds"),
        "conventions": CONVENTIONS,
    }
    export = out / dataset.EXPORT_FOLDER
    if export.exists():  # re-processing replaces the whole package
        for old_file in export.iterdir():
            # Files only: OneDrive may refuse to remove the folder itself.
            if old_file.is_file():
                old_file.unlink()
    dataset.write_export(export, rows, meta, flats, overlays)
    return export


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folder")
    parser.add_argument("out_folder", nargs="?", default=None)
    args = parser.parse_args(argv)
    print(process_run(args.run_folder, args.out_folder))


if __name__ == "__main__":
    main()
