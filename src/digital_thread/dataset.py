"""Dataset I/O: the file interface shared by all slices.

A calibration bundle is a folder with `calibration.json` and `mesh.npz`.
A run folder holds `run.json` and one compressed raw frame per entry of the
frame log (`<name>_raw.npz`, key `raw`, uint16, 3000 x 4096). Conversion adds
the polarisation maps per frame (`<name>_polarisation.npz`, keys `s0`, `dolp`,
`aolp`, float32, 1500 x 2048).

Intrinsics refer to the super-pixel image (2048 x 1500), the resolution of
S0, DoLP and AoLP. The pose maps world (STEP frame, mm) to camera
coordinates: x_cam = R @ X + t. The axis direction is oriented so that the
mandrel turns counter-clockwise about it.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    image_size: tuple[int, int]  # (width, height) in super-pixels
    K: np.ndarray
    distortion: np.ndarray  # OpenCV order (k1, k2, p1, p2, k3)
    R: np.ndarray
    t: np.ndarray
    axis_point: np.ndarray
    axis_dir: np.ndarray
    pin_phase_deg: float
    touch_down_deg: float
    mesh_vertices: np.ndarray  # (n, 3) mm, STEP frame
    mesh_faces: np.ndarray  # (m, 3) vertex indices
    mesh_normals: np.ndarray  # (n, 3) outward unit normals per vertex


_MESH_FIELDS = ("mesh_vertices", "mesh_faces", "mesh_normals")


def write_calibration(folder, cal: Calibration):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "image_size": list(cal.image_size),
        "K": np.asarray(cal.K).tolist(),
        "distortion": np.asarray(cal.distortion).tolist(),
        "R": np.asarray(cal.R).tolist(),
        "t": np.asarray(cal.t).tolist(),
        "axis_point": np.asarray(cal.axis_point).tolist(),
        "axis_dir": np.asarray(cal.axis_dir).tolist(),
        "pin_phase_deg": float(cal.pin_phase_deg),
        "touch_down_deg": float(cal.touch_down_deg),
    }
    (folder / "calibration.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        folder / "mesh.npz",
        vertices=cal.mesh_vertices,
        faces=cal.mesh_faces,
        normals=cal.mesh_normals,
    )


def read_calibration(folder) -> Calibration:
    folder = Path(folder)
    meta = json.loads((folder / "calibration.json").read_text(encoding="utf-8"))
    with np.load(folder / "mesh.npz") as mesh:
        return Calibration(
            image_size=tuple(meta["image_size"]),
            K=np.array(meta["K"]),
            distortion=np.array(meta["distortion"]),
            R=np.array(meta["R"]),
            t=np.array(meta["t"]),
            axis_point=np.array(meta["axis_point"]),
            axis_dir=np.array(meta["axis_dir"]),
            pin_phase_deg=meta["pin_phase_deg"],
            touch_down_deg=meta["touch_down_deg"],
            mesh_vertices=mesh["vertices"],
            mesh_faces=mesh["faces"],
            mesh_normals=mesh["normals"],
        )


def frame_file(name: str) -> str:
    """Raw frame file name: `reference_raw.npz`, `rev000_raw.npz`, ..."""
    return f"{name}_raw.npz"


@dataclass
class Frame:
    name: str  # "reference" or "rev000", "rev001", ...
    file: str  # raw frame file in the run folder
    timestamp_s: float


@dataclass
class Run:
    run_id: str
    calibration: str  # calibration bundle folder, relative to the run folder
    tow_file: str | None
    tow_width_mm: float
    polar_pattern: tuple[
        int, int, int, int
    ]  # (top-left, top-right, bottom-left, bottom-right)
    camera_settings: dict
    nominal_period_s: float
    frames: list[Frame] = field(default_factory=list)
    folder: Path | None = None  # set when read

    @property
    def calibration_folder(self) -> Path:
        return (self.folder / self.calibration).resolve()


def write_run(folder, run: Run):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    meta = asdict(run)
    meta.pop("folder")
    meta["polar_pattern"] = list(run.polar_pattern)
    (folder / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_run(folder) -> Run:
    folder = Path(folder)
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    meta["frames"] = [Frame(**f) for f in meta["frames"]]
    meta["polar_pattern"] = tuple(meta["polar_pattern"])
    return Run(**meta, folder=folder)


def write_frame(folder, name: str, raw: np.ndarray):
    np.savez_compressed(Path(folder) / frame_file(name), raw=raw.astype(np.uint16))


def read_frame(run: Run, name: str) -> np.ndarray:
    frame = next(f for f in run.frames if f.name == name)
    with np.load(run.folder / frame.file) as data:
        return data["raw"]


@dataclass
class PolarisationMaps:
    """S0, DoLP and AoLP (deg, image angle) of one frame per super-pixel, float32."""

    s0: np.ndarray
    dolp: np.ndarray
    aolp: np.ndarray


def polarisation_file(name: str) -> str:
    """Polarisation maps file name: `reference_polarisation.npz`, ..."""
    return f"{name}_polarisation.npz"


def write_polarisation(folder, name: str, maps: PolarisationMaps):
    """Write the polarisation maps of frame `name` into `folder`.

    Takes a folder rather than a Run: production passes the run folder, the
    notebook its save location, so the committed synthetic run stays unchanged.
    """
    np.savez_compressed(
        Path(folder) / polarisation_file(name),
        s0=maps.s0.astype(np.float32),
        dolp=maps.dolp.astype(np.float32),
        aolp=maps.aolp.astype(np.float32),
    )


def read_polarisation(folder, name: str) -> PolarisationMaps:
    """Read the polarisation maps of frame `name` from `folder`."""
    with np.load(Path(folder) / polarisation_file(name)) as data:
        return PolarisationMaps(data["s0"], data["dolp"], data["aolp"])


PATH_HEADER = "Point number,X-coordinate (mm),Y-coordinate (mm),Z-coordinate (mm)"


def read_path(file) -> np.ndarray:
    """Tow placement file: CSV with point number and X, Y, Z in mm -> (n, 3)."""
    return np.loadtxt(file, delimiter=",", skiprows=1, usecols=(1, 2, 3), ndmin=2)


def write_path(file, points: np.ndarray):
    rows = np.column_stack([np.arange(len(points)), points])
    np.savetxt(
        file,
        rows,
        delimiter=",",
        header=PATH_HEADER,
        comments="",
        fmt=["%d"] + ["%.9f"] * 3,
    )
