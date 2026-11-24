"""Calibration (production, slice 2): intrinsics and camera pose from markers.

Marker model: each ArUco marker is a flat square of side `size_mm` centred on
the mesh point closest to its table centre, in the tangent plane of the mesh
normal there. A centre on an edge (e.g. the joint of two mandrel segments)
lies on the face most perpendicular to the mandrel axis, the winding surface.
Markers further than `MAX_OFF_MESH_MM` from the mesh are not used. Its top edge (the arrow on the print sheet) points along +Y, or
along +Z where the normal is along Y; the square is then pressed onto the mesh
(each corner moved to its closest mesh point). Corners are ordered as ArUco
reports them: top-left, top-right, bottom-right, bottom-left, seen from
outside the mandrel.
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from digital_thread import dataset, geometry, touch_down
from digital_thread.conversion import MAX_ADU
from digital_thread.dataset import Calibration

UP = np.array([0.0, 1.0, 0.0])
UP_ON_AXIAL_FACES = np.array([0.0, 0.0, 1.0])
AXIAL_FACE_COS = 0.9  # |normal . UP| above this: the face is treated as axial
MAX_OFF_MESH_MM = 5.0
EDGE_PROBE_MM = 0.5  # probes around a centre to find all faces meeting at an edge
EDGE_TIE_MM = 0.3  # faces this close to the nearest one count as meeting there


def _probes():
    d = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    d[13] = 0  # the centre itself
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.divide(d, norm, out=np.zeros_like(d), where=norm > 0) * EDGE_PROBE_MM


def surface_points(cal: Calibration, centres):
    """Surface point, unit normal and distance from the table centre per marker."""
    centres = np.asarray(centres, dtype=float)
    probes = _probes()
    pts, nrm = geometry.closest_on_mesh(
        cal, (centres[:, None] + probes[None]).reshape(-1, 3)
    )
    pts, nrm = pts.reshape(len(centres), -1, 3), nrm.reshape(len(centres), -1, 3)
    dist = np.linalg.norm(pts - centres[:, None], axis=2)
    near = dist <= dist.min(axis=1, keepdims=True) + EDGE_TIE_MM
    axial = np.where(near, np.abs(nrm @ cal.axis_dir), np.inf)
    pick = np.argmin(axial, axis=1)
    rows = np.arange(len(centres))
    return pts[rows, pick], nrm[rows, pick], dist[rows, pick]


def marker_frames(cal: Calibration, centres):
    """Centre on the mesh, right, up and normal unit vectors per marker, each (n, 3)."""
    centre, normal, _ = surface_points(cal, centres)
    up = np.where(
        (np.abs(normal @ UP) > AXIAL_FACE_COS)[:, None], UP_ON_AXIAL_FACES, UP
    )
    up = up - np.einsum("ij,ij->i", up, normal)[:, None] * normal
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    right = np.cross(up, normal)
    return centre, right, up, normal


def marker_corners(cal: Calibration, centres, size_mm):
    """Corners of each marker in the STEP frame, shape (n, 4, 3), ArUco order."""
    centre, right, up, _ = marker_frames(cal, centres)
    half = size_mm / 2
    signs = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])  # (right, up) per corner
    flat = (
        centre[:, None]
        + half * signs[None, :, :1] * right[:, None]
        + half * signs[None, :, 1:] * up[:, None]
    )
    on_mesh, _ = geometry.closest_on_mesh(cal, flat.reshape(-1, 3))
    return on_mesh.reshape(-1, 4, 3)


S0_FULL_SCALE = 2 * MAX_ADU  # S0 of a pixel with all four channels at full scale


def to_uint8(s0):
    """S0 map -> 8-bit image for OpenCV (fixed scale, NaN -> 0)."""
    s0 = np.nan_to_num(np.asarray(s0, dtype=float))
    return np.clip(s0 / S0_FULL_SCALE * 255, 0, 255).astype(np.uint8)


@dataclass
class Intrinsics:
    K: np.ndarray
    distortion: np.ndarray  # OpenCV order (k1, k2, p1, p2, k3)
    rms_px: float  # reprojection error
    used: int  # images in which the board was found


def intrinsic(images, inner_corners=(9, 6), tile_mm=40.0) -> Intrinsics:
    """Intrinsics of the super-pixel image from 8-bit S0 checkerboard images."""
    cols, rows = inner_corners
    board = np.zeros((cols * rows, 3), np.float32)
    board[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * tile_mm
    object_points, image_points = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    for image in images:
        found, corners = cv2.findChessboardCorners(image, inner_corners)
        if not found:
            continue
        corners = cv2.cornerSubPix(image, corners, (7, 7), (-1, -1), criteria)
        object_points.append(board)
        image_points.append(corners)
    if len(object_points) < 3:
        raise ValueError(f"board found in {len(object_points)} images, need 3")
    h, w = images[0].shape
    # k3 is fixed at 0: 12 views do not separate it from k1 and k2.
    rms, K, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, (w, h), None, None, flags=cv2.CALIB_FIX_K3
    )
    return Intrinsics(K, distortion.ravel()[:5], float(rms), len(object_points))


DICTIONARY = cv2.aruco.DICT_4X4_50
MIN_MARKERS = 1


@dataclass
class Pose:
    R: np.ndarray  # x_cam = R @ X + t
    t: np.ndarray
    centre: np.ndarray  # camera centre in the STEP frame
    marker_errors_px: dict[int, float]  # RMS reprojection error per marker
    rms_px: float
    used: int  # images in which markers were found


def detect_markers(image):
    """ArUco markers in an 8-bit image -> {id: (4, 2) corners in ArUco order}."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(DICTIONARY), params
    )
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners, strict=True)}


def position(images, ids, corners, K, distortion) -> Pose:
    """Camera pose in the STEP frame from marker images taken at the pin pulse.

    `ids`, `corners`: marker IDs and their (n, 4, 3) corners (`marker_corners`).
    The pose is solved per image and averaged (mean rotation, mean centre).
    """
    table = {int(i): c for i, c in zip(ids, corners, strict=True)}
    K, distortion = np.asarray(K, float), np.asarray(distortion, float)
    rotations, centres, observations = [], [], []
    for image in images:
        found = {i: c for i, c in detect_markers(image).items() if i in table}
        if len(found) < MIN_MARKERS:
            continue
        obj = np.concatenate([table[i] for i in found])
        img = np.concatenate(list(found.values()))
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, distortion, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, distortion, rvec, tvec)
        rot, _ = cv2.Rodrigues(rvec)
        rotations.append(rot)
        centres.append(-rot.T @ tvec.ravel())
        observations.append(found)
    if not rotations:
        raise ValueError("no markers of the marker table found in the images")
    rot = Rotation.from_matrix(np.array(rotations)).mean().as_matrix()
    centre = np.mean(centres, axis=0)
    t = -rot @ centre
    rvec, _ = cv2.Rodrigues(rot)
    squared = {}
    for found in observations:
        for i, c in found.items():
            projected, _ = cv2.projectPoints(table[i], rvec, t, K, distortion)
            err = np.sum((projected.reshape(4, 2) - c) ** 2, axis=1)
            squared.setdefault(i, []).extend(err)
    errors = {i: float(np.sqrt(np.mean(e))) for i, e in sorted(squared.items())}
    rms = float(np.sqrt(np.mean(np.concatenate(list(squared.values())))))
    return Pose(rot, t, centre, errors, rms, len(rotations))


POSITION_WARNING_MM = 50.0


def position_mismatch(pose: Pose, entered_centre) -> float | None:
    """Distance (mm) between calibrated and entered camera position if it exceeds
    `POSITION_WARNING_MM` (a gross setup error), else None."""
    distance = float(np.linalg.norm(pose.centre - np.asarray(entered_centre)))
    return distance if distance > POSITION_WARNING_MM else None


def body_angle_facing(cal: Calibration) -> float:
    """Body angle (deg) of the camera centre about the mandrel axis: the guide for
    the touch-down angle entry (body angle convention of the touch-down model)."""
    centre = geometry.camera_centre(cal)[None]
    angle = touch_down.body_angle(centre, cal.axis_point, cal.axis_dir)[0]
    return float(np.mod(angle, 360.0))


def save_bundle(folder, cal: Calibration, marker_table=None):
    """Write the calibration bundle; the marker table is copied into it."""
    dataset.write_calibration(folder, cal)
    if marker_table is not None:
        shutil.copyfile(marker_table, Path(folder) / dataset.MARKER_TABLE_FILE)


_MISSING_STEP = {  # Session field -> what to do first
    "intrinsics": "calibrate the optics first",
    "mesh": "load the mandrel first",
    "marker_table": "choose the marker table first",
    "marker_images": "capture the marker images first",
    "pose": "solve the position first",
    "touch_down_deg": "enter the touch-down angle first",
}


INTRINSIC_IMAGES = 12  # board positions of the guided capture
POSITION_IMAGES = 5  # revolutions averaged for the pose


@dataclass
class Session:
    """State of the Calibration tab; `bundle()` assembles the calibration bundle."""

    board_images: list = field(default_factory=list)
    intrinsics: Intrinsics | None = None
    mesh: tuple | None = None  # (vertices, faces, normals)
    axis_point: np.ndarray | None = None
    axis_dir: np.ndarray | None = None
    marker_table: Path | None = None
    marker_size_mm: float = 30.0
    marker_images: list = field(default_factory=list)
    pose: Pose | None = None
    touch_down_deg: float | None = None
    crop_polygon: np.ndarray | None = None

    def load_mandrel(self, source, axis_point=None, axis_dir=None):
        """Mesh from a STEP file, or mesh and axis from an existing bundle folder."""
        source = Path(source)
        if source.is_dir():
            cal = dataset.read_calibration(source)
            self.mesh = (cal.mesh_vertices, cal.mesh_faces, cal.mesh_normals)
            axis_point = cal.axis_point if axis_point is None else axis_point
            axis_dir = cal.axis_dir if axis_dir is None else axis_dir
        else:
            from digital_thread.step_import import step_to_mesh

            self.mesh = step_to_mesh(source)
        self.axis_point = np.asarray(axis_point, dtype=float)
        axis_dir = np.asarray(axis_dir, dtype=float)
        self.axis_dir = axis_dir / np.linalg.norm(axis_dir)

    def calibrate_optics(self):
        if not self.board_images:
            raise ValueError("capture or load the board images first")
        self.intrinsics = intrinsic(self.board_images)
        return self.intrinsics

    def _require(self, *names):
        for name in names:
            if getattr(self, name) is None or (
                isinstance(getattr(self, name), list) and not getattr(self, name)
            ):
                raise ValueError(_MISSING_STEP[name])

    def _calibration(self):
        """Calibration from the steps done so far (pose: identity until solved)."""
        self._require("intrinsics", "mesh")
        image = (self.marker_images or self.board_images or [None])[0]
        if image is None:
            raise ValueError("no calibration images yet")
        pose = self.pose
        return Calibration(
            image_size=(int(image.shape[1]), int(image.shape[0])),
            K=self.intrinsics.K,
            distortion=self.intrinsics.distortion,
            R=np.eye(3) if pose is None else pose.R,
            t=np.zeros(3) if pose is None else pose.t,
            axis_point=self.axis_point,
            axis_dir=self.axis_dir,
            pin_phase_deg=0.0,
            touch_down_deg=0.0,
            mesh_vertices=self.mesh[0],
            mesh_faces=self.mesh[1],
            mesh_normals=self.mesh[2],
        )

    def markers_off_mesh(self):
        """IDs of table markers further than MAX_OFF_MESH_MM from the mesh."""
        self._require("marker_table")
        ids, centres = dataset.read_markers(self.marker_table)
        _, _, distance = surface_points(self._calibration(), centres)
        return [int(i) for i in ids[distance > MAX_OFF_MESH_MM]]

    def solve_position(self):
        self._require("marker_table", "marker_images")
        cal = self._calibration()
        ids, centres = dataset.read_markers(self.marker_table)
        used = surface_points(cal, centres)[2] <= MAX_OFF_MESH_MM
        corners = marker_corners(cal, centres[used], self.marker_size_mm)
        self.pose = position(
            self.marker_images, ids[used], corners, cal.K, cal.distortion
        )
        return self.pose

    def body_angle_guide(self):
        self._require("pose")
        return body_angle_facing(self._calibration())

    def bundle(self) -> Calibration:
        self._require("pose", "touch_down_deg")
        cal = self._calibration()
        cal.touch_down_deg = float(self.touch_down_deg)
        cal.crop_polygon = self.crop_polygon
        cal.marker_size_mm = self.marker_size_mm
        cal.intrinsic_rms_px = self.intrinsics.rms_px
        cal.position_rms_px = self.pose.rms_px
        return cal

    def save(self, folder):
        save_bundle(folder, self.bundle(), self.marker_table)
