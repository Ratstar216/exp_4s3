from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "AR Marker Tracker"
DEFAULT_MARKER_COLORS = [
    (0, 255, 0),
    (255, 255, 0),
    (0, 180, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
]
UNOWNED_TERRITORY = -1
ITEM_MUSHROOM = "mushroom"
CALIBRATION_CAMERA = "camera"
CALIBRATION_PROJECTOR = "projector"

ARUCO_DICTIONARIES = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "4x4_1000": cv2.aruco.DICT_4X4_1000,
    "5x5_50": cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "5x5_1000": cv2.aruco.DICT_5X5_1000,
    "6x6_50": cv2.aruco.DICT_6X6_50,
    "6x6_100": cv2.aruco.DICT_6X6_100,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "6x6_1000": cv2.aruco.DICT_6X6_1000,
    "7x7_50": cv2.aruco.DICT_7X7_50,
    "7x7_100": cv2.aruco.DICT_7X7_100,
    "7x7_250": cv2.aruco.DICT_7X7_250,
    "7x7_1000": cv2.aruco.DICT_7X7_1000,
    "aruco_original": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass
class SupportItem:
    item_type: str
    position: tuple[int, int]
    radius: int


@dataclass
class BuffState:
    size_multiplier: float
    expires_at: float


@dataclass
class CalibrationState:
    camera_points: list[tuple[int, int]] = field(default_factory=list)
    projector_points: list[tuple[int, int]] = field(default_factory=list)
    mode: str | None = None
    camera_dirty: bool = False
    projector_dirty: bool = False


@dataclass
class MouseState:
    support_items: list[SupportItem]
    calibration: CalibrationState
    item_radius: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track ArUco AR markers in real time from a PC camera."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index passed to OpenCV VideoCapture.",
    )
    parser.add_argument(
        "--camera-source",
        help=(
            "Camera source passed to OpenCV VideoCapture. Use this for an IP/RTSP/"
            "HTTP camera URL or a video file. Numeric values are treated as camera indexes."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe local OpenCV camera indexes and exit.",
    )
    parser.add_argument(
        "--camera-probe-limit",
        type=int,
        default=10,
        help="Number of camera indexes to check with --list-cameras.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Requested camera frame width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Requested camera frame height.",
    )
    parser.add_argument(
        "--dictionary",
        choices=sorted(ARUCO_DICTIONARIES),
        default="4x4_50",
        help="ArUco dictionary used by the printed marker.",
    )
    parser.add_argument(
        "--target-id",
        type=int,
        action="append",
        help="Only report and draw this marker ID. Can be passed more than once.",
    )
    parser.add_argument(
        "--robot-ids",
        type=int,
        nargs="+",
        help="Marker IDs for the robots to track, for example: --robot-ids 0 1.",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=600,
        help="Maximum number of center points kept per marker trajectory.",
    )
    parser.add_argument(
        "--min-trajectory-distance",
        type=float,
        default=2.0,
        help="Minimum pixel movement before adding a new trajectory point.",
    )
    parser.add_argument(
        "--trajectory-thickness",
        type=int,
        default=16,
        help="Trajectory line thickness in pixels.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.25,
        help="Seconds between console position updates.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Print tracking data without opening a preview window.",
    )
    parser.add_argument(
        "--generate-marker",
        type=int,
        metavar="ID",
        help="Generate a marker image for the given ID and exit.",
    )
    parser.add_argument(
        "--marker-pixels",
        type=int,
        default=600,
        help="Generated marker body size in pixels, excluding the white margin.",
    )
    parser.add_argument(
        "--marker-margin",
        type=int,
        default=80,
        help="White margin around generated marker images in pixels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("marker.png"),
        help="Output path for --generate-marker.",
    )
    parser.add_argument(
        "--trajectory-output",
        type=Path,
        default=Path("trajectories.png"),
        help="Image path written when tracking stops.",
    )
    parser.add_argument(
        "--game-duration",
        type=float,
        default=180.0,
        help="Game duration in seconds. Set to 0 to disable the timer.",
    )
    parser.add_argument(
        "--item-radius",
        type=int,
        default=28,
        help="Radius in pixels for support item markers.",
    )
    parser.add_argument(
        "--item-pickup-radius",
        type=int,
        default=40,
        help="Pickup radius in pixels for support items.",
    )
    parser.add_argument(
        "--mushroom-duration",
        type=float,
        default=8.0,
        help="Seconds that a mushroom size boost remains active.",
    )
    parser.add_argument(
        "--mushroom-size-multiplier",
        type=float,
        default=1.6,
        help="Size multiplier applied while a mushroom boost is active.",
    )
    args = parser.parse_args()
    if args.trajectory_length < 2:
        parser.error("--trajectory-length must be at least 2")
    if args.min_trajectory_distance < 0:
        parser.error("--min-trajectory-distance must be 0 or greater")
    if args.trajectory_thickness < 1:
        parser.error("--trajectory-thickness must be at least 1")
    if args.camera_probe_limit < 1:
        parser.error("--camera-probe-limit must be at least 1")
    if args.game_duration < 0:
        parser.error("--game-duration must be 0 or greater")
    if args.item_radius < 1:
        parser.error("--item-radius must be at least 1")
    if args.item_pickup_radius < 1:
        parser.error("--item-pickup-radius must be at least 1")
    if args.mushroom_duration <= 0:
        parser.error("--mushroom-duration must be greater than 0")
    if args.mushroom_size_multiplier < 1:
        parser.error("--mushroom-size-multiplier must be at least 1")
    return args


def get_aruco_dictionary(name: str) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[name])


def create_detector(
    aruco_dictionary: cv2.aruco.Dictionary,
) -> cv2.aruco.ArucoDetector | None:
    if not hasattr(cv2.aruco, "ArucoDetector"):
        return None
    parameters = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dictionary, parameters)


def detect_markers(
    frame: np.ndarray,
    aruco_dictionary: cv2.aruco.Dictionary,
    detector: cv2.aruco.ArucoDetector | None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if detector is not None:
        corners, ids, _rejected = detector.detectMarkers(frame)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _rejected = cv2.aruco.detectMarkers(
            frame, aruco_dictionary, parameters=parameters
        )
    return corners, ids


def marker_center(corners: np.ndarray) -> tuple[int, int]:
    points = corners.reshape(4, 2)
    center = points.mean(axis=0)
    return int(center[0]), int(center[1])


def marker_heading_degrees(corners: np.ndarray) -> float:
    top_left, top_right = corners.reshape(4, 2)[:2]
    dx, dy = top_right - top_left
    return math.degrees(math.atan2(float(dy), float(dx)))


def marker_color(marker_id: int) -> tuple[int, int, int]:
    return DEFAULT_MARKER_COLORS[marker_id % len(DEFAULT_MARKER_COLORS)]


def order_quad_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = [
        pts[int(np.argmin(sums))],
        pts[int(np.argmin(diffs))],
        pts[int(np.argmax(sums))],
        pts[int(np.argmax(diffs))],
    ]
    return [(int(x), int(y)) for x, y in ordered]


def build_field_mask(frame_shape: tuple[int, int, int], points: list[tuple[int, int]]) -> np.ndarray:
    if len(points) != 4:
        raise ValueError("field mask requires four points")
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def format_time(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remainder = divmod(total_seconds, 60)
    return f"{minutes:02d}:{remainder:02d}"


def leader_summary(scores: dict[int, int]) -> str:
    if not scores:
        return "Leader: none"
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_score = sorted_scores[0][1]
    leaders = [marker_id for marker_id, score in sorted_scores if score == top_score]
    if len(leaders) > 1:
        leaders_text = ", ".join(f"Player {marker_id}" for marker_id in leaders)
        return f"Leader: tie ({leaders_text})"
    leader_id = leaders[0]
    trailing_id = sorted_scores[-1][0] if len(sorted_scores) > 1 else None
    if trailing_id is not None and trailing_id != leader_id:
        return f"Leader: Player {leader_id} | Trailing: Player {trailing_id}"
    return f"Leader: Player {leader_id}"


def active_size_multiplier(marker_id: int, buffs: dict[int, BuffState], now: float) -> float:
    buff = buffs.get(marker_id)
    if buff is None:
        return 1.0
    if buff.expires_at <= now:
        buffs.pop(marker_id, None)
        return 1.0
    return buff.size_multiplier


def prune_expired_buffs(buffs: dict[int, BuffState], now: float) -> None:
    expired_ids = [marker_id for marker_id, buff in buffs.items() if buff.expires_at <= now]
    for marker_id in expired_ids:
        buffs.pop(marker_id, None)


def remove_nearest_item(items: list[SupportItem], point: tuple[int, int], max_distance: float) -> None:
    if not items:
        return
    px, py = point
    closest_index = None
    closest_distance = max_distance
    for index, item in enumerate(items):
        dx = px - item.position[0]
        dy = py - item.position[1]
        distance = math.hypot(dx, dy)
        if distance <= closest_distance:
            closest_distance = distance
            closest_index = index
    if closest_index is not None:
        del items[closest_index]


def set_calibration_mode(calibration: CalibrationState, mode: str | None) -> None:
    calibration.mode = mode
    if mode == CALIBRATION_CAMERA:
        calibration.camera_points.clear()
    elif mode == CALIBRATION_PROJECTOR:
        calibration.projector_points.clear()


def active_calibration_points(calibration: CalibrationState) -> list[tuple[int, int]] | None:
    if calibration.mode == CALIBRATION_CAMERA:
        return calibration.camera_points
    if calibration.mode == CALIBRATION_PROJECTOR:
        return calibration.projector_points
    return None


def add_calibration_point(calibration: CalibrationState, point: tuple[int, int]) -> None:
    points = active_calibration_points(calibration)
    if points is None:
        return
    points.append(point)
    if len(points) >= 4:
        ordered = order_quad_points(points[:4])
        points[:] = ordered
        if calibration.mode == CALIBRATION_CAMERA:
            calibration.camera_dirty = True
        elif calibration.mode == CALIBRATION_PROJECTOR:
            calibration.projector_dirty = True
        calibration.mode = None


def handle_mouse_event(
    event: int,
    x: int,
    y: int,
    _flags: int,
    state: MouseState | None,
) -> None:
    if state is None:
        return
    point = (x, y)
    if state.calibration.mode is not None:
        if event == cv2.EVENT_LBUTTONDOWN:
            add_calibration_point(state.calibration, point)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = active_calibration_points(state.calibration)
            if points:
                points.pop()
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        state.support_items.append(SupportItem(ITEM_MUSHROOM, point, state.item_radius))
    elif event == cv2.EVENT_RBUTTONDOWN:
        remove_nearest_item(state.support_items, point, state.item_radius * 1.5)


def check_item_pickups(
    support_items: list[SupportItem],
    marker_positions: list[tuple[int, tuple[int, int]]],
    buffs: dict[int, BuffState],
    pickup_radius: float,
    now: float,
    mushroom_duration: float,
    mushroom_size_multiplier: float,
) -> None:
    if not support_items or not marker_positions:
        return
    remaining_items: list[SupportItem] = []
    for item in support_items:
        picked = False
        for marker_id, (center_x, center_y) in marker_positions:
            dx = center_x - item.position[0]
            dy = center_y - item.position[1]
            if math.hypot(dx, dy) <= pickup_radius:
                if item.item_type == ITEM_MUSHROOM and mushroom_duration > 0:
                    buffs[marker_id] = BuffState(
                        size_multiplier=mushroom_size_multiplier,
                        expires_at=now + mushroom_duration,
                    )
                picked = True
                break
        if not picked:
            remaining_items.append(item)
    support_items[:] = remaining_items


def render_territory_overlay(
    frame: np.ndarray,
    territory_owner: np.ndarray,
    marker_ids: set[int] | None,
    field_mask: np.ndarray | None,
) -> None:
    overlay = np.zeros_like(frame)
    if marker_ids:
        ids = marker_ids
    else:
        ids = {int(marker_id) for marker_id in np.unique(territory_owner)}
    for marker_id in ids:
        if marker_id == UNOWNED_TERRITORY:
            continue
        overlay[territory_owner == marker_id] = marker_color(marker_id)
    if field_mask is not None:
        overlay[field_mask == 0] = 0
        mask = (territory_owner != UNOWNED_TERRITORY) & (field_mask > 0)
    else:
        mask = territory_owner != UNOWNED_TERRITORY
    if not np.any(mask):
        return
    blended = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
    frame[mask] = blended[mask]


def draw_support_items(frame: np.ndarray, support_items: list[SupportItem]) -> None:
    for item in support_items:
        color = (0, 140, 255)
        cv2.circle(frame, item.position, item.radius, color, 2)
        cv2.circle(frame, item.position, max(2, item.radius // 4), color, -1)
        label_origin = (item.position[0] - 8, item.position[1] + 6)
        cv2.putText(
            frame,
            "M",
            label_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_info_panel(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    align_right: bool = False,
) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    padding = 12
    gap = 8
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    panel_width = max(width for width, _height in sizes) + padding * 2
    panel_height = sum(height for _width, height in sizes) + gap * (len(lines) - 1) + padding * 2
    x, y = origin
    if align_right:
        x -= panel_width
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y_cursor = y + padding
    for line, (_width, height) in zip(lines, sizes):
        y_cursor += height
        cv2.putText(
            frame,
            line,
            (x + padding, y_cursor),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y_cursor += gap


def draw_center_panel(frame: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    padding = 16
    gap = 10
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    panel_width = max(width for width, _height in sizes) + padding * 2
    panel_height = sum(height for _width, height in sizes) + gap * (len(lines) - 1) + padding * 2
    x = (frame.shape[1] - panel_width) // 2
    y = (frame.shape[0] - panel_height) // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y_cursor = y + padding
    for line, (width, height) in zip(lines, sizes):
        y_cursor += height
        x_text = x + (panel_width - width) // 2
        cv2.putText(
            frame,
            line,
            (x_text, y_cursor),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y_cursor += gap


def draw_calibration_guides(frame: np.ndarray, calibration: CalibrationState) -> None:
    if calibration.camera_points:
        points = np.array(calibration.camera_points, dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)
        for point in calibration.camera_points:
            cv2.circle(frame, point, 6, (0, 255, 255), -1)
    if calibration.projector_points:
        points = np.array(calibration.projector_points, dtype=np.int32)
        cv2.polylines(frame, [points], isClosed=True, color=(255, 200, 0), thickness=2)
        for point in calibration.projector_points:
            cv2.circle(frame, point, 6, (255, 200, 0), -1)
    if calibration.mode == CALIBRATION_CAMERA:
        cv2.putText(
            frame,
            "Camera calibration: click 4 corners",
            (12, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    elif calibration.mode == CALIBRATION_PROJECTOR:
        cv2.putText(
            frame,
            "Projector calibration: click 4 corners",
            (12, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 0),
            2,
            cv2.LINE_AA,
        )


def draw_game_over_panel(frame: np.ndarray, scores: dict[int, int]) -> None:
    lines = ["GAME OVER"]
    if scores:
        leader = leader_summary(scores).replace("Leader:", "Winner:")
        lines.append(leader)
        for marker_id, score in scores.items():
            lines.append(f"Player {marker_id}: {score} px")
    draw_center_panel(frame, lines)


def target_ids(args: argparse.Namespace) -> set[int] | None:
    ids = set(args.target_id or [])
    if args.robot_ids:
        ids.update(args.robot_ids)
    return ids or None


def should_add_trajectory_point(
    trajectory: list[tuple[int, int]],
    point: tuple[int, int],
    min_distance: float,
) -> bool:
    if not trajectory:
        return True

    last_x, last_y = trajectory[-1]
    dx = point[0] - last_x
    dy = point[1] - last_y
    return math.hypot(dx, dy) >= min_distance


def update_trajectory(
    trajectories: dict[int, list[tuple[int, int]]],
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int], int]],
    territory_owner: np.ndarray,
    marker_id: int,
    point: tuple[int, int],
    max_length: int,
    min_distance: float,
    thickness: int,
    field_mask: np.ndarray | None,
) -> None:
    trajectory = trajectories.setdefault(marker_id, [])
    if should_add_trajectory_point(trajectory, point, min_distance):
        if trajectory:
            start = trajectory[-1]
            paint_segments.append((marker_id, start, point, thickness))
            paint_territory(territory_owner, marker_id, start, point, thickness, field_mask)
        trajectory.append(point)
        if len(trajectory) > max_length:
            del trajectory[: len(trajectory) - max_length]
        total_segments = sum(max(0, len(points) - 1) for points in trajectories.values())
        if len(paint_segments) > total_segments:
            del paint_segments[: len(paint_segments) - total_segments]


def paint_territory(
    territory_owner: np.ndarray,
    marker_id: int,
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int,
    field_mask: np.ndarray | None,
) -> None:
    stroke = np.zeros(territory_owner.shape, dtype=np.uint8)
    cv2.line(stroke, start, end, 255, thickness, cv2.LINE_AA)
    if field_mask is not None:
        stroke = cv2.bitwise_and(stroke, stroke, mask=field_mask)
    territory_owner[stroke > 0] = marker_id


def territory_scores(
    territory_owner: np.ndarray,
    marker_ids: set[int] | None,
) -> dict[int, int]:
    owned = territory_owner[territory_owner != UNOWNED_TERRITORY]
    if owned.size == 0:
        return {marker_id: 0 for marker_id in sorted(marker_ids or [])}

    ids, counts = np.unique(owned, return_counts=True)
    scores = {int(marker_id): int(count) for marker_id, count in zip(ids, counts)}
    if marker_ids is not None:
        for marker_id in marker_ids:
            scores.setdefault(marker_id, 0)
    return dict(sorted(scores.items()))


def format_scores(scores: dict[int, int]) -> str:
    if not scores:
        return "territory: none"
    return "territory: " + " | ".join(
        f"id={marker_id} {score_px}px" for marker_id, score_px in scores.items()
    )


def draw_trajectories(
    frame: np.ndarray,
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int], int]],
    trajectories: dict[int, list[tuple[int, int]]],
    thickness: int,
) -> None:
    for marker_id, start, end, segment_thickness in paint_segments:
        cv2.line(
            frame,
            start,
            end,
            marker_color(marker_id),
            max(1, segment_thickness),
            cv2.LINE_AA,
        )

    for marker_id, trajectory in trajectories.items():
        if not trajectory:
            continue
        color = marker_color(marker_id)
        cv2.circle(frame, trajectory[-1], max(6, thickness // 2), color, -1)


def draw_territory_scores(frame: np.ndarray, scores: dict[int, int]) -> None:
    if not scores:
        return

    line_height = 30
    padding = 12
    panel_width = 260
    panel_height = padding * 2 + line_height * len(scores)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (12 + panel_width, 12 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, (marker_id, score_px) in enumerate(scores.items()):
        y = 12 + padding + 22 + line_height * index
        color = marker_color(marker_id)
        cv2.circle(frame, (32, y - 6), 8, color, -1)
        cv2.putText(
            frame,
            f"Player {marker_id}: {score_px} px",
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_marker_details(
    frame: np.ndarray,
    corners: np.ndarray,
    marker_id: int,
    fps: float,
    size_multiplier: float,
) -> None:
    center_x, center_y = marker_center(corners)
    heading = marker_heading_degrees(corners)
    points = corners.reshape(4, 2).astype(int)
    top_left, top_right, bottom_right, bottom_left = points
    color = marker_color(marker_id)

    outline_thickness = max(2, int(2 * size_multiplier))
    marker_radius = max(5, int(5 * size_multiplier))
    arrow_thickness = max(2, int(2 * size_multiplier))
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=outline_thickness)
    cv2.circle(frame, (center_x, center_y), marker_radius, color, -1)
    cv2.arrowedLine(
        frame,
        tuple(top_left),
        tuple(top_right),
        color,
        arrow_thickness,
        tipLength=0.25,
    )

    label = f"id={marker_id} x={center_x} y={center_y} heading={heading:.1f} fps={fps:.1f}"
    text_origin = (int(bottom_left[0]), int(bottom_left[1]) + 24)
    cv2.putText(
        frame,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def generate_marker(args: argparse.Namespace) -> None:
    aruco_dictionary = get_aruco_dictionary(args.dictionary)
    if args.generate_marker >= len(aruco_dictionary.bytesList):
        raise ValueError(
            f"Marker ID {args.generate_marker} is outside dictionary "
            f"{args.dictionary}, which has IDs 0-{len(aruco_dictionary.bytesList) - 1}"
        )

    marker = cv2.aruco.generateImageMarker(
        aruco_dictionary,
        args.generate_marker,
        args.marker_pixels,
    )
    if args.marker_margin > 0:
        marker = cv2.copyMakeBorder(
            marker,
            args.marker_margin,
            args.marker_margin,
            args.marker_margin,
            args.marker_margin,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), marker):
        raise RuntimeError(f"Failed to write marker image: {args.output}")
    print(f"Wrote marker {args.generate_marker} to {args.output}")


def parse_camera_source(camera_source: str | None, camera_index: int) -> int | str:
    if camera_source is None:
        return camera_index
    try:
        return int(camera_source)
    except ValueError:
        return camera_source


def list_cameras(probe_limit: int, width: int, height: int) -> None:
    found_any = False
    for camera_index in range(probe_limit):
        capture = cv2.VideoCapture(camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        capture.release()
        if not ok or frame is None:
            continue

        found_any = True
        actual_width = int(frame.shape[1])
        actual_height = int(frame.shape[0])
        print(f"camera-index {camera_index}: opened {actual_width}x{actual_height}")

    if not found_any:
        print(f"No cameras opened in indexes 0-{probe_limit - 1}.")


def open_camera(camera_source: int | str, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera source {camera_source!r}")
    return capture


def save_trajectory_image(path: Path, frame: np.ndarray | None) -> None:
    if frame is None:
        print("No trajectory image was saved because no frame was captured.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to write trajectory image: {path}")
    print(f"Saved trajectory image to {path}")


def should_use_marker(marker_id: int, marker_ids: set[int] | None) -> bool:
    return marker_ids is None or marker_id in marker_ids


def track_markers(args: argparse.Namespace) -> None:
    aruco_dictionary = get_aruco_dictionary(args.dictionary)
    detector = create_detector(aruco_dictionary)
    camera_source = parse_camera_source(args.camera_source, args.camera_index)
    capture = open_camera(camera_source, args.width, args.height)
    marker_ids = target_ids(args)
    trajectories: dict[int, list[tuple[int, int]]] = {}
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    territory_owner: np.ndarray | None = None
    support_items: list[SupportItem] = []
    active_buffs: dict[int, BuffState] = {}
    calibration = CalibrationState()
    field_mask: np.ndarray | None = None
    game_start_time = time.monotonic()
    game_over = False
    final_scores: dict[int, int] = {}

    last_frame_time = time.monotonic()
    last_print_time = 0.0
    last_visualization: np.ndarray | None = None
    fps = 0.0
    mouse_state: MouseState | None = None

    if args.headless:
        print("Tracking started. Press Ctrl+C to quit.")
    else:
        print(
            "Tracking started. Press q to quit, c to clear trails, r to reset, "
            "k to calibrate camera, p to calibrate projector."
        )
        cv2.namedWindow(WINDOW_NAME)
        mouse_state = MouseState(support_items, calibration, args.item_radius)
        cv2.setMouseCallback(WINDOW_NAME, handle_mouse_event, mouse_state)

    def reset_game() -> None:
        nonlocal game_start_time, game_over, final_scores
        trajectories.clear()
        paint_segments.clear()
        support_items.clear()
        active_buffs.clear()
        if territory_owner is not None:
            territory_owner.fill(UNOWNED_TERRITORY)
            if field_mask is not None:
                territory_owner[field_mask == 0] = UNOWNED_TERRITORY
        game_start_time = time.monotonic()
        game_over = False
        final_scores = {}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")
            if territory_owner is None:
                territory_owner = np.full(frame.shape[:2], UNOWNED_TERRITORY, dtype=np.int16)

            now = time.monotonic()
            if calibration.camera_dirty:
                field_mask = build_field_mask(frame.shape, calibration.camera_points)
                calibration.camera_dirty = False
                reset_game()
            if calibration.projector_dirty:
                calibration.projector_dirty = False
            if field_mask is None and len(calibration.camera_points) == 4:
                field_mask = build_field_mask(frame.shape, calibration.camera_points)
                territory_owner[field_mask == 0] = UNOWNED_TERRITORY
            elapsed = now - last_frame_time
            last_frame_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            corners, ids = detect_markers(frame, aruco_dictionary, detector)
            detections: list[str] = []
            visible_markers: list[tuple[np.ndarray, int, float]] = []
            marker_positions: list[tuple[int, tuple[int, int]]] = []
            prune_expired_buffs(active_buffs, now)
            remaining_seconds = None
            if args.game_duration > 0:
                remaining_seconds = max(0.0, args.game_duration - (now - game_start_time))
            updates_enabled = not game_over
            if remaining_seconds is not None and remaining_seconds <= 0:
                updates_enabled = False

            if ids is not None:
                for marker_corners, marker_id_array in zip(corners, ids):
                    marker_id = int(marker_id_array[0])
                    if not should_use_marker(marker_id, marker_ids):
                        continue

                    center_x, center_y = marker_center(marker_corners)
                    heading = marker_heading_degrees(marker_corners)
                    size_multiplier = active_size_multiplier(marker_id, active_buffs, now)
                    if updates_enabled:
                        thickness = max(1, int(args.trajectory_thickness * size_multiplier))
                        update_trajectory(
                            trajectories,
                            paint_segments,
                            territory_owner,
                            marker_id,
                            (center_x, center_y),
                            args.trajectory_length,
                            args.min_trajectory_distance,
                            thickness,
                            field_mask,
                        )
                    detections.append(
                        f"id={marker_id} x={center_x} y={center_y} "
                        f"heading={heading:.1f} path={len(trajectories[marker_id])}"
                    )
                    visible_markers.append((marker_corners, marker_id, size_multiplier))
                    marker_positions.append((marker_id, (center_x, center_y)))

            if updates_enabled:
                check_item_pickups(
                    support_items,
                    marker_positions,
                    active_buffs,
                    args.item_pickup_radius,
                    now,
                    args.mushroom_duration,
                    args.mushroom_size_multiplier,
                )

            scores = territory_scores(territory_owner, marker_ids)
            if remaining_seconds is not None and remaining_seconds <= 0:
                if not game_over:
                    game_over = True
                    final_scores = scores
                if final_scores:
                    scores = final_scores
            leader_text = leader_summary(scores)
            if now - last_print_time >= args.print_every:
                status_parts = [format_scores(scores), leader_text]
                if remaining_seconds is not None:
                    status_parts.insert(0, f"time={format_time(remaining_seconds)}")
                if game_over:
                    status_parts.append("GAME OVER")
                status_message = " | ".join(status_parts)
                if detections:
                    print(f"{' | '.join(detections)} | {status_message}", flush=True)
                else:
                    print(f"no marker | {status_message}", flush=True)
                last_print_time = now

            render_territory_overlay(frame, territory_owner, marker_ids, field_mask)
            draw_trajectories(
                frame,
                paint_segments,
                trajectories,
                args.trajectory_thickness,
            )
            draw_support_items(frame, support_items)
            for marker_corners, marker_id, size_multiplier in visible_markers:
                draw_marker_details(frame, marker_corners, marker_id, fps, size_multiplier)
            draw_territory_scores(frame, scores)
            status_lines: list[str] = []
            if remaining_seconds is not None:
                status_lines.append(f"Time: {format_time(remaining_seconds)}")
            status_lines.append(leader_text)
            if active_buffs:
                for marker_id, buff in sorted(active_buffs.items()):
                    buff_remaining = max(0.0, buff.expires_at - now)
                    status_lines.append(f"Boost P{marker_id}: {buff_remaining:.1f}s")
            draw_info_panel(frame, status_lines, (frame.shape[1] - 12, 12), align_right=True)
            draw_calibration_guides(frame, calibration)
            if game_over:
                draw_game_over_panel(frame, scores)
            last_visualization = frame.copy()

            if args.headless:
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("c"):
                trajectories.clear()
                paint_segments.clear()
                if territory_owner is not None:
                    territory_owner.fill(UNOWNED_TERRITORY)
                    if field_mask is not None:
                        territory_owner[field_mask == 0] = UNOWNED_TERRITORY
                print("cleared trajectories", flush=True)
            if key == ord("r"):
                reset_game()
                print("reset game state", flush=True)
            if key == ord("k"):
                new_mode = None if calibration.mode == CALIBRATION_CAMERA else CALIBRATION_CAMERA
                set_calibration_mode(calibration, new_mode)
            if key == ord("p"):
                new_mode = None if calibration.mode == CALIBRATION_PROJECTOR else CALIBRATION_PROJECTOR
                set_calibration_mode(calibration, new_mode)
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.")
    finally:
        save_trajectory_image(args.trajectory_output, last_visualization)
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        list_cameras(args.camera_probe_limit, args.width, args.height)
        return
    if args.generate_marker is not None:
        generate_marker(args)
        return
    track_markers(args)


if __name__ == "__main__":
    main()
