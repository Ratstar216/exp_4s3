from __future__ import annotations

import argparse
import math
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "AR Marker Tracker"
DEFAULT_MARKER_COLORS = [
    (100, 20, 255),    # marker 0: neon pink  RGB(255, 20, 100)
    (10, 255, 50),     # marker 1: lime green RGB(50, 255, 10)
    (0, 180, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
]
UNOWNED_TERRITORY = -1
ITEM_MUSHROOM = "mushroom"
TOOL_MUSHROOM = ITEM_MUSHROOM
CALIBRATION_CAMERA = "camera"
CALIBRATION_PROJECTOR = "projector"
MIN_TRAJECTORY_THICKNESS = 1
MIN_LOOP_GAP = 8  # minimum old segments skipped when checking for self-intersection
DEFAULT_OUTLINE_THICKNESS = 2
DEFAULT_MARKER_RADIUS = 5
DEFAULT_ARROW_THICKNESS = 2
TOP_TERRITORY_BAR_HEIGHT = 72
MUSHROOM_SPRITE_PATH = Path(__file__).resolve().parent / "assets" / "mushroom.png"
BUTTON_HEIGHT = 42
BUTTON_WIDTH = 168

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
    active_tool: str | None = None
    mushroom_button_rect: tuple[int, int, int, int] | None = None
    manual_draw_marker_id: int | None = None
    manual_draw_last_point: tuple[int, int] | None = None


@dataclass
class TrackerSnapshot:
    scores: dict[int, int]
    total_area: int
    remaining_seconds: float | None
    game_over: bool
    active_tool: str | None
    buff_remaining: dict[int, float]
    calibration_mode: str | None
    fps: float


@dataclass
class TrackerCommand:
    name: str
    payload: object = None


class TrackerController:
    def __init__(self) -> None:
        self._commands: queue.Queue[TrackerCommand] = queue.Queue()

    def enqueue(self, name: str, payload: object = None) -> None:
        self._commands.put(TrackerCommand(name=name, payload=payload))

    def drain(self) -> list[TrackerCommand]:
        commands: list[TrackerCommand] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def set_tool_mode(self, tool: str | None) -> None:
        self.enqueue("set_tool_mode", tool)

    def clear_trajectories(self) -> None:
        self.enqueue("clear_trajectories")

    def reset_game(self) -> None:
        self.enqueue("reset_game")

    def set_calibration_mode(self, mode: str | None) -> None:
        self.enqueue("set_calibration_mode", mode)

    def pointer_event(self, phase: str, button: str, x: int, y: int) -> None:
        self.enqueue("pointer_event", (phase, button, x, y))

    def left_press(self, x: int, y: int) -> None:
        self.pointer_event("press", "left", x, y)

    def left_drag(self, x: int, y: int) -> None:
        self.pointer_event("move", "left", x, y)

    def left_release(self, x: int, y: int) -> None:
        self.pointer_event("release", "left", x, y)

    def right_press(self, x: int, y: int) -> None:
        self.pointer_event("press", "right", x, y)

    def stop(self) -> None:
        self.enqueue("stop")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    args = parser.parse_args(argv)
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
    point_diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = [
        pts[int(np.argmin(sums))],
        pts[int(np.argmin(point_diffs))],
        pts[int(np.argmax(sums))],
        pts[int(np.argmax(point_diffs))],
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
    lowest_score_id = sorted_scores[-1][0] if len(sorted_scores) > 1 else None
    if lowest_score_id is not None and lowest_score_id != leader_id:
        return f"Leader: Player {leader_id} | Trailing: Player {lowest_score_id}"
    return f"Leader: Player {leader_id}"


def winning_marker_ids(scores: dict[int, int]) -> list[int]:
    if not scores:
        return []
    top_score = max(scores.values())
    if top_score <= 0:
        return []
    return [marker_id for marker_id, score in scores.items() if score == top_score]


def winner_banner_text(scores: dict[int, int]) -> str:
    winners = winning_marker_ids(scores)
    if not winners:
        return "NO WINNER"
    if len(winners) == 1:
        return f"WINNER P{winners[0]}"
    joined = " / ".join(f"P{marker_id}" for marker_id in winners)
    return f"DRAW {joined}"


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


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int] | None) -> bool:
    if rect is None:
        return False
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def manual_draw_tool(marker_id: int) -> str:
    return f"manual_draw:{marker_id}"


def manual_draw_marker_id(tool: str | None) -> int | None:
    if tool is None or not tool.startswith("manual_draw:"):
        return None
    _, marker_id_text = tool.split(":", maxsplit=1)
    try:
        return int(marker_id_text)
    except ValueError:
        return None


def set_active_tool(state: MouseState, tool: str | None) -> None:
    state.active_tool = tool
    state.manual_draw_marker_id = None
    state.manual_draw_last_point = None


def toggle_tool_mode(state: MouseState, tool: str) -> None:
    if state.active_tool == tool:
        set_active_tool(state, None)
    else:
        set_active_tool(state, tool)


def draw_manual_segment(
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int], int]],
    territory_owner: np.ndarray,
    marker_id: int,
    start: tuple[int, int],
    end: tuple[int, int],
    thickness: int,
    field_mask: np.ndarray | None,
) -> None:
    paint_segments.append((marker_id, start, end, thickness))
    paint_territory(territory_owner, marker_id, start, end, thickness, field_mask)


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
    if event == cv2.EVENT_LBUTTONDOWN and point_in_rect(point, state.mushroom_button_rect):
        toggle_tool_mode(state, TOOL_MUSHROOM)
        return
    if state.active_tool is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN and state.active_tool == TOOL_MUSHROOM:
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


def load_item_sprite(path: Path) -> np.ndarray:
    sprite = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if sprite is None:
        raise RuntimeError(f"Failed to load support item sprite: {path}")
    if sprite.ndim != 3 or sprite.shape[2] != 3:
        raise RuntimeError(
            f"Expected a 3-channel support item sprite image, got shape {sprite.shape!r}: {path}"
        )
    return sprite


def draw_sprite(
    frame: np.ndarray,
    sprite: np.ndarray,
    center: tuple[int, int],
    radius: int,
) -> None:
    target_height = max(12, radius * 2)
    scale = target_height / sprite.shape[0]
    target_width = max(12, int(round(sprite.shape[1] * scale)))
    resized = cv2.resize(sprite, (target_width, target_height), interpolation=cv2.INTER_AREA)

    x0 = center[0] - target_width // 2
    y0 = center[1] - target_height // 2
    x1 = x0 + target_width
    y1 = y0 + target_height
    frame_x0 = max(0, x0)
    frame_y0 = max(0, y0)
    frame_x1 = min(frame.shape[1], x1)
    frame_y1 = min(frame.shape[0], y1)
    if frame_x0 >= frame_x1 or frame_y0 >= frame_y1:
        return

    sprite_x0 = frame_x0 - x0
    sprite_y0 = frame_y0 - y0
    sprite_x1 = sprite_x0 + (frame_x1 - frame_x0)
    sprite_y1 = sprite_y0 + (frame_y1 - frame_y0)
    sprite_region = resized[sprite_y0:sprite_y1, sprite_x0:sprite_x1]

    # Treat the near-white background in the source PNG as transparent.
    alpha_mask = np.any(sprite_region < 245, axis=2)
    if not np.any(alpha_mask):
        return

    frame_region = frame[frame_y0:frame_y1, frame_x0:frame_x1]
    frame_region[alpha_mask] = sprite_region[alpha_mask]


def draw_support_items(
    frame: np.ndarray,
    support_items: list[SupportItem],
    mushroom_sprite: np.ndarray,
) -> None:
    for item in support_items:
        if item.item_type == ITEM_MUSHROOM:
            draw_sprite(frame, mushroom_sprite, item.position, item.radius)
            continue
        raise RuntimeError(f"Unsupported support item type: {item.item_type}")


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


def draw_toggle_button(
    frame: np.ndarray,
    label: str,
    rect: tuple[int, int, int, int],
    active: bool,
) -> None:
    left, top, right, bottom = rect
    overlay = frame.copy()
    fill_color = (20, 120, 245) if active else (35, 35, 35)
    text_color = (255, 255, 255) if active else (230, 230, 230)
    border_color = (255, 255, 255) if active else (140, 140, 140)
    cv2.rectangle(overlay, (left, top), (right, bottom), fill_color, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    text_size, _baseline = cv2.getTextSize(label, font, scale, thickness)
    text_x = left + (right - left - text_size[0]) // 2
    text_y = top + (bottom - top + text_size[1]) // 2
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


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
        lines.append(winner_banner_text(scores))
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


def _segments_intersect_params(
    p1: tuple[int, int],
    p2: tuple[int, int],
    p3: tuple[int, int],
    p4: tuple[int, int],
) -> tuple[float, float] | None:
    """Return (t, s) if segment p1→p2 and p3→p4 properly intersect, else None."""
    dx1 = float(p2[0] - p1[0])
    dy1 = float(p2[1] - p1[1])
    dx2 = float(p4[0] - p3[0])
    dy2 = float(p4[1] - p3[1])
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < 1e-10:
        return None
    dx3 = float(p3[0] - p1[0])
    dy3 = float(p3[1] - p1[1])
    t = (dx3 * dy2 - dy3 * dx2) / denom
    s = (dx3 * dy1 - dy3 * dx1) / denom
    eps = 1e-9
    if eps < t < 1.0 - eps and eps < s < 1.0 - eps:
        return t, s
    return None


def detect_loop(
    trajectory: list[tuple[int, int]],
    new_point: tuple[int, int],
) -> tuple[tuple[float, float], int] | None:
    """Check if trajectory[-1]→new_point crosses any earlier segment.

    Returns (intersection_xy, loop_start_index) for the smallest loop found,
    where loop_start_index is i such that trajectory[i]→trajectory[i+1] is crossed.
    The loop polygon is [intersection, trajectory[i+1], ..., trajectory[-1]].
    """
    n = len(trajectory)
    if n < MIN_LOOP_GAP + 2:
        return None
    prev = trajectory[-1]
    # Iterate from most-recent eligible segment downward to find the smallest loop first.
    for i in range(n - 1 - MIN_LOOP_GAP, -1, -1):
        result = _segments_intersect_params(prev, new_point, trajectory[i], trajectory[i + 1])
        if result is not None:
            _t, s = result
            ix = (
                trajectory[i][0] + s * (trajectory[i + 1][0] - trajectory[i][0]),
                trajectory[i][1] + s * (trajectory[i + 1][1] - trajectory[i][1]),
            )
            return ix, i
    return None


def fill_loop_area(
    territory_owner: np.ndarray,
    marker_id: int,
    intersection: tuple[float, float],
    trajectory: list[tuple[int, int]],
    loop_start_index: int,
    field_mask: np.ndarray | None,
    filled_loops: list[tuple[int, list[tuple[int, int]]]],
) -> None:
    """Fill the interior of a detected loop in territory_owner."""
    polygon_points = [
        (int(round(intersection[0])), int(round(intersection[1]))),
        *trajectory[loop_start_index + 1 :],
    ]
    if len(polygon_points) < 3:
        return
    fill_mask = np.zeros(territory_owner.shape, dtype=np.uint8)
    cv2.fillPoly(fill_mask, [np.array(polygon_points, dtype=np.int32)], 255)
    if field_mask is not None:
        fill_mask = cv2.bitwise_and(fill_mask, fill_mask, mask=field_mask)
    territory_owner[fill_mask > 0] = marker_id
    filled_loops.append((marker_id, polygon_points))


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
    filled_loops: list[tuple[int, list[tuple[int, int]]]] | None = None,
) -> None:
    trajectory = trajectories.setdefault(marker_id, [])
    if should_add_trajectory_point(trajectory, point, min_distance):
        if trajectory:
            start = trajectory[-1]
            loop = detect_loop(trajectory, point)
            paint_segments.append((marker_id, start, point, thickness))
            paint_territory(territory_owner, marker_id, start, point, thickness, field_mask)
            if loop is not None and filled_loops is not None:
                intersection, loop_start_index = loop
                fill_loop_area(
                    territory_owner, marker_id, intersection,
                    trajectory, loop_start_index, field_mask,
                    filled_loops,
                )
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


def territory_total_area(territory_owner: np.ndarray, field_mask: np.ndarray | None) -> int:
    if field_mask is not None:
        return int(np.count_nonzero(field_mask))
    return int(territory_owner.size)


def tool_label(tool: str | None) -> str:
    if tool is None:
        return "None"
    if tool == TOOL_MUSHROOM:
        return "Mushroom placement"
    marker_id = manual_draw_marker_id(tool)
    if marker_id is not None:
        return f"Manual draw P{marker_id}"
    return tool


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
    filled_loops: list[tuple[int, list[tuple[int, int]]]] | None = None,
) -> None:
    for loop_marker_id, polygon_points in (filled_loops or []):
        cv2.fillPoly(frame, [np.array(polygon_points, dtype=np.int32)], marker_color(loop_marker_id))

    for marker_id, start, end, segment_thickness in paint_segments:
        cv2.line(
            frame,
            start,
            end,
            marker_color(marker_id),
            max(MIN_TRAJECTORY_THICKNESS, segment_thickness),
            cv2.LINE_AA,
        )

    for marker_id, trajectory in trajectories.items():
        if not trajectory:
            continue
        color = marker_color(marker_id)
        cv2.circle(frame, trajectory[-1], max(6, thickness // 2), color, -1)


def draw_territory_bar(
    frame: np.ndarray,
    scores: dict[int, int],
    territory_owner: np.ndarray,
    field_mask: np.ndarray | None,
) -> None:
    total_area = territory_total_area(territory_owner, field_mask)
    if total_area <= 0:
        return

    padding = 12
    panel_height = TOP_TERRITORY_BAR_HEIGHT
    panel_left = 12
    panel_top = 12
    panel_right = frame.shape[1] - 12
    panel_bottom = panel_top + panel_height
    bar_left = panel_left + padding
    bar_right = panel_right - padding
    bar_top = panel_top + 34
    bar_bottom = bar_top + 18
    bar_width = bar_right - bar_left
    if bar_width <= 0:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_left, panel_top), (panel_right, panel_bottom), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (90, 90, 90), -1)

    leaders: set[int] = set()
    if scores:
        top_score = max(scores.values())
        leaders = {marker_id for marker_id, score in scores.items() if score == top_score and score > 0}

    score_items = list(scores.items())
    right_marker_id = score_items[-1][0] if len(score_items) > 1 else None
    left_cursor = bar_left
    right_cursor = bar_right
    for marker_id, score_px in score_items:
        if score_px <= 0:
            continue
        segment_width = int(round(bar_width * score_px / total_area))
        if segment_width <= 0:
            continue
        color = marker_color(marker_id)
        if marker_id == right_marker_id:
            next_left = max(left_cursor, right_cursor - segment_width)
            cv2.rectangle(frame, (next_left, bar_top), (right_cursor, bar_bottom), color, -1)
            right_cursor = next_left
        else:
            next_right = min(right_cursor, left_cursor + segment_width)
            cv2.rectangle(frame, (left_cursor, bar_top), (next_right, bar_bottom), color, -1)
            left_cursor = next_right

    claimed_area = sum(scores.values())
    unclaimed_area = max(0, total_area - claimed_area)
    if left_cursor < right_cursor and unclaimed_area > 0:
        cv2.rectangle(frame, (left_cursor, bar_top), (right_cursor, bar_bottom), (90, 90, 90), -1)

    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (255, 255, 255), 1)

    label_y = panel_top + 24
    if scores:
        left_marker_id = next(iter(scores))
        left_score = scores[left_marker_id]
        left_text = f"P{left_marker_id} {left_score / total_area * 100:.0f}%"
        left_color = marker_color(left_marker_id)
        left_thickness = 3 if left_marker_id in leaders and len(leaders) == 1 else 2
        cv2.putText(
            frame,
            left_text,
            (bar_left, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            left_color,
            left_thickness,
            cv2.LINE_AA,
        )

        right_marker_id = next(reversed(scores))
        if right_marker_id != left_marker_id:
            right_score = scores[right_marker_id]
            right_text = f"{right_score / total_area * 100:.0f}% P{right_marker_id}"
            right_color = marker_color(right_marker_id)
            right_thickness = 3 if right_marker_id in leaders and len(leaders) == 1 else 2
            text_size, _baseline = cv2.getTextSize(
                right_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                right_thickness,
            )
            cv2.putText(
                frame,
                right_text,
                (bar_right - text_size[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                right_color,
                right_thickness,
                cv2.LINE_AA,
            )

    if unclaimed_area > 0:
        neutral_text = f"Unclaimed {unclaimed_area / total_area * 100:.0f}%"
        neutral_size, _baseline = cv2.getTextSize(
            neutral_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        cv2.putText(
            frame,
            neutral_text,
            ((frame.shape[1] - neutral_size[0]) // 2, panel_bottom - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
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

    outline_thickness = max(
        DEFAULT_OUTLINE_THICKNESS,
        int(DEFAULT_OUTLINE_THICKNESS * size_multiplier),
    )
    marker_radius = max(
        DEFAULT_MARKER_RADIUS,
        int(DEFAULT_MARKER_RADIUS * size_multiplier),
    )
    arrow_thickness = max(
        DEFAULT_ARROW_THICKNESS,
        int(DEFAULT_ARROW_THICKNESS * size_multiplier),
    )
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


def track_markers(
    args: argparse.Namespace,
    *,
    controller: TrackerController | None = None,
    frame_callback: Callable[[np.ndarray, TrackerSnapshot], None] | None = None,
    render_hud: bool = True,
) -> None:
    aruco_dictionary = get_aruco_dictionary(args.dictionary)
    detector = create_detector(aruco_dictionary)
    camera_source = parse_camera_source(args.camera_source, args.camera_index)
    capture = open_camera(camera_source, args.width, args.height)
    mushroom_sprite = load_item_sprite(MUSHROOM_SPRITE_PATH)
    marker_ids = target_ids(args)
    trajectories: dict[int, list[tuple[int, int]]] = {}
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int], int]] = []
    filled_loops: list[tuple[int, list[tuple[int, int]]]] = []
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
    interaction_state = MouseState(support_items, calibration, args.item_radius)
    use_opencv_window = not args.headless and controller is None

    if args.headless and controller is None:
        print("Tracking started. Press Ctrl+C to quit.")
    elif use_opencv_window:
        print(
            "Tracking started. Press q to quit, c to clear trails, r to reset, "
            "click the Mushroom button to toggle placement, k to calibrate camera, "
            "p to calibrate projector."
        )
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, handle_mouse_event, interaction_state)

    def clear_trajectories() -> None:
        trajectories.clear()
        paint_segments.clear()
        filled_loops.clear()
        if territory_owner is not None:
            territory_owner.fill(UNOWNED_TERRITORY)
            if field_mask is not None:
                territory_owner[field_mask == 0] = UNOWNED_TERRITORY

    def reset_game() -> None:
        nonlocal game_start_time, game_over, final_scores
        clear_trajectories()
        support_items.clear()
        active_buffs.clear()
        game_start_time = time.monotonic()
        game_over = False
        final_scores = {}

    def handle_pointer_event(
        phase: str,
        button: str,
        point: tuple[int, int],
    ) -> None:
        if calibration.mode is not None:
            if button == "left" and phase == "press":
                add_calibration_point(calibration, point)
            elif button == "right" and phase == "press":
                points = active_calibration_points(calibration)
                if points:
                    points.pop()
            return

        if interaction_state.active_tool == TOOL_MUSHROOM:
            if button == "left" and phase == "press":
                support_items.append(SupportItem(ITEM_MUSHROOM, point, args.item_radius))
            elif button == "right" and phase == "press":
                remove_nearest_item(support_items, point, args.item_radius * 1.5)
            return

        draw_marker_id = manual_draw_marker_id(interaction_state.active_tool)
        if draw_marker_id is None or button != "left":
            return
        if phase == "press":
            interaction_state.manual_draw_marker_id = draw_marker_id
            interaction_state.manual_draw_last_point = point
            return
        if phase == "move":
            if (
                interaction_state.manual_draw_marker_id != draw_marker_id
                or interaction_state.manual_draw_last_point is None
            ):
                return
            if not should_add_trajectory_point(
                [interaction_state.manual_draw_last_point],
                point,
                args.min_trajectory_distance,
            ):
                return
            draw_manual_segment(
                paint_segments,
                territory_owner,
                draw_marker_id,
                interaction_state.manual_draw_last_point,
                point,
                args.trajectory_thickness,
                field_mask,
            )
            interaction_state.manual_draw_last_point = point
            return
        if phase == "release":
            interaction_state.manual_draw_marker_id = None
            interaction_state.manual_draw_last_point = None

    def apply_controller_command(command: TrackerCommand) -> bool:
        payload = command.payload
        if command.name == "stop":
            return False
        if command.name == "set_tool_mode":
            set_active_tool(interaction_state, payload if isinstance(payload, str) else None)
            return True
        if command.name == "clear_trajectories":
            clear_trajectories()
            return True
        if command.name == "reset_game":
            reset_game()
            return True
        if command.name == "set_calibration_mode":
            if payload in {CALIBRATION_CAMERA, CALIBRATION_PROJECTOR, None}:
                set_calibration_mode(calibration, payload)
            return True
        if command.name == "pointer_event" and isinstance(payload, tuple):
            phase, button, x, y = payload
            if (
                isinstance(phase, str)
                and isinstance(button, str)
                and isinstance(x, int)
                and isinstance(y, int)
                and territory_owner is not None
            ):
                handle_pointer_event(phase, button, (x, y))
            return True
        return True
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")
            if territory_owner is None:
                territory_owner = np.full(frame.shape[:2], UNOWNED_TERRITORY, dtype=np.int16)
            if controller is not None:
                keep_running = True
                for command in controller.drain():
                    if not apply_controller_command(command):
                        keep_running = False
                        break
                if not keep_running:
                    break

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
                        thickness = max(
                            MIN_TRAJECTORY_THICKNESS,
                            int(args.trajectory_thickness * size_multiplier),
                        )
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
                            filled_loops,
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
            total_area = territory_total_area(territory_owner, field_mask)
            buff_remaining = {
                marker_id: max(0.0, buff.expires_at - now)
                for marker_id, buff in sorted(active_buffs.items())
            }
            if controller is None and now - last_print_time >= args.print_every:
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
                filled_loops,
            )
            draw_support_items(frame, support_items, mushroom_sprite)
            for marker_corners, marker_id, size_multiplier in visible_markers:
                draw_marker_details(frame, marker_corners, marker_id, fps, size_multiplier)
            if render_hud:
                draw_territory_bar(frame, scores, territory_owner, field_mask)
                if use_opencv_window:
                    button_top = TOP_TERRITORY_BAR_HEIGHT + 24
                    mushroom_button_rect = (
                        12,
                        button_top,
                        12 + BUTTON_WIDTH,
                        button_top + BUTTON_HEIGHT,
                    )
                    interaction_state.mushroom_button_rect = mushroom_button_rect
                    draw_toggle_button(
                        frame,
                        "Mushroom",
                        mushroom_button_rect,
                        interaction_state.active_tool == TOOL_MUSHROOM,
                    )
                status_lines: list[str] = []
                if remaining_seconds is not None:
                    status_lines.append(f"Time: {format_time(remaining_seconds)}")
                status_lines.append(f"Mode: {tool_label(interaction_state.active_tool)}")
                if buff_remaining:
                    for marker_id, seconds_left in buff_remaining.items():
                        status_lines.append(f"Boost P{marker_id}: {seconds_left:.1f}s")
                draw_info_panel(
                    frame,
                    status_lines,
                    (frame.shape[1] - 12, TOP_TERRITORY_BAR_HEIGHT + 24),
                    align_right=True,
                )
            draw_calibration_guides(frame, calibration)
            if game_over and render_hud:
                draw_game_over_panel(frame, scores)
            snapshot = TrackerSnapshot(
                scores=dict(scores),
                total_area=total_area,
                remaining_seconds=remaining_seconds,
                game_over=game_over,
                active_tool=interaction_state.active_tool,
                buff_remaining=buff_remaining,
                calibration_mode=calibration.mode,
                fps=fps,
            )
            last_visualization = frame.copy()
            if frame_callback is not None:
                frame_callback(last_visualization, snapshot)

            if not use_opencv_window:
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("c"):
                clear_trajectories()
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
