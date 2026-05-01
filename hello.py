from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "AR Marker Tracker"
DEFAULT_MARKER_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 180, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 0, 255),
]

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
    args = parser.parse_args()
    if args.trajectory_length < 2:
        parser.error("--trajectory-length must be at least 2")
    if args.min_trajectory_distance < 0:
        parser.error("--min-trajectory-distance must be 0 or greater")
    if args.trajectory_thickness < 1:
        parser.error("--trajectory-thickness must be at least 1")
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
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int]]],
    marker_id: int,
    point: tuple[int, int],
    max_length: int,
    min_distance: float,
) -> None:
    trajectory = trajectories.setdefault(marker_id, [])
    if should_add_trajectory_point(trajectory, point, min_distance):
        if trajectory:
            paint_segments.append((marker_id, trajectory[-1], point))
        trajectory.append(point)
        if len(trajectory) > max_length:
            del trajectory[: len(trajectory) - max_length]
        total_segments = sum(max(0, len(points) - 1) for points in trajectories.values())
        if len(paint_segments) > total_segments:
            del paint_segments[: len(paint_segments) - total_segments]


def draw_trajectories(
    frame: np.ndarray,
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int]]],
    trajectories: dict[int, list[tuple[int, int]]],
    thickness: int,
) -> None:
    for marker_id, start, end in paint_segments:
        cv2.line(frame, start, end, marker_color(marker_id), thickness, cv2.LINE_AA)

    for marker_id, trajectory in trajectories.items():
        if not trajectory:
            continue
        color = marker_color(marker_id)
        cv2.circle(frame, trajectory[-1], max(6, thickness // 2), color, -1)


def draw_marker_details(
    frame: np.ndarray,
    corners: np.ndarray,
    marker_id: int,
    fps: float,
) -> None:
    center_x, center_y = marker_center(corners)
    heading = marker_heading_degrees(corners)
    points = corners.reshape(4, 2).astype(int)
    top_left, top_right, bottom_right, bottom_left = points
    color = marker_color(marker_id)

    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
    cv2.circle(frame, (center_x, center_y), 5, color, -1)
    cv2.arrowedLine(
        frame,
        tuple(top_left),
        tuple(top_right),
        color,
        2,
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


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")
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
    capture = open_camera(args.camera_index, args.width, args.height)
    marker_ids = target_ids(args)
    trajectories: dict[int, list[tuple[int, int]]] = {}
    paint_segments: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    last_frame_time = time.monotonic()
    last_print_time = 0.0
    last_visualization: np.ndarray | None = None
    fps = 0.0

    if args.headless:
        print("Tracking started. Press Ctrl+C to quit.")
    else:
        print("Tracking started. Press q to quit or c to clear trajectories.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the camera")

            now = time.monotonic()
            elapsed = now - last_frame_time
            last_frame_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps else 1.0 / elapsed

            corners, ids = detect_markers(frame, aruco_dictionary, detector)
            detections: list[str] = []
            visible_markers: list[tuple[np.ndarray, int]] = []

            if ids is not None:
                for marker_corners, marker_id_array in zip(corners, ids):
                    marker_id = int(marker_id_array[0])
                    if not should_use_marker(marker_id, marker_ids):
                        continue

                    center_x, center_y = marker_center(marker_corners)
                    heading = marker_heading_degrees(marker_corners)
                    update_trajectory(
                        trajectories,
                        paint_segments,
                        marker_id,
                        (center_x, center_y),
                        args.trajectory_length,
                        args.min_trajectory_distance,
                    )
                    detections.append(
                        f"id={marker_id} x={center_x} y={center_y} "
                        f"heading={heading:.1f} path={len(trajectories[marker_id])}"
                    )
                    visible_markers.append((marker_corners, marker_id))

            if now - last_print_time >= args.print_every:
                if detections:
                    print(" | ".join(detections), flush=True)
                else:
                    print("no marker", flush=True)
                last_print_time = now

            draw_trajectories(
                frame,
                paint_segments,
                trajectories,
                args.trajectory_thickness,
            )
            for marker_corners, marker_id in visible_markers:
                draw_marker_details(frame, marker_corners, marker_id, fps)
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
                print("cleared trajectories", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.")
    finally:
        save_trajectory_image(args.trajectory_output, last_visualization)
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.generate_marker is not None:
        generate_marker(args)
        return
    track_markers(args)


if __name__ == "__main__":
    main()
