# Robot AR Marker Tracker

Real-time ArUco marker tracking for small robots using a PC camera.

## Setup

```sh
uv sync
```

## Generate a Marker

Print one marker per robot and attach it flat on top of the robot.

```sh
uv run python hello.py --generate-marker 0 --output marker-0.png
uv run python hello.py --generate-marker 1 --output marker-1.png
```

The generated image includes a white margin around the marker. The default
dictionary is `4x4_50`; use the same dictionary when generating and tracking
markers.

## Track from the Camera

```sh
uv run python hello.py --target-id 0
```

For two robots, track both marker IDs at the same time:

```sh
uv run python hello.py --robot-ids 0 1
```

The preview window draws each detected marker border, center point, ID, heading,
frame rate, and broad colored trajectory. Trajectories are drawn in movement
order, so a robot can overwrite territory by driving over the opponent's older
trajectory. The preview also displays each player's current territory size in
painted pixels. The console prints image-space coordinates, stored trajectory
points, and territory scores:

```text
id=0 x=645 y=318 heading=2.4 path=42 | id=1 x=220 y=510 heading=-88.1 path=39 | territory: id=0 8200px | id=1 7600px
```

Press `q` or `Esc` in the preview window to quit. Press `c` to clear all
trajectories while tracking continues. When tracking stops, including by
`Ctrl+C`, the latest image with trajectories is saved to `trajectories.png`.

Useful options:

- `--list-cameras`: probe local camera indexes and exit.
- `--camera-index 1`: use another camera.
- `--camera-source <url-or-file>`: use an IP/RTSP/HTTP camera URL or video file.
- `--width 1920 --height 1080`: request a different capture resolution.
- `--dictionary 5x5_100`: use another ArUco dictionary.
- `--trajectory-length 1200`: keep more path history per robot.
- `--min-trajectory-distance 5`: add path points only after larger pixel moves.
- `--trajectory-thickness 24`: make territory paths broader.
- `--trajectory-output result.png`: choose where to save the final trajectory image.
- `--headless`: print detections without opening a preview window.

## iPhone Camera

On macOS, iPhone Continuity Camera usually appears to OpenCV as one of the
numeric camera indexes. The index can change, so probe it before tracking:

```sh
uv run python hello.py --list-cameras
```

Then run tracking with the index that shows the iPhone view:

```sh
uv run python hello.py --camera-index 1 --robot-ids 0 1 --trajectory-thickness 96
```

If Continuity Camera does not appear reliably, use an iPhone camera app that
publishes an HTTP, RTSP, or similar stream, then pass that stream URL:

```sh
uv run python hello.py --camera-source http://IPHONE_ADDRESS:PORT/video --robot-ids 0 1 --trajectory-thickness 96
```

For an overhead camera, `x`, `y`, and territory sizes are measured in camera
image pixels. Calibrate the camera and field later if you need real-world field
coordinates or square-centimeter territory scoring.
