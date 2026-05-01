# Robot AR Marker Tracker

Real-time ArUco marker tracking for a small robot using a PC camera.

## Setup

```sh
uv sync
```

## Generate a Marker

Print the generated image and attach it flat on top of the robot.

```sh
uv run python hello.py --generate-marker 0 --output marker-0.png
```

The generated image includes a white margin around the marker. The default
dictionary is `4x4_50`; use the same dictionary when generating and tracking
markers.

## Track from the Camera

```sh
uv run python hello.py --target-id 0
```

The preview window draws the detected marker border, center point, ID, heading,
and frame rate. The console prints image-space coordinates:

```text
id=0 x=645 y=318 heading=2.4
```

Press `q` or `Esc` in the preview window to quit.

Useful options:

- `--camera-index 1`: use another camera.
- `--width 1920 --height 1080`: request a different capture resolution.
- `--dictionary 5x5_100`: use another ArUco dictionary.
- `--headless`: print detections without opening a preview window.

For an overhead camera, `x` and `y` are pixel coordinates in the camera image.
Calibrate the camera and field later if you need real-world field coordinates.
