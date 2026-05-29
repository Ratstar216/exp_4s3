from __future__ import annotations

import sys
from contextlib import suppress

import cv2
import numpy as np
from PySide6.QtCore import QSignalBlocker, QRectF, QThread, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .tracker import (
    CALIBRATION_CAMERA,
    CALIBRATION_PROJECTOR,
    ITEM_MUSHROOM,
    TrackerController,
    TrackerSnapshot,
    generate_marker,
    list_cameras,
    manual_draw_marker_id,
    manual_draw_tool,
    marker_color,
    parse_args,
    track_markers,
    tool_label,
    winner_banner_text,
    winning_marker_ids,
)


class VideoCanvas(QWidget):
    left_pressed = Signal(int, int)
    left_dragged = Signal(int, int)
    left_released = Signal(int, int)
    right_pressed = Signal(int, int)

    def __init__(self, bg_color: str = "#0c1118") -> None:
        super().__init__()
        self._image: QImage | None = None
        self._rgb_frame: np.ndarray | None = None
        self._left_drag_active = False
        self._bg_color = QColor(bg_color)
        self.setMinimumSize(960, 540)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._rgb_frame = np.ascontiguousarray(rgb)
        height, width, _channels = self._rgb_frame.shape
        bytes_per_line = self._rgb_frame.strides[0]
        self._image = QImage(
            self._rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self.update()

    def _target_rect(self) -> QRectF | None:
        if self._image is None:
            return None
        image_width = self._image.width()
        image_height = self._image.height()
        if image_width <= 0 or image_height <= 0:
            return None
        widget_width = max(1.0, float(self.width()))
        widget_height = max(1.0, float(self.height()))
        scale = min(widget_width / image_width, widget_height / image_height)
        drawn_width = image_width * scale
        drawn_height = image_height * scale
        offset_x = (widget_width - drawn_width) / 2.0
        offset_y = (widget_height - drawn_height) / 2.0
        return QRectF(offset_x, offset_y, drawn_width, drawn_height)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._bg_color)
        target = self._target_rect()
        if self._image is None or target is None:
            painter.setPen(QColor("#9fb0c0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for camera frames...")
            return
        painter.drawImage(target, self._image)

    def _image_point(self, event: QMouseEvent) -> tuple[int, int] | None:
        target = self._target_rect()
        if self._image is None or target is None:
            return None
        point = event.position()
        if not target.contains(point):
            return None
        x_ratio = (point.x() - target.left()) / target.width()
        y_ratio = (point.y() - target.top()) / target.height()
        image_x = max(0, min(self._image.width() - 1, int(round(x_ratio * self._image.width()))))
        image_y = max(0, min(self._image.height() - 1, int(round(y_ratio * self._image.height()))))
        return image_x, image_y

    def mousePressEvent(self, event: QMouseEvent) -> None:
        image_point = self._image_point(event)
        if image_point is None:
            return
        image_x, image_y = image_point
        if event.button() == Qt.MouseButton.LeftButton:
            self._left_drag_active = True
            self.left_pressed.emit(image_x, image_y)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_pressed.emit(image_x, image_y)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._left_drag_active:
            return
        image_point = self._image_point(event)
        if image_point is None:
            return
        self.left_dragged.emit(*image_point)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if not self._left_drag_active:
            return
        self._left_drag_active = False
        image_point = self._image_point(event)
        if image_point is None:
            return
        self.left_released.emit(*image_point)


class TerritoryBarWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._snapshot: TrackerSnapshot | None = None
        self.setMinimumHeight(92)
        self.setMaximumHeight(92)

    def set_snapshot(self, snapshot: TrackerSnapshot) -> None:
        self._snapshot = snapshot
        self.update()

    def _player_color(self, marker_id: int) -> QColor:
        b, g, r = marker_color(marker_id)
        return QColor(r, g, b)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101722"))

        panel_rect = self.rect().adjusted(12, 12, -12, -12)
        painter.fillRect(panel_rect, QColor(18, 24, 34, 220))

        if self._snapshot is None or self._snapshot.total_area <= 0:
            painter.setPen(QColor("#b8c5d2"))
            painter.drawText(panel_rect, Qt.AlignmentFlag.AlignCenter, "Waiting for territory data...")
            return

        scores = self._snapshot.scores
        total_area = self._snapshot.total_area
        bar_rect = QRectF(panel_rect.left() + 14, panel_rect.top() + 32, panel_rect.width() - 28, 20)
        painter.fillRect(bar_rect, QColor("#5f6670"))

        claimed_area = sum(max(0, score) for score in scores.values())
        if claimed_area <= 0:
            painter.setPen(QColor("#ffffff"))
            painter.drawRect(bar_rect)
            painter.setPen(QColor("#d3dde7"))
            painter.drawText(
                panel_rect,
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
                "No territory painted",
            )
            return

        score_items = list(scores.items())
        right_marker_id = score_items[-1][0] if len(score_items) > 1 else None
        left_cursor = bar_rect.left()
        right_cursor = bar_rect.right()
        for marker_id, score in score_items:
            if score <= 0:
                continue
            width = bar_rect.width() * score / claimed_area
            if marker_id == right_marker_id:
                next_left = max(left_cursor, right_cursor - width)
                segment_rect = QRectF(next_left, bar_rect.top(), right_cursor - next_left, bar_rect.height())
                painter.fillRect(segment_rect, self._player_color(marker_id))
                right_cursor = next_left
            else:
                next_right = min(right_cursor, left_cursor + width)
                segment_rect = QRectF(left_cursor, bar_rect.top(), next_right - left_cursor, bar_rect.height())
                painter.fillRect(segment_rect, self._player_color(marker_id))
                left_cursor = next_right

        painter.setPen(QColor("#ffffff"))
        painter.drawRect(bar_rect)

        label_y = panel_rect.top() + 22
        if scores:
            first_id = next(iter(scores))
            first_score = scores[first_id]
            painter.setPen(self._player_color(first_id))
            painter.drawText(
                int(bar_rect.left()),
                int(label_y),
                f"P{first_id} {first_score / claimed_area * 100:.0f}%",
            )
            last_id = next(reversed(scores))
            if last_id != first_id:
                last_score = scores[last_id]
                label = f"{last_score / claimed_area * 100:.0f}% P{last_id}"
                metrics = painter.fontMetrics()
                painter.setPen(self._player_color(last_id))
                painter.drawText(int(bar_rect.right() - metrics.horizontalAdvance(label)), int(label_y), label)


class TrackerWorker(QThread):
    frame_ready = Signal(object, object)
    projection_frame_ready = Signal(object, object)
    failed = Signal(str)

    def __init__(self, args, controller: TrackerController) -> None:
        super().__init__()
        self._args = args
        self._controller = controller

    def run(self) -> None:
        try:
            track_markers(
                self._args,
                controller=self._controller,
                frame_callback=self._publish_frame,
                projection_frame_callback=self._publish_projection_frame,
            )
        except Exception as exc:
            self.failed.emit(str(exc))

    def _publish_frame(self, frame: np.ndarray, snapshot: TrackerSnapshot) -> None:
        self.frame_ready.emit(frame.copy(), snapshot)

    def _publish_projection_frame(
        self,
        frame: np.ndarray,
        snapshot: TrackerSnapshot,
    ) -> None:
        self.projection_frame_ready.emit(frame.copy(), snapshot)

    def stop(self) -> None:
        self._controller.stop()
        self.wait(3000)


class SpectatorWindow(QMainWindow):
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AR Marker Tracker Spectator")
        self.resize(1280, 800)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #05080d;
                color: #f3f7fb;
            }
            QLabel#projectorInfo {
                font-size: 22px;
                font-weight: 700;
                color: #f6fbff;
                padding: 6px 14px;
                background: rgba(12, 18, 26, 180);
                border-radius: 12px;
            }
            QLabel#projectorGameOver {
                font-size: 46px;
                font-weight: 800;
                color: #ffffff;
                padding: 18px 28px;
                background: rgba(8, 12, 18, 210);
                border: 2px solid rgba(255, 255, 255, 120);
                border-radius: 18px;
            }
            QLabel#projectorWinner {
                font-size: 74px;
                font-weight: 900;
                color: #ffffff;
                padding: 22px 34px;
                background: rgba(8, 12, 18, 225);
                border: 3px solid rgba(255, 255, 255, 150);
                border-radius: 24px;
            }
            """
        )

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self._territory_bar = TerritoryBarWidget()
        root_layout.addWidget(self._territory_bar)

        self._video = VideoCanvas()
        root_layout.addWidget(self._video, stretch=1)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(18, 12, 18, 18)
        info_row.setSpacing(12)
        self._time_label = QLabel("00:00")
        self._time_label.setObjectName("projectorInfo")
        info_row.addWidget(self._time_label, alignment=Qt.AlignmentFlag.AlignLeft)
        info_row.addStretch(1)
        self._mode_label = QLabel("Spectator")
        self._mode_label.setObjectName("projectorInfo")
        info_row.addWidget(self._mode_label, alignment=Qt.AlignmentFlag.AlignRight)
        root_layout.addLayout(info_row)

        self._game_over_label = QLabel("GAME OVER")
        self._game_over_label.setObjectName("projectorGameOver")
        self._game_over_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._game_over_label.hide()

        self._winner_label = QLabel("WINNER")
        self._winner_label.setObjectName("projectorWinner")
        self._winner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._winner_label.hide()

        container_layout = QVBoxLayout(self._video)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.addStretch(1)
        container_layout.addWidget(
            self._winner_label,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        container_layout.addSpacing(14)
        container_layout.addWidget(
            self._game_over_label,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        container_layout.addStretch(1)

        self.setCentralWidget(central)

    def apply_frame_update(self, frame: np.ndarray, snapshot: TrackerSnapshot) -> None:
        self._video.set_frame(frame)
        self._territory_bar.set_snapshot(snapshot)
        self._time_label.setText(
            "--:--"
            if snapshot.remaining_seconds is None
            else f"{int(snapshot.remaining_seconds) // 60:02d}:{int(snapshot.remaining_seconds) % 60:02d}"
        )
        self._mode_label.setText("Spectator")
        winners = winning_marker_ids(snapshot.scores)
        self._winner_label.setText(winner_banner_text(snapshot.scores))
        if len(winners) == 1:
            winner_color = QColor(*reversed(marker_color(winners[0])))
            style = (
                "font-size: 74px; font-weight: 900; color: #ffffff; "
                f"background: rgba({winner_color.red()}, {winner_color.green()}, {winner_color.blue()}, 230); "
                "border: 3px solid rgba(255, 255, 255, 160); border-radius: 24px; "
                "padding: 22px 34px;"
            )
            self._winner_label.setStyleSheet(style)
        else:
            self._winner_label.setStyleSheet("")
        self._winner_label.setVisible(snapshot.game_over)
        self._game_over_label.setVisible(snapshot.game_over)

    def set_fullscreen(self, enabled: bool) -> None:
        if enabled:
            self.showFullScreen()
        else:
            self.showNormal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_F11:
            self.set_fullscreen(not self.isFullScreen())
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.set_fullscreen(False)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)


class ProjectorWindow(QMainWindow):
    closed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AR Marker Tracker Projector")
        self.resize(1280, 720)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #ffffff;
            }
            """
        )

        self._video = VideoCanvas(bg_color="#ffffff")
        self._video.setMinimumSize(1, 1)
        self.setCentralWidget(self._video)

    def apply_frame_update(self, frame: np.ndarray, _snapshot: TrackerSnapshot) -> None:
        self._video.set_frame(frame)

    def set_fullscreen(self, enabled: bool) -> None:
        if enabled:
            self.showFullScreen()
        else:
            self.showNormal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_F11:
            self.set_fullscreen(not self.isFullScreen())
            return
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.set_fullscreen(False)
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closed.emit()
        super().closeEvent(event)


def resolve_player_ids(args) -> list[int]:
    ids: set[int] = set()
    if args.target_id:
        ids.update(args.target_id)
    if args.robot_ids:
        ids.update(args.robot_ids)
    return sorted(ids) or [0, 1]


class MainWindow(QMainWindow):
    def __init__(self, args) -> None:
        super().__init__()
        self._args = args
        self._player_ids = resolve_player_ids(args)
        self._controller = TrackerController()
        self._worker = TrackerWorker(args, self._controller)
        self._worker.frame_ready.connect(self._apply_frame_update)
        self._worker.failed.connect(self._show_worker_error)
        self._spectator_window = SpectatorWindow()
        self._worker.frame_ready.connect(self._spectator_window.apply_frame_update)
        self._spectator_window.closed.connect(self._on_spectator_closed)
        self._projector_window = ProjectorWindow()
        self._worker.projection_frame_ready.connect(self._projector_window.apply_frame_update)
        self._projector_window.closed.connect(self._on_projector_closed)

        self.setWindowTitle("AR Marker Tracker")
        self.resize(1500, 920)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0b1117;
                color: #ecf3f9;
            }
            QFrame#sidePanel {
                background: #131c27;
                border: 1px solid #243244;
                border-radius: 18px;
            }
            QLabel#panelTitle {
                font-size: 24px;
                font-weight: 700;
                color: #f9fbff;
            }
            QLabel#sectionLabel {
                color: #8ea0b3;
                font-size: 12px;
                text-transform: uppercase;
            }
            QLabel#valueLabel {
                font-size: 18px;
                font-weight: 600;
                color: #f2f7fb;
            }
            QPushButton {
                min-height: 44px;
                border-radius: 14px;
                border: 1px solid #32475f;
                background: #182331;
                color: #f4f8fc;
                font-size: 15px;
                font-weight: 600;
                padding: 10px 14px;
            }
            QPushButton:hover {
                background: #223143;
            }
            QPushButton:checked {
                background: #e28f1a;
                border-color: #ffbf5e;
                color: #101317;
            }
            QPushButton#dangerButton:checked {
                background: #c65252;
                border-color: #f28b8b;
                color: #ffffff;
            }
            """
        )

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(18)

        self._territory_bar = TerritoryBarWidget()
        root_layout.addWidget(self._territory_bar)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(18)
        root_layout.addLayout(content_layout, stretch=1)

        self._video = VideoCanvas()
        self._video.left_pressed.connect(self._controller.left_press)
        self._video.left_dragged.connect(self._controller.left_drag)
        self._video.left_released.connect(self._controller.left_release)
        self._video.right_pressed.connect(self._controller.right_press)
        content_layout.addWidget(self._video, stretch=1)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_panel.setFixedWidth(320)
        panel_layout = QVBoxLayout(side_panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(16)

        title = QLabel("Battle Controls")
        title.setObjectName("panelTitle")
        panel_layout.addWidget(title)

        self._mushroom_button = QPushButton("Mushroom Placement")
        self._mushroom_button.setCheckable(True)
        self._mushroom_button.toggled.connect(self._on_mushroom_toggled)
        panel_layout.addWidget(self._mushroom_button)

        self._draw_buttons: dict[int, QPushButton] = {}
        for marker_id in self._player_ids:
            button = QPushButton(f"Manual Draw P{marker_id}")
            button.setCheckable(True)
            button.toggled.connect(
                lambda checked, marker_id=marker_id: self._on_draw_button_toggled(marker_id, checked)
            )
            panel_layout.addWidget(button)
            self._draw_buttons[marker_id] = button

        self._clear_button = QPushButton("Clear Territory")
        self._clear_button.clicked.connect(self._controller.clear_trajectories)
        panel_layout.addWidget(self._clear_button)

        self._reset_button = QPushButton("Reset Match")
        self._reset_button.clicked.connect(self._controller.reset_game)
        panel_layout.addWidget(self._reset_button)

        self._camera_calibration_button = QPushButton("Camera Calibration")
        self._camera_calibration_button.setCheckable(True)
        self._camera_calibration_button.toggled.connect(self._on_camera_calibration_toggled)
        panel_layout.addWidget(self._camera_calibration_button)

        self._projector_calibration_button = QPushButton("Projector Calibration")
        self._projector_calibration_button.setCheckable(True)
        self._projector_calibration_button.toggled.connect(self._on_projector_calibration_toggled)
        panel_layout.addWidget(self._projector_calibration_button)

        self._show_spectator_button = QPushButton("Show Spectator Window")
        self._show_spectator_button.setCheckable(True)
        self._show_spectator_button.setChecked(True)
        self._show_spectator_button.toggled.connect(self._on_show_spectator_toggled)
        panel_layout.addWidget(self._show_spectator_button)

        self._show_projector_button = QPushButton("Show Projector Window")
        self._show_projector_button.setCheckable(True)
        self._show_projector_button.setChecked(True)
        self._show_projector_button.toggled.connect(self._on_show_projector_toggled)
        panel_layout.addWidget(self._show_projector_button)

        panel_layout.addSpacing(8)
        self._time_label = self._add_status_value(panel_layout, "Time")
        self._mode_label = self._add_status_value(panel_layout, "Tool Mode")
        self._fps_label = self._add_status_value(panel_layout, "FPS")
        self._boost_label = self._add_status_value(panel_layout, "Boosts")
        self._calibration_label = self._add_status_value(panel_layout, "Calibration")
        panel_layout.addStretch(1)

        content_layout.addWidget(side_panel)
        self.setCentralWidget(central)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

        self._spectator_window.show()
        self._projector_window.show()
        self._worker.start()

    def _add_status_value(self, layout: QVBoxLayout, title: str) -> QLabel:
        section = QLabel(title)
        section.setObjectName("sectionLabel")
        value = QLabel("Waiting...")
        value.setObjectName("valueLabel")
        layout.addWidget(section)
        layout.addWidget(value)
        return value

    def _on_mushroom_toggled(self, checked: bool) -> None:
        if checked:
            for marker_id, button in self._draw_buttons.items():
                with QSignalBlocker(button):
                    button.setChecked(False)
            self._controller.set_tool_mode(ITEM_MUSHROOM)
        else:
            self._controller.set_tool_mode(None)

    def _on_draw_button_toggled(self, marker_id: int, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self._mushroom_button):
                self._mushroom_button.setChecked(False)
            for other_marker_id, button in self._draw_buttons.items():
                if other_marker_id == marker_id:
                    continue
                with QSignalBlocker(button):
                    button.setChecked(False)
            self._controller.set_tool_mode(manual_draw_tool(marker_id))
        else:
            self._controller.set_tool_mode(None)

    def _on_camera_calibration_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self._projector_calibration_button):
                self._projector_calibration_button.setChecked(False)
            self._controller.set_calibration_mode(CALIBRATION_CAMERA)
        else:
            self._controller.set_calibration_mode(None)

    def _on_projector_calibration_toggled(self, checked: bool) -> None:
        if checked:
            with QSignalBlocker(self._camera_calibration_button):
                self._camera_calibration_button.setChecked(False)
            self._controller.set_calibration_mode(CALIBRATION_PROJECTOR)
        else:
            self._controller.set_calibration_mode(None)

    def _on_show_spectator_toggled(self, checked: bool) -> None:
        if checked:
            self._spectator_window.show()
            self._spectator_window.raise_()
            self._spectator_window.activateWindow()
        else:
            self._spectator_window.set_fullscreen(False)
            self._spectator_window.hide()

    def _on_spectator_closed(self) -> None:
        self._spectator_window.set_fullscreen(False)
        with QSignalBlocker(self._show_spectator_button):
            self._show_spectator_button.setChecked(False)

    def _on_show_projector_toggled(self, checked: bool) -> None:
        if checked:
            self._projector_window.show()
            self._projector_window.raise_()
            self._projector_window.activateWindow()
        else:
            self._projector_window.set_fullscreen(False)
            self._projector_window.hide()

    def _on_projector_closed(self) -> None:
        self._projector_window.set_fullscreen(False)
        with QSignalBlocker(self._show_projector_button):
            self._show_projector_button.setChecked(False)

    def _apply_frame_update(self, frame: np.ndarray, snapshot: TrackerSnapshot) -> None:
        self._video.set_frame(frame)
        self._territory_bar.set_snapshot(snapshot)
        self._time_label.setText(
            "--:--"
            if snapshot.remaining_seconds is None
            else f"{int(snapshot.remaining_seconds) // 60:02d}:{int(snapshot.remaining_seconds) % 60:02d}"
        )
        self._mode_label.setText(tool_label(snapshot.active_tool))
        self._fps_label.setText(f"{snapshot.fps:.1f}")
        self._boost_label.setText(
            "None"
            if not snapshot.buff_remaining
            else ", ".join(f"P{marker_id} {seconds_left:.1f}s" for marker_id, seconds_left in snapshot.buff_remaining.items())
        )
        calibration_mode = snapshot.calibration_mode or "None"
        self._calibration_label.setText(calibration_mode.title())
        with QSignalBlocker(self._mushroom_button):
            self._mushroom_button.setChecked(snapshot.active_tool == ITEM_MUSHROOM)
        active_draw_marker = manual_draw_marker_id(snapshot.active_tool)
        for marker_id, button in self._draw_buttons.items():
            with QSignalBlocker(button):
                button.setChecked(active_draw_marker == marker_id)
        with QSignalBlocker(self._camera_calibration_button):
            self._camera_calibration_button.setChecked(snapshot.calibration_mode == CALIBRATION_CAMERA)
        with QSignalBlocker(self._projector_calibration_button):
            self._projector_calibration_button.setChecked(snapshot.calibration_mode == CALIBRATION_PROJECTOR)

    def _show_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Tracker Error", message)
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        with suppress(RuntimeError):
            self._worker.stop()
        if self._spectator_window.isVisible():
            self._spectator_window.close()
        if self._projector_window.isVisible():
            self._projector_window.close()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cameras:
        list_cameras(args.camera_probe_limit, args.width, args.height)
        return 0
    if args.generate_marker is not None:
        generate_marker(args)
        return 0

    app = QApplication(sys.argv if argv is None else ["hello.py", *argv])
    window = MainWindow(args)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
