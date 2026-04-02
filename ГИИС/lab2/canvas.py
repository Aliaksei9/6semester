from typing import List, Optional, Tuple
from PySide6 import QtCore, QtGui, QtWidgets
from constants import *
from managers import LineManager, CurveManager
from grid_renderer import GridRenderer
from line_strategies import PointStep
from line_strategies import LineStrategy
from curve_strategies import CurveStrategy
from step_logger import StepLogger

class Canvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(CANVAS_MIN_WIDTH, CANVAS_MIN_HEIGHT)
        self.canvas_size = QtCore.QSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.image = QtGui.QImage(self.canvas_size, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.white)
        self.line_manager = LineManager(self)
        self.curve_manager = CurveManager(self)
        self.grid_renderer = GridRenderer()
        self.start_pt: Optional[tuple[int, int]] = None
        self.current_end: Optional[tuple[int, int]] = None
        self.mode = "LINE"
        self.is_debug = False
        self.scale = 1.0
        self.offset = QtCore.QPointF(INITIAL_OFFSET_X, INITIAL_OFFSET_Y)
        self.last_pos = QtCore.QPointF()
        self.anim_steps: List[PointStep] = []
        self.anim_idx = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_anim)
        StepLogger.init_file()

    def set_line_strategy(self, strategy: LineStrategy):
        self.line_manager.set_strategy(strategy)

    def set_curve_strategy(self, strategy: CurveStrategy):
        self.curve_manager.set_strategy(strategy)

    def _to_logical(self, pos: QtCore.QPointF):
        lx = (pos.x() - self.offset.x()) / self.scale
        ly = -(pos.y() - self.offset.y()) / self.scale
        return int(round(lx)), int(round(ly))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.mode == "PAN":
                self.last_pos = event.position()
            else:
                gx, gy = self._to_logical(event.position())
                if 0 <= gx < self.canvas_size.width() and 0 <= gy < self.canvas_size.height():
                    if self.mode == "LINE":
                        if self.start_pt is None:
                            self.start_pt = (gx, gy)
                        else:
                            steps = self.line_manager.calculate_steps(self.start_pt, (gx, gy))
                            if steps:
                                name = self.line_manager.get_name()
                                StepLogger.log_result(name, self.start_pt, (gx, gy), steps)
                                if self.is_debug:
                                    self.anim_steps = steps
                                    self.anim_idx = 0
                                    self.timer.start(ANIM_TIMER_INTERVAL_MS)
                                else:
                                    self._draw_to_img(steps)
                            self.start_pt = None
                    elif self.mode == "CURVE":
                        if self.start_pt is None:
                            self.start_pt = (gx, gy)
                        self.current_end = (gx, gy)
            self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "PAN" and event.buttons() & QtCore.Qt.LeftButton:
            delta = event.position() - self.last_pos
            self.offset += delta
            self.last_pos = event.position()
            self.update()
        elif self.mode == "CURVE" and event.buttons() & QtCore.Qt.LeftButton and self.start_pt:
            gx, gy = self._to_logical(event.position())
            self.current_end = (gx, gy)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.mode == "CURVE" and self.start_pt and self.current_end:
                steps = self.curve_manager.calculate_steps(self.start_pt, self.current_end)
                if steps:
                    name = self.curve_manager.get_name()
                    StepLogger.log_result(name, self.start_pt, self.current_end, steps)
                    if self.is_debug:
                        self.anim_steps = steps
                        self.anim_idx = 0
                        self.timer.start(ANIM_TIMER_INTERVAL_MS)
                    else:
                        self._draw_to_img(steps)
                self.start_pt = None
                self.current_end = None
                self.update()

    def _on_anim(self):
        if self.anim_idx < len(self.anim_steps):
            self.anim_idx += 1
            self.update()
        else:
            self.timer.stop()
            self._draw_to_img(self.anim_steps)

    def _draw_to_img(self, steps: List[PointStep]):
        p = QtGui.QPainter(self.image)
        h = self.image.height()
        for s in steps:
            if 0 <= s.x < self.canvas_size.width() and 0 <= (h - 1 - s.y) < self.canvas_size.height():
                p.setPen(QtGui.QColor(0, 0, 0, int(s.alpha * ALPHA_MAX)))
                p.drawPoint(s.x, (h - 1) - s.y)
        p.end()
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.save()
        p.translate(self.offset)
        p.scale(self.scale, self.scale)
        h = self.image.height()
        p.drawImage(QtCore.QPoint(0, -h), self.image)
        self.grid_renderer.draw(p, self.canvas_size, self.scale)
        if self.timer.isActive():
            p.setPen(QtGui.QPen(QtCore.Qt.red, ANIM_PEN_THICKNESS / self.scale))
            for i in range(self.anim_idx):
                s = self.anim_steps[i]
                p.drawPoint(s.x, -s.y)
        if self.start_pt:
            p.setPen(QtGui.QPen(QtCore.Qt.red, START_PT_PEN_THICKNESS / self.scale))
            p.drawPoint(self.start_pt[0], -self.start_pt[1])
        if self.mode == "CURVE" and self.start_pt and self.current_end:
            steps = self.curve_manager.calculate_steps(self.start_pt, self.current_end)
            if steps:
                p.setPen(QtGui.QPen(QtCore.Qt.green, ANIM_PEN_THICKNESS / self.scale))
                for s in steps:
                    p.drawPoint(s.x, -s.y)
        p.restore()