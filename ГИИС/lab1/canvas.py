# canvas.py
from PySide6 import QtCore, QtGui, QtWidgets
from linestrategies import PointStep, LineStrategy, DDAStrategy
from logger import StepLogger
from constants import (
    CANVAS_MIN_WIDTH,
    CANVAS_MIN_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_HEIGHT,
    INITIAL_OFFSET_X,
    INITIAL_OFFSET_Y,
    ANIM_TIMER_INTERVAL_MS,
    ALPHA_MAX,
    GRID_STEP,
    GRID_COLOR_R,
    GRID_COLOR_G,
    GRID_COLOR_B,
    AXIS_THICKNESS,
    LABEL_FONT_SIZE,
    LABEL_STEP,
    X_LABEL_OFFSET_X,
    X_LABEL_OFFSET_Y,
    Y_LABEL_OFFSET_X,
    Y_LABEL_OFFSET_Y,
    AXIS_LABEL_FONT_SIZE,
    Y_AXIS_LABEL_OFFSET_X,
    Y_AXIS_LABEL_OFFSET_Y,
    X_AXIS_LABEL_OFFSET_X,
    X_AXIS_LABEL_OFFSET_Y,
    ANIM_PEN_THICKNESS,
    START_PT_PEN_THICKNESS,
)


class Canvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(CANVAS_MIN_WIDTH, CANVAS_MIN_HEIGHT)
        self.canvas_size = QtCore.QSize(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.image = QtGui.QImage(self.canvas_size, QtGui.QImage.Format_ARGB32)
        self.image.fill(QtCore.Qt.white)

        self.strategy: LineStrategy = DDAStrategy()
        self.start_pt = None
        self.mode = "DRAW"
        self.is_debug = False
        self.scale = 1.0
        self.offset = QtCore.QPointF(INITIAL_OFFSET_X, INITIAL_OFFSET_Y)  # Начальная позиция (0,0) снизу-слева
        self.last_pos = QtCore.QPointF()

        self.anim_steps = []
        self.anim_idx = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._on_anim)
        StepLogger.init_file()

    def set_strategy(self, strategy: LineStrategy):
        self.strategy = strategy

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
                    if self.start_pt is None:
                        self.start_pt = (gx, gy)
                    else:
                        self._draw_line(self.start_pt, (gx, gy))
                        self.start_pt = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "PAN" and event.buttons() & QtCore.Qt.LeftButton:
            delta = event.position() - self.last_pos
            self.offset += delta
            self.last_pos = event.position()
            self.update()

    def _draw_line(self, p1, p2):
        steps = self.strategy.calculate(p1[0], p1[1], p2[0], p2[1])
        StepLogger.log_result(self.strategy.__class__.__name__, p1, p2, steps)
        if self.is_debug:
            self.anim_steps = steps;
            self.anim_idx = 0;
            self.timer.start(ANIM_TIMER_INTERVAL_MS)
        else:
            self._draw_to_img(steps)

    def _on_anim(self):
        if self.anim_idx < len(self.anim_steps):
            self.anim_idx += 1; self.update()
        else:
            self.timer.stop(); self._draw_to_img(self.anim_steps)

    def _draw_to_img(self, steps):
        p = QtGui.QPainter(self.image)
        h = self.image.height()
        for s in steps:
            p.setPen(QtGui.QColor(0, 0, 0, int(s.alpha * ALPHA_MAX)))
            p.drawPoint(s.x, (h - 1) - s.y)  # Инверсия для растра
        p.end();
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)

        p.save()
        p.translate(self.offset)
        p.scale(self.scale, self.scale)

        # 1. Белый холст
        h = self.image.height()
        p.drawImage(QtCore.QPoint(0, -h), self.image)

        # 2. Сетка
        step = GRID_STEP
        p.setPen(QtGui.QPen(QtGui.QColor(GRID_COLOR_R, GRID_COLOR_G, GRID_COLOR_B), 0))
        for i in range(0, self.image.width() + 1, step):
            p.drawLine(i, 0, i, -h)
        for i in range(0, h + 1, step):
            p.drawLine(0, -i, self.image.width(), -i)

        # 3. ОСИ
        p.setPen(QtGui.QPen(QtCore.Qt.blue, AXIS_THICKNESS / self.scale))
        p.drawLine(0, 0, self.image.width(), 0)   # X
        p.drawLine(0, 0, 0, -h)                   # Y

        # 4. РАЗМЕТКА ЦИФРАМИ
        font = p.font()
        font.setPointSizeF(LABEL_FONT_SIZE / self.scale)
        p.setFont(font)
        p.setPen(QtCore.Qt.white)

        label_step = LABEL_STEP

        # X-метки
        for i in range(label_step, self.image.width(), label_step):
            p.drawText(i + (X_LABEL_OFFSET_X / self.scale), X_LABEL_OFFSET_Y / self.scale, str(i))

        # Y-метки
        for i in range(label_step, self.image.height(), label_step):
            p.drawText(Y_LABEL_OFFSET_X / self.scale, -i + (Y_LABEL_OFFSET_Y / self.scale), str(i))

        # 5. ПОДПИСИ X и Y
        font.setBold(True)
        font.setPointSizeF(AXIS_LABEL_FONT_SIZE / self.scale)
        p.setFont(font)
        p.setPen(QtCore.Qt.white)

        p.drawText(Y_AXIS_LABEL_OFFSET_X / self.scale, -h + (Y_AXIS_LABEL_OFFSET_Y / self.scale), "Y")
        p.drawText(self.image.width() + (X_AXIS_LABEL_OFFSET_X / self.scale), X_AXIS_LABEL_OFFSET_Y / self.scale, "X")

        # 6. Отрисовка анимации и стартовой точки
        if self.timer.isActive():
            p.setPen(QtGui.QPen(QtCore.Qt.red, ANIM_PEN_THICKNESS / self.scale))
            for i in range(self.anim_idx):
                s = self.anim_steps[i]
                p.drawPoint(s.x, -s.y)

        if self.start_pt:
            p.setPen(QtGui.QPen(QtCore.Qt.red, START_PT_PEN_THICKNESS / self.scale))
            p.drawPoint(self.start_pt[0], -self.start_pt[1])

        p.restore()