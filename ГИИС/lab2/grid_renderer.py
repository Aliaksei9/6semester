from PySide6 import QtCore, QtGui
from constants import *

class GridRenderer:
    def draw(self, p: QtGui.QPainter, image_size: QtCore.QSize, scale: float):
        h = image_size.height()
        w = image_size.width()
        step = GRID_STEP
        p.setPen(QtGui.QPen(QtGui.QColor(GRID_COLOR_R, GRID_COLOR_G, GRID_COLOR_B), 0))
        for i in range(0, w + 1, step):
            p.drawLine(i, 0, i, -h)
        for i in range(0, h + 1, step):
            p.drawLine(0, -i, w, -i)
        p.setPen(QtGui.QPen(QtCore.Qt.blue, AXIS_THICKNESS / scale))
        p.drawLine(0, 0, w, 0)
        p.drawLine(0, 0, 0, -h)
        font = p.font()
        font.setPointSizeF(LABEL_FONT_SIZE / scale)
        p.setFont(font)
        p.setPen(QtCore.Qt.white)
        label_step = LABEL_STEP
        for i in range(label_step, w, label_step):
            p.drawText(i + X_LABEL_OFFSET_X / scale, X_LABEL_OFFSET_Y / scale, str(i))
        for i in range(label_step, h, label_step):
            p.drawText(Y_LABEL_OFFSET_X / scale, -i + Y_LABEL_OFFSET_Y / scale, str(i))
        font.setBold(True)
        font.setPointSizeF(AXIS_LABEL_FONT_SIZE / scale)
        p.setFont(font)
        p.setPen(QtCore.Qt.white)
        p.drawText(Y_AXIS_LABEL_OFFSET_X / scale, -h + Y_AXIS_LABEL_OFFSET_Y / scale, "Y")
        p.drawText(w + X_AXIS_LABEL_OFFSET_X / scale, X_AXIS_LABEL_OFFSET_Y / scale, "X")