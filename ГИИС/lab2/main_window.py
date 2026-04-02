from PySide6 import QtCore, QtGui, QtWidgets
from constants import *
from canvas import Canvas
from line_strategies import DDAStrategy, BresenhamStrategy, WuStrategy
from curve_strategies import CircleBresenhamStrategy, EllipseStrategy, HyperbolaStrategy, ParabolaStrategy

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Editor: Cartesian System")
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)
        menubar = self.menuBar()
        line_menu = menubar.addMenu("Отрезки")
        curve_menu = menubar.addMenu("Линии второго порядка")
        tools_tb = self.addToolBar("Tools")
        tools_tb.setObjectName("Tools")
        act_line = QtGui.QAction("Отрезки", self, checkable=True)
        act_line.setChecked(True)
        act_line.triggered.connect(lambda: setattr(self.canvas, 'mode', "LINE"))
        act_curve = QtGui.QAction("Кривые", self, checkable=True)
        act_curve.triggered.connect(lambda: setattr(self.canvas, 'mode', "CURVE"))
        act_pan = QtGui.QAction("Рука", self, checkable=True)
        act_pan.triggered.connect(lambda: setattr(self.canvas, 'mode', "PAN"))
        mode_group = QtGui.QActionGroup(self)
        mode_group.addAction(act_line)
        mode_group.addAction(act_curve)
        mode_group.addAction(act_pan)
        tools_tb.addActions([act_line, act_curve, act_pan])
        tools_tb.addSeparator()
        line_tb = self.addToolBar("Отрезки")
        line_tb.setObjectName("Отрезки")
        self.line_strats = {
            "ЦДА":        DDAStrategy(),
            "Брезенхем":  BresenhamStrategy(),
            "Ву":         WuStrategy()
        }
        line_strat_group = QtGui.QActionGroup(self)
        line_strat_group.setExclusive(True)
        for name, strategy in self.line_strats.items():
            act = QtGui.QAction(name, self, checkable=True)
            act.triggered.connect(lambda checked, s=strategy: self._set_line_strategy(s) if checked else None)
            line_strat_group.addAction(act)
            line_menu.addAction(act)
            line_tb.addAction(act)
            if name == "ЦДА":
                act.setChecked(True)
        line_tb.addSeparator()
        curve_tb = self.addToolBar("Линии второго порядка")
        curve_tb.setObjectName("Линии второго порядка")
        self.curve_strats = {
            "Окружность": CircleBresenhamStrategy(),
            "Эллипс":     EllipseStrategy(),
            "Гипербола":  HyperbolaStrategy(),
            "Парабола":   ParabolaStrategy()
        }
        curve_strat_group = QtGui.QActionGroup(self)
        curve_strat_group.setExclusive(True)
        for name, strategy in self.curve_strats.items():
            act = QtGui.QAction(name, self, checkable=True)
            act.triggered.connect(lambda checked, s=strategy: self._set_curve_strategy(s) if checked else None)
            curve_strat_group.addAction(act)
            curve_menu.addAction(act)
            curve_tb.addAction(act)
            if name == "Окружность":
                act.setChecked(True)
        curve_tb.addSeparator()
        chk = QtWidgets.QCheckBox("Пошагово")
        chk.toggled.connect(lambda v: setattr(self.canvas, 'is_debug', v))
        tools_tb.addWidget(chk)
        tools_tb.addAction("Zoom +", lambda: self._zoom(ZOOM_IN_FACTOR))
        tools_tb.addAction("Zoom -", lambda: self._zoom(ZOOM_OUT_FACTOR))
        tools_tb.addAction("Очистить", self._reset)
        self._set_line_strategy(self.line_strats["ЦДА"])
        self._set_curve_strategy(self.curve_strats["Окружность"])

    def _set_line_strategy(self, strategy):
        self.canvas.set_line_strategy(strategy)

    def _set_curve_strategy(self, strategy):
        self.canvas.set_curve_strategy(strategy)

    def _zoom(self, f: float):
        self.canvas.scale *= f
        self.canvas.update()

    def _reset(self):
        self.canvas.image.fill(QtCore.Qt.white)
        self.canvas.update()