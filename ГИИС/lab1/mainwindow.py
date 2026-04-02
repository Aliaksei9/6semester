# mainwindow.py
# mainwindow.py
from PySide6 import QtCore, QtGui, QtWidgets
from linestrategies import DDAStrategy, BresenhamStrategy, WuStrategy, LineStrategy
from canvas import Canvas
from constants import ZOOM_IN_FACTOR, ZOOM_OUT_FACTOR


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Line Editor: Cartesian System")
        self.canvas = Canvas()
        self.setCentralWidget(self.canvas)

        #МЕНЮ
        menubar = self.menuBar()
        line_menu = menubar.addMenu("Отрезки")

        #Панель инструментов
        tools_tb = self.addToolBar("Tools")
        tools_tb.setObjectName("Tools")

        #Режимы: Рука и Отрезки
        act_line = QtGui.QAction("Отрезки", self, checkable=True)   # ← новый режим
        act_line.setChecked(True)                                   # по умолчанию включено
        act_line.triggered.connect(lambda: setattr(self.canvas, 'mode', "DRAW"))

        act_pan = QtGui.QAction("Рука", self, checkable=True)
        act_pan.triggered.connect(lambda: setattr(self.canvas, 'mode', "PAN"))

        mode_group = QtGui.QActionGroup(self)
        mode_group.addAction(act_line)
        mode_group.addAction(act_pan)

        tools_tb.addActions([act_line, act_pan])
        tools_tb.addSeparator()

        #Панель «Отрезки» (выбор алгоритма)
        line_tb = self.addToolBar("Отрезки")
        line_tb.setObjectName("Отрезки")

        self.strats = {
            "ЦДА":        DDAStrategy(),
            "Брезенхем":  BresenhamStrategy(),
            "Ву":         WuStrategy()
        }

        strat_group = QtGui.QActionGroup(self)
        strat_group.setExclusive(True)

        for name, strategy in self.strats.items():
            act = QtGui.QAction(name, self, checkable=True)
            act.triggered.connect(lambda checked, s=strategy: self._set_strategy(s) if checked else None)
            strat_group.addAction(act)
            line_menu.addAction(act)
            line_tb.addAction(act)

            if name == "ЦДА":
                act.setChecked(True)

        line_tb.addSeparator()

        # Пошаговый режим
        chk = QtWidgets.QCheckBox("Пошагово")
        chk.toggled.connect(lambda v: setattr(self.canvas, 'is_debug', v))
        line_tb.addWidget(chk)

        # Зум и очистка
        tools_tb.addAction("Zoom +", lambda: self._zoom(ZOOM_IN_FACTOR))
        tools_tb.addAction("Zoom -", lambda: self._zoom(ZOOM_OUT_FACTOR))
        tools_tb.addAction("Очистить", self._reset)

        # Начальная стратегия
        self._set_strategy(self.strats["ЦДА"])

    def _set_strategy(self, strategy: LineStrategy):
        self.canvas.set_strategy(strategy)

    def _zoom(self, f: float):
        self.canvas.scale *= f
        self.canvas.update()

    def _reset(self):
        self.canvas.image.fill(QtCore.Qt.white)
        self.canvas.update()