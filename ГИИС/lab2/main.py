from PySide6 import QtWidgets
from main_window import MainWindow
from constants import WINDOW_WIDTH, WINDOW_HEIGHT

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    win.show()
    app.exec()