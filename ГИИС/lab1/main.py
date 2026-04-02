# main.py
from PySide6 import QtWidgets
from mainwindow import MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    win = MainWindow()
    win.resize(1100, 850)
    win.show()
    app.exec()