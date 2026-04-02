import tkinter as tk
from model import LexemeModel
from view import MorphView
from controller import MorphController


if __name__ == "__main__":
    root = tk.Tk()
    model = LexemeModel()
    view = MorphView(root)
    controller = MorphController(model, view)
    view.controller = controller
    view.build_ui()
    root.mainloop()