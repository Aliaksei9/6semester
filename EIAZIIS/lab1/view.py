import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict


class MorphView:
    def __init__(self, root):
        self.root = root
        self.root.title("Лексемы и словоформы")
        self.root.geometry("1250x720")
        self.controller = None

    def build_ui(self):
        self._create_menu()
        self._create_widgets()

    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Открыть PDF...", command=self.controller.load_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Экспорт в CSV...", command=self.controller.export_csv)
        file_menu.add_command(label="Экспорт в TXT...", command=self.controller.export_txt)
        file_menu.add_separator()

        # Сортировка
        sort_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Сортировка", menu=sort_menu)
        sort_menu.add_command(label="По лексемам", command=lambda: self.controller.sort('lemma'))
        sort_menu.add_command(label="По словоформам", command=lambda: self.controller.sort('form'))
        sort_menu.add_command(label="По частоте словоформ", command=lambda: self.controller.sort('form_freq'))
        sort_menu.add_command(label="По частоте лексем", command=lambda: self.controller.sort('lemma_freq'))

        # Помощь
        menubar.add_command(label="Помощь", command=self.controller.show_help)

        # Терминология
        menubar.add_command(label="Терминология", command=self.controller.show_terminology)

    def _create_widgets(self):
        # Кнопка загрузки
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10, fill="x", padx=10)
        tk.Button(top_frame, text="Загрузить PDF", width=20, font=("Arial", 10, "bold"),
                  command=self.controller.load_pdf).pack(side="left")

        # Статистика
        self.stats_label = tk.Label(top_frame, text="", font=("Arial", 10))
        self.stats_label.pack(side="left", padx=20)

        # Поиск
        search_frame = tk.Frame(self.root)
        search_frame.pack(fill="x", padx=10, pady=(0, 10))
        tk.Label(search_frame, text="Поиск:").pack(side="left")
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=60)
        self.search_entry.pack(side="left", padx=10, fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", lambda e: self.controller.apply_filter())

        # Таблица
        columns = ("form", "lemma", "form_freq", "lemma_freq", "morph_info")
        self.tree = ttk.Treeview(self.root, columns=columns, show="headings", height=30)
        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=5)
        self.tree.heading("form", text="Словоформа")
        self.tree.heading("lemma", text="Лексема")
        self.tree.heading("form_freq", text="Частота словоформы")
        self.tree.heading("lemma_freq", text="Частота лексемы")
        self.tree.heading("morph_info", text="Морфологическая информация")
        self.tree.column("form", width=300)
        self.tree.column("lemma", width=220)
        self.tree.column("form_freq", width=120, anchor="center")
        self.tree.column("lemma_freq", width=90, anchor="center")
        self.tree.column("morph_info", width=330)

        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.bind("<Double-1>", self._on_double_click)

    def _on_double_click(self, event):
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return
        if self.tree.identify_column(event.x) != "#5":
            return
        values = self.tree.item(item_id, "values")
        form = values[0]
        current_text = values[4]

        edit_win = tk.Toplevel(self.root)
        edit_win.title("Морфологическая информация")
        edit_win.geometry("620x420")
        tk.Label(edit_win, text="Введите морфологическую информацию:").pack(pady=10)
        text_widget = tk.Text(edit_win, wrap="word", height=16, width=70)
        text_widget.insert("1.0", current_text)
        text_widget.pack(padx=15, pady=10)

        def save():
            new_info = text_widget.get("1.0", "end").strip()
            self.controller.update_morph_info(form, new_info)
            edit_win.destroy()

        tk.Button(edit_win, text="Сохранить", command=save, bg="#4CAF50", fg="white",
                  font=("Arial", 10, "bold")).pack(pady=10)

    def update_stats(self, total_tokens: int, wordforms_count: int, lexemes_count: int):
        self.stats_label.config(text=f"Всего слов: {total_tokens}, Уникальных словоформ: {wordforms_count}, Уникальных лексем: {lexemes_count}")

    def populate_tree(self, data: List[Dict]):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for item in data:
            self.tree.insert("", "end", values=(
                item["form"],
                item["lemma"],
                item["form_freq"],
                item["lemma_freq"],
                item["morph_info"]
            ))

    def get_search_query(self) -> str:
        return self.search_var.get()

    def show_info(self, title: str, message: str):
        messagebox.showinfo(title, message)

    def show_error(self, title: str, message: str):
        messagebox.showerror(title, message)

    def show_warning(self, title: str, message: str):
        messagebox.showwarning(title, message)

    def ask_open_filename(self, filetypes):
        return filedialog.askopenfilename(filetypes=filetypes)

    def ask_save_filename(self, defaultextension, filetypes):
        return filedialog.asksaveasfilename(defaultextension=defaultextension, filetypes=filetypes)