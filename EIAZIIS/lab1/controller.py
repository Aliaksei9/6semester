# controller.py
import csv


class MorphController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def load_pdf(self):
        path = self.view.ask_open_filename(filetypes=[("PDF files", "*.pdf")])
        if not path:
            return

        try:
            self.model.load_pdf(path)
            self.model.sort('lemma')
            self.apply_filter()
            total_tokens = self.model.get_total_tokens()
            wordforms_count = self.model.get_wordforms_count()
            lexemes_count = self.model.get_lexeme_count()
            self.view.update_stats(total_tokens, wordforms_count, lexemes_count)
            self.view.show_info("Готово", f"Обработано {total_tokens} слов, {wordforms_count} уникальных словоформ, {lexemes_count} уникальных лексем")
        except Exception as e:
            self.view.show_error("Ошибка", str(e))

    def sort(self, key: str):
        self.model.sort(key)
        self.apply_filter()

    def apply_filter(self):
        query = self.view.get_search_query()
        filtered_data = self.model.filter_data(query)
        self.view.populate_tree(filtered_data)

    def update_morph_info(self, form: str, new_info: str):
        self.model.update_morph_info(form, new_info)
        # Обновляем отображаемые данные (
        self.apply_filter()

    def export_csv(self):
        data = self.model.get_full_data()
        if not data:
            self.view.show_warning("Ничего нет", "Сначала загрузите PDF")
            return
        path = self.view.ask_save_filename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerows([
                ["Словоформа", "Лексема", "Частота словоформы", "Частота лексемы", "Морфологическая информация"]
            ] + [[d["form"], d["lemma"], d["form_freq"], d["lemma_freq"], d["morph_info"]] for d in data])
        self.view.show_info("Экспорт", f"Сохранено → {path}")

    def export_txt(self):
        data = self.model.get_full_data()
        if not data:
            self.view.show_warning("Ничего нет", "Сначала загрузите PDF")
            return
        path = self.view.ask_save_filename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== Лексемы и словоформы ===\n\n")
            for d in data:
                f.write(f"Словоформа: {d['form']}\nЛексема:    {d['lemma']}\nЧастота словоформы: {d['form_freq']}\n"
                        f"Частота лексемы: {d['lemma_freq']}\nМорфология: {d['morph_info'] or '—'}\n{'-'*60}\n\n")
        self.view.show_info("Экспорт", f"Сохранено → {path}")

    def show_help(self):
        text = self.model.get_help_text()
        self.view.show_info("Помощь", text)

    def show_terminology(self):
        text = self.model.get_terminology_text()
        self.view.show_info("Терминология", text)