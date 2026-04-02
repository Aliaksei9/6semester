from unittest.mock import DEFAULT

import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
import fitz  # PyMuPDF вместо pypdf
from docx import Document
try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None

import nltk
from razdel import tokenize
import pymorphy3
from performance_timer import PerformanceTimer

from constants import (
    MIN_YEAR, MAX_YEAR, DEFAULT_YEAR,
    POS_TRANSLATE, GRAMMEMES_TRANSLATE,
    TERMINOLOGY_TEXT, HELP_TEXT, MENU_ITEMS, CORPUS_COLUMNS
)

nltk.download('punkt', quiet=True)


class TagTranslator:
    """Класс для перевода морфологических тегов на русский язык."""

    POS_TRANSLATE = POS_TRANSLATE
    GRAMMEMES_TRANSLATE = GRAMMEMES_TRANSLATE

    def translate(self, tag_str):
        if not tag_str or tag_str == 'None':
            return ''
        parts = str(tag_str).replace(',', ', ').split()
        translated = []
        for part in parts:
            part = part.strip(', ')
            sub = [self.GRAMMEMES_TRANSLATE.get(p.strip(), "") for p in part.split(',')]
            translated.append(', '.join(sub))
        return ' '.join(translated).strip()

    def get_pos_rus(self, pos):
        return self.POS_TRANSLATE.get(pos, pos or "NONE")


class FileHandler:
    def extract_text(self, uploaded_file):
        name = uploaded_file.name.lower()
        data = uploaded_file.getvalue()
        if name.endswith('.txt'):
            return data.decode('utf-8', errors='ignore')
        elif name.endswith('.pdf'):
            doc = fitz.open(stream=data, filetype="pdf")
            text = '\n'.join(page.get_text() or '' for page in doc)
            doc.close()
            return text
        elif name.endswith('.docx'):
            doc = Document(BytesIO(data))
            return '\n'.join(p.text for p in doc.paragraphs)
        elif name.endswith('.rtf') and rtf_to_text:
            return rtf_to_text(data.decode('utf-8', errors='ignore'))
        return None


class TextParser:
    """Класс для разбора текста: токенизация и морфологический анализ."""

    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        self.translator = TagTranslator()

    def analyze_text(self, text: str, metadata: dict, doc_id: int):
        tokens = [token.text for token in tokenize(text) if re.match(r'^\w+$', token.text)]
        rows = []
        for token in tokens:
            if token.isdigit():  # числа исключены
                continue
            p = self.morph.parse(token)[0]
            pos = p.tag.POS or 'UNK'
            morph_tag = str(p.tag)
            rows.append({
                'doc_id': doc_id,
                'wordform': token.lower(),
                'lemma': p.normal_form.lower(),
                'pos_rus': self.translator.get_pos_rus(pos),
                'morph_rus': self.translator.translate(morph_tag),
                'doc_title': metadata['title'],
                'author': metadata['author'],
                'year': metadata['year'],
                'type': metadata['type']
            })
        return rows


class DataStorage:
    """Класс для хранения данных корпуса."""

    def __init__(self):
        if 'corpus_df' not in st.session_state:
            st.session_state.corpus_df = pd.DataFrame(columns=CORPUS_COLUMNS)
        if 'docs' not in st.session_state:
            st.session_state.docs = {}
        if 'next_doc_id' not in st.session_state:
            st.session_state.next_doc_id = 1

    def add_to_corpus(self, rows):
        if rows:
            new_df = pd.DataFrame(rows)
            st.session_state.corpus_df = pd.concat([st.session_state.corpus_df, new_df], ignore_index=True)

    def get_corpus_df(self):
        return st.session_state.corpus_df

    def get_docs(self):
        return st.session_state.docs

    def set_doc(self, doc_id, text, metadata):
        st.session_state.docs[doc_id] = {'text': text, 'metadata': metadata}

    def get_next_doc_id(self):
        return st.session_state.next_doc_id

    def increment_doc_id(self):
        st.session_state.next_doc_id += 1

    def remove_from_corpus(self, doc_id):
        st.session_state.corpus_df = st.session_state.corpus_df[st.session_state.corpus_df['doc_id'] != doc_id]

    def save_to_json(self):
        data = {
            "docs": st.session_state.docs,
            "corpus_df": st.session_state.corpus_df.to_dict(orient='records'),
            "next_doc_id": st.session_state.next_doc_id
        }
        return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')

    def load_from_json(self, data):
        loaded = json.loads(data)
        st.session_state.docs = loaded['docs']
        st.session_state.corpus_df = pd.DataFrame(loaded['corpus_df'])
        st.session_state.next_doc_id = loaded.get('next_doc_id', 1)


class View:

    def __init__(self, corpus_manager):
        self.corpus_manager = corpus_manager
        st.set_page_config(page_title="Корпусный менеджер", layout="wide")
        st.title("Корпусный менеджер текстов")

    def get_phrase_context(self, text: str, phrase: str, width: int = 90):
        pattern = re.compile(r'(.{0,' + str(width) + r'})(' + re.escape(phrase) + r')(.{0,' + str(width) + r'})', re.I)
        return pattern.findall(text)

    def display_menu(self):
        menu = st.sidebar.selectbox(
            "Меню",
            MENU_ITEMS
        )
        self.corpus_manager.handle_menu_selection(menu)

    def render_upload(self):
        st.header("Загрузка текстов в корпус")
        uploaded_files = st.file_uploader(
            "Выберите файлы (TXT, PDF, DOCX, RTF)",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx', 'rtf']
        )

        st.subheader("Метаданные")
        col1, col2 = st.columns(2)
        with col1:
            author = st.text_input("Автор", "Неизвестен")
            title_base = st.text_input("Базовое название", "Документ")
        with col2:
            year = st.number_input("Год", MIN_YEAR, MAX_YEAR, DEFAULT_YEAR)
            doc_type = st.selectbox("Тип текста", ["Художественный", "Научный", "Публицистический", "Другой"])

        process_button = st.button(" Обработать и добавить в корпус", type="primary", key="process_files_button")
        return uploaded_files, author, title_base, year, doc_type, process_button

    def render_view_edit(self):
        st.header(" Просмотр и редактирование корпуса")
        docs = self.corpus_manager.storage.get_docs()
        if not docs:
            st.info("Корпус пока пустой")
            return None, None, False

        selected_id = st.selectbox("Выберите документ", list(docs.keys()),
                                   format_func=lambda x: f"{x} — {docs[x]['metadata']['title']}")
        doc = docs[selected_id]
        edited_text = st.text_area("Редактировать текст", doc['text'], height=400)
        save_button = st.button("Сохранить изменения", key="save_changes_button")
        return selected_id, edited_text, save_button

    def render_search_analysis(self):
        st.header("Поиск и анализ корпуса")
        df = self.corpus_manager.storage.get_corpus_df()
        if df.empty:
            st.warning("Сначала загрузите документы в раздел «Загрузка»")
            return

        tab_stats, tab_word, tab_phrase, tab_filter = st.tabs([
            "Общая статистика корпуса",
            "Поиск словоформ и лемм",
            "Поиск по фразе / кускам текста",
            "Фильтры по метаданным"
        ])

        with tab_stats:
            self.display_stats(df)

        with tab_word:
            query = self.render_word_search()
            self.display_word_search_results(df, query)

        with tab_phrase:
            phrase = self.render_phrase_search()
            self.display_phrase_search_results(phrase)

        with tab_filter:
            self.render_filters(df)

    def display_stats(self, df):
        st.subheader("Частотные характеристики по всему корпусу")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Топ-20 словоформ**")
            wf = df['wordform'].value_counts().head(20).reset_index()
            wf.columns = ['Словоформа', 'Частота']
            st.dataframe(wf, use_container_width=True)

            st.write("**Топ-20 лемм**")
            lem = df['lemma'].value_counts().head(20).reset_index()
            lem.columns = ['Лемма', 'Частота']
            st.dataframe(lem, use_container_width=True)

        with col2:
            st.write("**Распределение по частям речи**")
            pos_df = df['pos_rus'].value_counts().reset_index()
            pos_df.columns = ['Часть речи', 'Количество']
            st.dataframe(pos_df, use_container_width=True)

        st.write("**Морфологические характеристики (топ-15)**")
        morph_df = df['morph_rus'].replace('', pd.NA).dropna().value_counts().head(15).reset_index()
        morph_df.columns = ['Морфологические характеристики', 'Частота']
        st.dataframe(morph_df, use_container_width=True)

        st.write("**Статистика по документам**")
        doc_stat = df.groupby('doc_title').size().reset_index(name='Количество словоформ')
        doc_stat = doc_stat.rename(columns={'doc_title': 'Название документа'})
        st.dataframe(doc_stat, use_container_width=True)

    def render_word_search(self):
        query = st.text_input("Введите слово или лемму для поиска")
        return query

    def display_word_search_results(self, df, query):
        if query:
            mask = (
                df['wordform'].str.contains(query.lower(), na=False) |
                df['lemma'].str.contains(query.lower(), na=False)
            )
            result = df[mask][['doc_title', 'wordform', 'lemma', 'pos_rus', 'morph_rus', 'author', 'year', 'type']].copy()
            result = result.rename(columns={
                'doc_title': 'Название документа',
                'wordform': 'Словоформа',
                'lemma': 'Лемма',
                'pos_rus': 'Часть речи',
                'morph_rus': 'Морфологические характеристики',
                'author': 'Автор',
                'year': 'Год',
                'type': 'Тип текста'
            })
            st.dataframe(result, use_container_width=True)

    def render_phrase_search(self):
        st.subheader("Поиск по произвольному тексту / словосочетанию")
        phrase = st.text_input("Введите фразу или кусок текста", placeholder="Пользовательский интерфейс")
        return phrase

    def display_phrase_search_results(self, phrase):
        if phrase:
            st.write(f"**Результаты поиска:** `{phrase}`")
            found_any = False
            docs = self.corpus_manager.storage.get_docs()
            for doc_id, doc_data in docs.items():
                contexts = self.get_phrase_context(doc_data['text'], phrase)
                if contexts:
                    found_any = True
                    title = doc_data['metadata']['title']
                    st.subheader(f" {title}")
                    for left, match, right in contexts[:6]:
                        st.markdown(f"...{left}**{match}**{right}...")
                    st.divider()
            if not found_any:
                st.info("Фраза не найдена ни в одном документе корпуса.")

    def render_filters(self, df):
        st.subheader("Фильтрация по метаданным")
        authors = ["Все"] + sorted(df['author'].unique().tolist())
        types_list = ["Все"] + sorted(df['type'].unique().tolist())
        years = ["Все"] + sorted(df['year'].unique().tolist(), reverse=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            sel_author = st.selectbox("Автор", authors)
        with col2:
            sel_type = st.selectbox("Тип текста", types_list)
        with col3:
            sel_year = st.selectbox("Год", years)

        filtered = df.copy()
        if sel_author != "Все":
            filtered = filtered[filtered['author'] == sel_author]
        if sel_type != "Все":
            filtered = filtered[filtered['type'] == sel_type]
        if sel_year != "Все":
            filtered = filtered[filtered['year'] == sel_year]

        display = filtered[['doc_title', 'wordform', 'lemma', 'pos_rus', 'morph_rus', 'author', 'year', 'type']].copy()
        display = display.rename(columns={
            'doc_title': 'Название документа',
            'wordform': 'Словоформа',
            'lemma': 'Лемма',
            'pos_rus': 'Часть речи',
            'morph_rus': 'Морфологические характеристики',
            'author': 'Автор',
            'year': 'Год',
            'type': 'Тип текста'
        })
        st.dataframe(display, use_container_width=True)

    def render_save_load(self):
        st.header("Сохранение и загрузка корпуса")
        col1, col2 = st.columns(2)
        save_button = False
        uploaded_json = None
        with col1:
            save_button = st.button("Сохранить корпус как JSON", key="save_json_button")
        with col2:
            uploaded_json = st.file_uploader("Загрузить сохранённый корпус", type='json')
        return save_button, uploaded_json

    def display_saved_json(self, data):
        st.download_button(
            label="Скачать corpus.json",
            data=data,
            file_name="corpus.json",
            mime="application/json",
            key="download_corpus"
        )

    def render_help(self):
        st.header("Система помощи")
        st.markdown(HELP_TEXT)

    def render_terminology(self):
        st.header("Терминологическая справка")
        st.markdown(TERMINOLOGY_TEXT)

    def render_sidebar_caption(self):
        st.sidebar.caption("Корпусный менеджер v0.999")


class CorpusManager:
    """Координирующий класс для управления всем процессом."""

    def __init__(self):
        self.storage = DataStorage()
        self.file_handler = FileHandler()
        self.parser = TextParser()
        self.view = View(self)

    def add_document(self, text: str, metadata: dict):
        doc_id = self.storage.get_next_doc_id()
        rows = self.parser.analyze_text(text, metadata, doc_id)
        self.storage.add_to_corpus(rows)
        self.storage.set_doc(doc_id, text, metadata)
        self.storage.increment_doc_id()
        return doc_id

    def update_document(self, doc_id: int, new_text: str):
        metadata = self.storage.get_docs()[doc_id]['metadata']
        self.storage.remove_from_corpus(doc_id)
        rows = self.parser.analyze_text(new_text, metadata, doc_id)
        self.storage.add_to_corpus(rows)
        self.storage.set_doc(doc_id, new_text, metadata)

    def process_uploaded_files(self, uploaded_files, author, title_base, year, doc_type):
        PerformanceTimer.start()
        for file in uploaded_files:
            text = self.file_handler.extract_text(file)
            if text:
                metadata = {
                    "title": f"{title_base} — {file.name}",
                    "author": author,
                    "year": year,
                    "type": doc_type,
                    "filename": file.name
                }
                doc_id = self.add_document(text, metadata)
                st.success(f"{file.name} добавлен (ID {doc_id})")
        PerformanceTimer.stop()

    def handle_menu_selection(self, menu):
        if menu == "Загрузка и построение корпуса":
            self.handle_upload()
        elif menu == "Просмотр и редактирование корпуса":
            self.handle_view_edit()
        elif menu == "Поиск и анализ":
            self.handle_search_analysis()
        elif menu == "Сохранение / Загрузка":
            self.handle_save_load()
        elif menu == "Терминологическая справка":
            self.handle_terminology()
        else:
            self.handle_help()
        self.view.render_sidebar_caption()

    def handle_upload(self):
        uploaded_files, author, title_base, year, doc_type, process_button = self.view.render_upload()
        if uploaded_files and process_button:
            self.process_uploaded_files(uploaded_files, author, title_base, year, doc_type)

    def handle_view_edit(self):
        selected_id, edited_text, save_button = self.view.render_view_edit()
        if save_button and selected_id is not None:
            self.update_document(selected_id, edited_text)
            st.success("Изменения сохранены и корпус обновлён")

    def handle_search_analysis(self):
        self.view.render_search_analysis()

    def handle_save_load(self):
        save_button, uploaded_json = self.view.render_save_load()
        if save_button:
            data = self.storage.save_to_json()
            self.view.display_saved_json(data)
        if uploaded_json:
            data = uploaded_json.getvalue()
            self.storage.load_from_json(data)
            st.success("Корпус успешно загружен!")

    def handle_terminology(self):
        self.view.render_terminology()

    def handle_help(self):
        self.view.render_help()

    def run(self):
        self.view.display_menu()


if __name__ == "__main__":
    manager = CorpusManager()
    manager.run()