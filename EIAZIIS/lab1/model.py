import fitz
from collections import Counter
import pymorphy3
from razdel import tokenize
import re
from typing import List, Dict


class LexemeModel:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer(lang='ru')
        self.full_data: List[Dict] = []   # [{form, lemma, form_freq, lemma_freq, morph_info}, ...]
        self.lexeme_count: int = 0
        self.total_tokens: int = 0

    @staticmethod
    def extract_words(pdf_path: str) -> List[str]:
        try:
            doc = fitz.open(pdf_path)
            raw_text = "\n".join(page.get_text() for page in doc)
            doc.close()

            # Переносы
            text = re.sub(r'-\s*\n\s*', '', raw_text)
            text = re.sub(r'\s+', ' ', text)

            words = [token.text.lower() for token in tokenize(text) if token.text.isalpha()]
            return words
        except Exception as e:
            raise RuntimeError(f"Ошибка при чтении PDF: {e}")

    def process_words(self, words: List[str]):
        freq = Counter(words)
        self.total_tokens = sum(freq.values())
        self.full_data.clear()
        seen = set()

        for word in sorted(freq.keys()):
            if word in seen:
                continue
            seen.add(word)
            lemma = self.morph.parse(word)[0].normal_form
            self.full_data.append({
                "form": word,
                "lemma": lemma,
                "form_freq": freq[word],
                "morph_info": ""
            })

        # Подсчёт частот лексем
        lemma_freq_counter = Counter()
        for item in self.full_data:
            lemma_freq_counter[item['lemma']] += item['form_freq']

        for item in self.full_data:
            item['lemma_freq'] = lemma_freq_counter[item['lemma']]

        # Подсчёт уникальных лексем
        self.lexeme_count = len(lemma_freq_counter)

    def load_pdf(self, pdf_path: str):
        words = self.extract_words(pdf_path)
        self.process_words(words)

    def sort(self, key: str):
        if key == 'lemma':
            self.full_data.sort(key=lambda x: (x["lemma"], x["form"]))
        elif key == 'form':
            self.full_data.sort(key=lambda x: x["form"])
        elif key == 'form_freq':
            self.full_data.sort(key=lambda x: -x["form_freq"])
        elif key == 'lemma_freq':
            self.full_data.sort(key=lambda x: -x["lemma_freq"])

    def update_morph_info(self, form: str, new_info: str):
        for item in self.full_data:
            if item["form"] == form:
                item["morph_info"] = new_info
                break

    def get_full_data(self) -> List[Dict]:
        return self.full_data

    def get_lexeme_count(self) -> int:
        return self.lexeme_count

    def get_total_tokens(self) -> int:
        return self.total_tokens

    def get_wordforms_count(self) -> int:
        return len(self.full_data)

    def filter_data(self, query: str) -> List[Dict]:
        if not query:
            return self.full_data[:]
        query = query.lower().strip()
        return [
            item for item in self.full_data
            if query in item["form"].lower() or query in item["lemma"].lower()
        ]

    def get_help_text(self) -> str:
        try:
            with open('help.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "Файл help.txt не найден."
        except Exception as e:
            return f"Ошибка при чтении файла: {e}"

    def get_terminology_text(self) -> str:
        try:
            with open('terminology.txt', 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return "Файл terminology.txt не найден."
        except Exception as e:
            return f"Ошибка при чтении файла: {e}"