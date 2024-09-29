import re
from datetime import datetime
from typing import Optional

from dateparser.search import search_dates
from ml.dates_parser.config import DatesParserConfig
from ml.dates_parser.preload import MORPH, STOPWORDS
from text_to_num import alpha2digit


class DatesParser:
    """Класс для обработки текста и извлечения дат с учетом настроек конфигурации."""

    def __init__(self, config: Optional[DatesParserConfig] = None):
        if config is None:
            config = DatesParserConfig()
        self.config = config
        self.morph = MORPH
        self.stopwords = STOPWORDS
        self.date_parser_settings = {
            "PREFER_DATES_FROM": config.PREFER_DATES_FROM,
            "PREFER_DAY_OF_MONTH": config.PREFER_DAY_OF_MONTH,
            "RELATIVE_BASE": config.RELATIVE_BASE,
        }

    def normalize_text(self, text):
        """Нормализует текст, приводя слова к их нормальной форме."""
        words = text.split()
        normalized_words = [self.morph.parse(word)[0].normal_form for word in words]
        return " ".join(normalized_words)

    def convert_numbers(self, text):
        """Конвертирует числовые значения в тексте из слов в цифры."""
        return alpha2digit(
            text, self.config.LANGUAGE, ordinal_threshold=self.config.ORDINAL_THRESHOLD
        )

    def remove_ordinal_suffixes(self, text):
        """Удаляет суффиксы у порядковых чисел."""
        return re.sub(r"\b(\d+)[а-яА-Я]*\b", r"\1", text)

    def remove_stopwords(self, text):
        """Удаляет стоп-слова из текста."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        return " ".join(filtered_words)

    def filter_out_non_dates(self, text):
        """Фильтрует числовые выражения, которые не могут быть датами."""
        text = re.sub(r"\d+\s*(рубл|евр|доллар)", "", text)
        return text

    def parse_dates(self, text):
        """Извлекает даты из текста."""
        return search_dates(
            text, languages=[self.config.LANGUAGE], settings=self.date_parser_settings
        )

    def format_dates(self, dates):
        """Форматирует даты в список объектов datetime."""
        if dates:
            formatted_dates = [
                self.normalize_datetime(date[1])
                for date in sorted(dates, key=lambda x: x[1])
            ]
            return formatted_dates
        return []

    def normalize_datetime(self, date):
        """Приводит объект datetime к стандартному формату."""
        return datetime(date.year, date.month, date.day, date.hour, date.minute)

    def extract_dates(self, text):
        """Процесс извлечения дат из текста с использованием всех методов."""
        normalized_text = self.normalize_text(text)
        text_with_numbers = self.convert_numbers(normalized_text)
        clean_text = self.remove_ordinal_suffixes(text_with_numbers)
        text_without_stopwords = self.remove_stopwords(clean_text)
        filtered_text = self.filter_out_non_dates(text_without_stopwords)
        dates = self.parse_dates(filtered_text)
        return self.format_dates(dates)


# Примеры использования
if __name__ == "__main__":
    texts = [
        "19 мая",
        "Забронируйте авиабилеты на рейс из Москвы в Санкт-Петербург с вылетом завтра и возвращением через неделю.",
        "Забронируйте номер в отеле в Москве со второго мая по десятое мая 15:40.",
        "двадцать первое мая 2024",
        "встреча назначена на 15 октября 2023 года в 15:10",
        "Скидка до 5000 рублей действует до конца мая",
        "Собрание состоится в 20:00 3-го числа",
    ]

    extractor = DatesParser()
    for text in texts:
        print(f"Текст: {text} -> Даты: {extractor.extract_dates(text)}")
