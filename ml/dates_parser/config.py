from datetime import datetime
from typing import Optional


class DatesParserConfig:
    """Конфигурационный класс для настройки параметров парсинга дат и обработки текста."""

    def __init__(
        self,
        prefer_dates_from: str = "future",
        prefer_day_of_month: str = "first",
        relative_base: Optional[datetime] = None,
        language: str = "ru",
        ordinal_threshold: int = 0,
    ):
        self.PREFER_DATES_FROM = prefer_dates_from
        self.PREFER_DAY_OF_MONTH = prefer_day_of_month
        self.RELATIVE_BASE = (
            relative_base if relative_base is not None else datetime.now()
        )
        self.LANGUAGE = language
        self.ORDINAL_THRESHOLD = ordinal_threshold
