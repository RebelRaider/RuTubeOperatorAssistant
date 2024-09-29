import uuid

from sqlalchemy.orm import Mapped, mapped_column

from models.BaseModel import EntityMeta


class Analytic(EntityMeta):
    __tablename__ = "analytic"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    question: Mapped[str]
    answer: Mapped[str]
    satisfaction: Mapped[bool | None]  # удовлетворен ли пользователь ответом или нет
    new_functionality: Mapped[
        bool
    ]  # является ли это запросом какой-то новой функциональности
