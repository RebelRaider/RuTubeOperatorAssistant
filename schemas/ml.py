from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    class_1: str
    class_2: str


class AnswerByImage(BaseModel):
    answer: str
    class_1: list[str]
    class_2: list[str]
