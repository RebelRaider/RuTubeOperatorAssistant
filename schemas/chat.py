from pydantic import BaseModel


class ChatRequesst(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    class_1: str
    class_2: str


class ChatResponseWithImage(BaseModel):
    answer: str
    class_1: list[str]
    class_2: list[str]


class TestChatRequest(BaseModel):
    question: str


class TestChatResponse(BaseModel):
    answer: str
    class_1: str
    class_2: str
