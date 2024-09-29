from pydantic import BaseModel


class ChatReqeust(BaseModel):
    question: str
    answer: str
    satisfaction: bool
