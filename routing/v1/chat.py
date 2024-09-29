from fastapi import APIRouter, UploadFile, File, Form
from fastapi.params import Depends

from ml.rag.indexing import indexing
from schemas.chat import ChatResponse, ChatRequesst
from services.chat import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post(
    "/answer",
    summary="ответ на вопрос",
    response_model=ChatResponse,
)
async def answer(
    req: ChatRequesst,
    chat_service: ChatService = Depends(ChatService),
):
    return await chat_service.answer(req.question)


@router.post(
    "/answer_with_photo",
    summary="ответ на вопрос",
    response_model=ChatResponse,
)
async def answer_with_photo(
    question: str = Form(),
    image: UploadFile | None = File(...),
    chat_service: ChatService = Depends(ChatService),
):
    image_bytes = await image.read()

    return await chat_service.answer_with_image(question, image_bytes)


@router.post(
    "/indexing",
    summary="индексация данных",
)
def create_index():
    indexing()
