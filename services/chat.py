from fastapi import Depends
from loguru import logger

from schemas.chat import ChatResponse
from services.analytic import AnalyticService
from services.ml import MlService


class ChatService:
    def __init__(
        self,
        ml: MlService = Depends(MlService),
        analytic: AnalyticService = Depends(AnalyticService),
    ):
        self._ml = ml
        self._analytic = analytic

    async def answer(self, question: str) -> ChatResponse:
        logger.debug("Chat - Service - answer")
        resp = self._ml.answer(question)

        return ChatResponse(
            answer=resp.answer, class_1=resp.class_1, class_2=resp.class_2
        )

    async def answer_with_image(
        self, question: str, image: bytes | None
    ) -> ChatResponse:
        logger.debug("Chat - Service - answer_with_image")
        resp = self._ml.answer_with_photo(question, image)

        return ChatResponse(
            answer=resp.answer, class_1=resp.class_1, class_2=resp.class_2
        )
