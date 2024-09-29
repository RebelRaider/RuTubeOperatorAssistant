import random
from uuid import uuid4
from loguru import logger

from fastapi import Depends

from models.analytic import Analytic
from repositories.postgres.analytic import AnalyticRepository
from schemas.analytic import ChatReqeust


class AnalyticService:
    def __init__(self, repo: AnalyticRepository = Depends(AnalyticRepository)):
        self._repo = repo

    async def create(self, req: ChatReqeust) -> Analytic:
        logger.debug("Analytic - Service - create")
        return await self._repo.create(
            Analytic(
                id=uuid4(),
                question=req.question,
                answer=req.answer,
                satisfaction=req.satisfaction,
                new_functionality=bool(random.randint(0, 1)),
            )
        )
