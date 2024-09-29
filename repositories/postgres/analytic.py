import uuid

from fastapi import Depends
from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from configs.Database import get_db_connection
from errors.errors import ErrEntityNotFound
from models.analytic import Analytic


class AnalyticRepository:
    def __init__(self, db: AsyncSession = Depends(get_db_connection)):
        self._db = db

    async def create(self, analytic: Analytic) -> Analytic:
        logger.debug("Analytic - Repository - create")

        self._db.add(analytic)
        await self._db.commit()
        await self._db.refresh(analytic)
        return analytic

    async def get(self, uuid: uuid.UUID) -> Analytic:
        logger.debug("Analytic - Repository - get")

        query = select(Analytic).where(Analytic.id == uuid)

        result = await self._db.execute(query)

        try:
            analytic = result.scalar_one()
        except NoResultFound:
            raise ErrEntityNotFound("entity not found")

        return analytic

    async def update(self, analytic: Analytic) -> Analytic:
        logger.debug("Analytic - Repository - update")

        await self._db.commit()
        await self._db.refresh(analytic)
        return analytic
