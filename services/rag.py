from fastapi import Depends
from loguru import logger

from repositories.clickhose.rag import ClickhouseRepository
from schemas.rag import CreateRagOpts, RagResponse


class ClickhouseService:
    def __init__(self, repo: ClickhouseRepository = Depends(ClickhouseRepository)):
        self._repo = repo

    def create(self, opts: CreateRagOpts):
        logger.debug("Rag - Service - create")
        self._repo.create(opts)

    def get_by_text(
        self, embeddings: list[float], class_1: str, top_k: int
    ) -> list[RagResponse]:
        logger.debug("Rag - Service - get_by_text")
        return self._repo.get_by_text_embeddings(embeddings, class_1, top_k)

    def get_by_image(self, embeddings: list[float], class_1: str, top_k: int):
        logger.debug("Rag - Service - get_by_image")
        return self._repo.get_by_image_embeddings(embeddings, class_1, top_k)
