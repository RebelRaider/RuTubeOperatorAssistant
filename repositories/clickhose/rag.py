from uuid import uuid4

from loguru import logger

from schemas.rag import CreateRagOpts, RagResponse
from configs.Clickhouse import client


class ClickhouseRepository:
    def __init__(self):
        self._client = client

    def create(self, opts: CreateRagOpts):
        logger.debug("Rag - Repository - create")
        query = """
            INSERT INTO `rag` (id, class_1, class_2, text, text_embeddings, urls, image_path, image_embeddings)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)   
        """

        self._client.command(
            query,
            (
                uuid4(),
                opts.class_1,
                opts.class_2,
                opts.text,
                opts.text_embeddings,
                opts.urls,
                opts.image_path,
                opts.image_embeddings,
            ),
        )

    def get_by_text_embeddings(
        self, embeddings: list[float], class_1: str, top_k: int
    ) -> list[RagResponse]:
        logger.debug("Rag - Repository - get_by_text_embeddings")
        query = f"""
            WITH {embeddings} as query_vector
            SELECT id, class_1, class_2, text, urls, image_path,
            arraySum(x -> x * x, text_embeddings) * arraySum(x -> x * x, query_vector) != 0
            ? arraySum((x, y) -> x * y, text_embeddings, query_vector) / sqrt(arraySum(x -> x * x, text_embeddings) * arraySum(x -> x * x, query_vector))
            : 0 AS cosine_distance
            FROM rag
            WHERE length(query_vector) == length(text_embeddings) and class_1='{class_1}'
            ORDER BY cosine_distance DESC 
            LIMIT {top_k}
        """

        result = self._client.query(
            query, settings={"max_query_size": "10000000000000"}
        )

        rows = result.result_rows

        answers: list[RagResponse] = []

        for row in rows:
            answers.append(
                RagResponse(
                    id=row[0],
                    class_1=row[1],
                    class_2=row[2],
                    text=row[3],
                    urls=row[4],
                    image_path=row[5],
                    confidence=row[6],
                )
            )

        return answers

    def get_by_image_embeddings(
        self, embeddings: list[float], class_1: str, top_k: int
    ) -> list[RagResponse]:
        logger.debug("Rag - Repository - get_by_image_embeddings")
        query = f"""
            WITH {embeddings} as query_vector
            SELECT id, class_1, class_2, text, urls, image_path 
            arraySum(x -> x * x, image_embeddings) * arraySum(x -> x * x, query_vector) != 0
            ? arraySum((x, y) -> x * y, image_embeddings, query_vector) / sqrt(arraySum(x -> x * x, image_embeddings) * arraySum(x -> x * x, query_vector))
            : 0 AS cosine_distance
            FROM rag
            WHERE length(query_vector) == length(image_embeddings) and class_1='{class_1}'
            ORDER BY cosine_distance DESC 
            LIMIT {top_k}
        """

        result = self._client.query(
            query, settings={"max_query_size": "10000000000000"}
        )

        rows = result.result_rows

        answers: list[RagResponse] = []

        for row in rows:
            answers.append(
                RagResponse(
                    id=row[0],
                    class_1=row[1],
                    class_2=row[2],
                    text=row[3],
                    urls=row[4],
                    image_path=row[5],
                )
            )

        return answers
