import torch.nn.functional
from asyncpg import InternalServerError
from fastapi.params import Depends
from loguru import logger
from transformers.models.auto.image_processing_auto import image_processors

from ml.classificators.swear_classifier import has_swear
from ml.classificators.toxic_classifier import is_toxic
from ml.constants import REGENERATE_TRIES_COUNT, SYSTEM_PROMPT
from ml.embeddings.image import image2embeddings
from ml.embeddings.text import txt2embeddings
from ml.lifespan import (
    intent_model,
    llm,
    toxic_clf,
    swear_clf,
    text_tokenizer,
    text_embedder,
    img_embedder,
)
from schemas.ml import Answer
from schemas.rag import RagResponse
from services.rag import ClickhouseService


class MlService:
    def __init__(self, clickhouse: ClickhouseService = Depends(ClickhouseService)):
        self._clickhouse = clickhouse

        self.standard_answer = "В базе знаний нет ответа на данный вопрос. Поприветсвуйтся, извинись, и скажи честно \"я не знаю\"."

    def _generate_answer(
        self, question: str, intention: str, rag_answer: str, class_1: str, class_2: str
    ) -> Answer:
        """
        Генерирует ответ на основе входного вопроса, намерения и базового ответа, используя модель языка (LLM).
        Повторяет попытки генерации ответа, пока не будет получен не токсичный результат или не будет исчерпано
        максимальное количество попыток.

        Parameters
        ----------
        question : str
            Вопрос, на который необходимо сгенерировать ответ.
        intention : str
            Намерение или цель, связанная с вопросом (может быть использовано для определения контекста генерации).
        rag_answer : str
            Начальный (базовый) ответ, полученный из Retrieval-Augmented Generation (RAG) для сравнения с финальным ответом.
        class_1 : str
            Категория или класс, к которому относится вопрос (может использоваться для классификации ответа).
        class_2 : str
            Дополнительная категория или класс, к которому относится вопрос (может использоваться для уточнения классификации).

        Returns
        -------
        Answer
            Экземпляр класса `Answer`, содержащий сгенерированный ответ и соответствующие классы.

        Raises
        ------
        InternalServerError
            Если не удалось сгенерировать валидный ответ после максимального количества попыток.

        Notes
        -----
        Процесс генерации ответа включает следующие шаги:
        1. Попытка генерации ответа с использованием LLM (модель языка) до достижения максимального количества попыток.
        2. Проверка ответа на токсичность и наличие ненормативной лексики.
        3. Сравнение сгенерированного ответа с базовым ответом (rag_answer) по степени схожести.
        4. Возврат ответа, если он прошел все проверки.

        В случае неудачи (токсичный ответ или низкая схожесть) функция повторяет попытку до исчерпания лимита.

        Пример использования:
        --------------------
        answer = _generate_answer("Какова погода?", "информация о погоде", "Сегодня солнечно", "информация", "погода")
        """
        logger.debug("generating answer")
        for index in range(REGENERATE_TRIES_COUNT):
            logger.debug(
                f"running the llm generation, {index} try of the {REGENERATE_TRIES_COUNT}"
            )

            llm.add_message("system", SYSTEM_PROMPT.format(rag_answer))

            if question != "":
                llm.add_message("user", question)

            answer = llm.inference()

            if not is_toxic(toxic_clf, answer) and not has_swear(swear_clf, answer):
                similarity = self._check_output_and_answer_similarity(
                    txt2embeddings(
                        [rag_answer], text_tokenizer, text_embedder
                    ).squeeze(),
                    txt2embeddings([answer], text_tokenizer, text_embedder).squeeze(),
                )

                if similarity < 0.3:
                    logger.error(
                        f"the similarity between this rag_answer {rag_answer} and this answer {answer} is {similarity}"
                    )
                    llm.clean_chat()
                    continue

                llm.clean_chat()
                return Answer(
                    answer=answer,
                    class_1=class_1,
                    class_2=class_2,
                )

            logger.debug(f"toxic answer: {answer}")

        llm.clean_chat()
        raise InternalServerError(
            f"error creating answer, question = {question}, intent = {intention}, rag_answer = {rag_answer}"
        )

    def answer(self, question: str) -> Answer:
        logger.debug("ML - Service - answer")

        intention = self._get_intention(question)

        rag_answer = self._get_rag_answer_by_text(question, intention)

        logger.debug(f"question: {question}, confidence: {rag_answer[0].confidence}")

        if rag_answer[0].confidence < 0.5:
            return self._generate_answer(
                "",
                intention,
                self.standard_answer,
                rag_answer[0].class_1,
                rag_answer[0].class_2,
            )

        try:
            ans = self._generate_answer(
                question,
                intention,
                "\n".join([obj.text for obj in rag_answer]),
                rag_answer[0].class_1,
                rag_answer[0].class_2,
            )

            ans.answer += '\n\n' + "Ответ из базы знаний: " + rag_answer[0].text + '\n\n' + ("Cсылки:" + "\n\n".join(rag_answer[0].urls) if rag_answer[0].urls  else "")

            return ans
        except InternalServerError as e:
            logger.error(f"Error generating answer: {e}")
            return self._generate_answer(
                "",
                intention,
                self.standard_answer,
                rag_answer[0].class_1,
                rag_answer[0].class_2,
            )

    def answer_with_photo(self, question: str, image: bytes | None) -> Answer:
        logger.debug("ML - Service - answer_with_photo")

        intention = self._get_intention(question)

        rag_text_answer = self._get_rag_answer_by_text(question, intention)

        rag_image_answer = self._get_rag_answer_by_image(image, intention)

        return self._generate_answer(
            question,
            intention,
            "\n".join([obj.text for obj in rag_image_answer + rag_text_answer]),
            rag_text_answer[0].class_1,
            rag_text_answer[0].class_2,
        )

    def _check_output_and_answer_similarity(
        self, answer_embedding: torch.Tensor, output_embedding: torch.Tensor
    ) -> float:
        """
        Вычисляет косинусное сходство между векторными представлениями (эмбеддингами) двух текстов.
        Используется для оценки степени схожести между сгенерированным ответом и исходным ответом (RAG).

        Parameters
        ----------
        answer_embedding : torch.Tensor
            Тензорное представление (эмбеддинг) исходного ответа, полученное с помощью модели эмбеддинга.
        output_embedding : torch.Tensor
            Тензорное представление (эмбеддинг) сгенерированного ответа, полученное с помощью модели эмбеддинга.

        Returns
        -------
        float
            Значение косинусного сходства между двумя эмбеддингами, где 1.0 указывает на полную идентичность,
            а -1.0 на противоположность. Обычно значения находятся в диапазоне от 0 до 1, где 0 означает отсутствие
            сходства, а 1 – полное сходство.

        Notes
        -----
        Косинусное сходство - это мера угла между двумя векторами в пространстве эмбеддингов. Чем ближе значение к 1.0,
        тем более схожими являются ответы. Значение сходства преобразуется из тензорного формата в стандартный
        тип данных float для дальнейшего использования в коде.

        Пример использования:
        --------------------
        similarity = _check_output_and_answer_similarity(answer_embedding, output_embedding)
        """
        similarity = torch.nn.functional.cosine_similarity(
            answer_embedding,
            output_embedding,
            dim=0,
        )

        return float(similarity)

    def _get_intention(self, question: str):
        logger.debug("getting intentions")
        intent = intent_model.predict_intent(question)

        intention = intent.get("intent").get("name")

        return intention

    def _get_rag_answer_by_image(
        self, image: bytes, intention: str
    ) -> list[RagResponse]:
        logger.debug("getting image embeddings")
        embedding = image2embeddings(image, image_processors, img_embedder)

        rag_answer = self._clickhouse.get_by_image(
            embedding.squeeze().tolist(), intention, 2
        )

        return rag_answer

    def _get_rag_answer_by_text(self, text: str, intention: str) -> list[RagResponse]:
        logger.debug("getting text embeddings")
        embedding = txt2embeddings([text], text_tokenizer, text_embedder)

        # TODO подумать, что делать если нет класса в RAG
        logger.debug("getting from rag")
        rag_answer = self._clickhouse.get_by_text(
            embedding.squeeze().tolist(), intention, 2
        )

        return rag_answer
