import re
import uuid

import pandas as pd
from loguru import logger

from ml.embeddings.text import txt2embeddings
from ml.lifespan import text_tokenizer, text_embedder
from repositories.clickhose.rag import ClickhouseRepository
from schemas.rag import CreateRagOpts
from services.rag import ClickhouseService

url_regex = re.compile(r"https?://(?:www\\.)?[ a-zA-Z0-9./]+")

db_dataset = pd.read_csv("ml/rag/data/db_dataset.csv")
dataset = pd.read_csv("ml/rag/data/dataset.csv")

service = ClickhouseService(ClickhouseRepository())


def process_text(text) -> (str, str):
    urls = url_regex.findall(text)

    clean_text = url_regex.sub("", text).strip()

    clean_text = re.sub(r"\(\)", "", clean_text).strip()

    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text, urls


def indexing():
    logger.debug("starting indexing db_dataset")
    for index, row in db_dataset.iterrows():
        text, urls = process_text(row["db_answer"])

        service.create(
            CreateRagOpts(
                id=uuid.uuid4(),
                class_1=row["label_1"],
                class_2=row["label_2"],
                text=text,
                text_embeddings=txt2embeddings(
                    row["user_question"], text_tokenizer, text_embedder
                )
                .squeeze()
                .tolist(),
                urls=urls,
                image_path="",
                image_embeddings=[],
            )
        )

        logger.debug(f"indexing {index} of {len(db_dataset) - 1}")

    logger.debug("starting indexing dataset")
    for index, row in dataset.iterrows():
        text, urls = process_text(row["db_answer"])

        service.create(
            CreateRagOpts(
                id=uuid.uuid4(),
                class_1=row["label_1"],
                class_2=row["label_2"],
                text=text,
                text_embeddings=txt2embeddings(
                    row["db_quesion"], text_tokenizer, text_embedder
                )
                .squeeze()
                .tolist(),
                urls=urls,
                image_path="",
                image_embeddings=[],
            )
        )

        logger.debug(f"indexing {index} of {len(dataset) - 1}")
