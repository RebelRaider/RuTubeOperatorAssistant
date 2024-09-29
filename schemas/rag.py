import uuid

from pydantic import BaseModel


class CreateRagOpts(BaseModel):
    id: uuid.UUID

    class_1: str
    class_2: str
    text: str  # the metadata text
    text_embeddings: list[float]

    urls: list[str]  # the urls metadata

    image_path: str  # the path to S3 image
    image_embeddings: list[float]


class RagResponse(BaseModel):
    id: uuid.UUID

    class_1: str
    class_2: str
    text: str

    urls: list[str]

    image_path: str

    confidence: float
