from ml.embeddings.image import load_image_embeddings
from ml.embeddings.text import load_text_embeddings
from ml.llm.llm import LLama3Quantized
from ml.classificators.swear_classifier import load_swear_model
from ml.classificators.toxic_classifier import load_toxic_model
from ml.constants import (
    LLM_PATH,
    TOXIC_CLF_PATH,
    MODEL_TXT_EMB_NAME,
    MODEL_IMG_EMB_NAME,
)
from ml.llm.config import ModelKwargs
from ml.intention.intention import Intention

llm = LLama3Quantized()

kwargs = ModelKwargs(
    temperature=0.7,
    top_k=30,
    top_p=0.9,
    max_tokens=8192,
    repeat_penalty=1.1,
)

intent_model = Intention()

llm.load_model(kwargs, LLM_PATH)

swear_clf = load_swear_model()

toxic_clf = load_toxic_model(TOXIC_CLF_PATH)

text_tokenizer, text_embedder = load_text_embeddings(MODEL_TXT_EMB_NAME)

img_embedder, img_processor = load_image_embeddings(MODEL_IMG_EMB_NAME)
