import os
from typing import Any, Optional

import joblib
import numpy as np
import torch
from loguru import logger
from ml.intention.config import ClassifierConfig
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class Intention:
    """
    Класс для обработки запросов на предсказание намерений с использованием модели DistilBERT.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        if not config:
            config = ClassifierConfig()
        self.config = config
        self.max_length = config.MAX_LENGTH
        self.method = config.METHOD
        self.threshold = config.THRESHOLD
        self.device = torch.device(config.DEVICE)
        if os.listdir(config.SAVE_PATH):
            self.model = (
                DistilBertForSequenceClassification.from_pretrained(
                    config.SAVE_PATH, output_hidden_states=config.OUTPUT_HIDDEN_STATES
                )
                .to(self.device)
                .eval()
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(config.SAVE_PATH)
            self.label_encoder = joblib.load(
                f"{config.SAVE_PATH}/{config.LABEL_ENCODER_NAME}"
            )
        else:
            logger.warning("Модель не найдена. Для начала обучите модель.")

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Токенизация текстов."""
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False,
        )

    def get_model_outputs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Получение выходных данных модели."""
        inputs = self._tokenize(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()
        logits = outputs.logits.cpu().numpy()
        return embeddings, logits

    def _compute_label_probabilities(
        self, probabilities: np.ndarray
    ) -> list[dict[str, Any]]:
        """Вычисление вероятностей меток."""
        return sorted(
            [
                {
                    "name": self.label_encoder.inverse_transform([i])[0],
                    "confidence": float(prob),
                }
                for i, prob in enumerate(probabilities)
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )

    def _find_similar_words(
        self,
        sentences: list[str],
        word_list: list[str],
        word_embeddings: np.ndarray,
    ) -> dict[str, dict[str, list[str]]]:
        """Нахождение похожих слов."""
        if self.method == "sentence":
            return self._find_similar_words_by_sentence(
                sentences, word_list, word_embeddings
            )
        elif self.method == "words":
            return self._find_similar_words_by_words(
                sentences, word_list, word_embeddings
            )
        else:
            raise ValueError("Method must be 'sentence' or 'words'")

    def _find_similar_words_by_sentence(
        self, sentences: list[str], word_list: list[str], word_embeddings: np.ndarray
    ):
        """Нахождение похожих слов по предложениям."""
        sentence_embeddings, _ = self.get_model_outputs(sentences)
        similarities = cosine_similarity(word_embeddings, sentence_embeddings)
        return {
            sentences[i]: [
                word_list[j]
                for j in range(len(similarities))
                if similarities[j, i] > self.threshold
            ]
            for i in range(len(sentences))
        }

    def _find_similar_words_by_words(
        self,
        sentences: list[str],
        word_list: list[str],
        word_embeddings: np.ndarray,
    ) -> dict[str, dict[str, list[str]]]:
        """Нахождение похожих слов по словам."""
        results = {}
        for sentence in sentences:
            words = sentence.split()
            word_embeddings_sentence, _ = self.get_model_outputs(words)
            similarities = cosine_similarity(word_embeddings_sentence, word_embeddings)
            word_matches: dict[str, list[str]] = {word: [] for word in word_list}
            for i, word in enumerate(words):
                for j, sim in enumerate(similarities[i]):
                    if sim > self.threshold:
                        word_matches[word_list[j]].append(word)
            results[sentence] = word_matches
        return results

    def predict_intent(
        self,
        text: str,
        word_list: Optional[list[str]] = None,
        word_embeddings: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """
        Предсказание намерения для одного текста.
        """
        if not text:
            raise ValueError("Input text cannot be empty")
        _, logits = self.get_model_outputs([text])
        probabilities = (
            torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy().flatten()
        )

        label_probabilities = self._compute_label_probabilities(probabilities)
        top_prediction = label_probabilities[0]

        if word_embeddings is not None and word_list is not None:
            entities_list = self._find_similar_words([text], word_list, word_embeddings)
        else:
            entities_list = {}

        return {
            "text": text,
            "intent": top_prediction,
            "intent_ranking": label_probabilities[:10],
            "entities": entities_list.get(text, []) if entities_list else {},
        }

    def bulk_predict_intent(
        self,
        texts: list[str],
        word_list: Optional[list[str]] = None,
        word_embeddings: Optional[np.ndarray] = None,
    ) -> list[dict[str, Any]]:
        """
        Предсказание намерений для нескольких текстов.
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")

        _, logits = self.get_model_outputs(texts)
        probabilities = torch.nn.functional.softmax(
            torch.tensor(logits), dim=-1
        ).numpy()

        label_probabilities_list = [
            self._compute_label_probabilities(prob) for prob in probabilities
        ]
        top_predictions = [
            label_probabilities[0] for label_probabilities in label_probabilities_list
        ]
        if word_embeddings is not None and word_list is not None:
            entities_list = self._find_similar_words(texts, word_list, word_embeddings)
        else:
            entities_list = {}

        return [
            {
                "text": text,
                "intent": top_predictions[i],
                "intent_ranking": label_probabilities_list[i][:10],
                "entities": entities_list.get(text, []) if entities_list else {},
            }
            for i, text in enumerate(texts)
        ]


# Пример использования
if __name__ == "__main__":
    import time

    model_path = "./saved_model"
    tokenizer_path = "./saved_model"
    label_encoder_path = "./saved_model/label_encoder.pkl"
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"

    handler = Intention()

    unique_words = []

    if unique_words:
        word_embeddings = handler.get_model_outputs(unique_words)[0]
    else:
        word_embeddings = None

    example_texts = [
        "Где можно поесть в Казани?",
        "чё по турам в москве?",
        "как найти экскурсии в Сочи?",
        "Карта Москвы",
        "Подскажи ближайшие достопримечательности",
        "где билеты на поезд в питер?",
        "что интересного рядом?",
        "какие концерты в екате на выходных?",
        "куда можно пойти в питере?",
        "расскажи про коломенский кремль",
        "где поесть в москве недорого?",
        "помощь по использованию сайта",
        "какие фесты в сочи?",
        "спланируй поездку в сочи",
        "чё посмотреть в казани?",
        "как забронировать отель в екатеринбурге?",
        "какие музеи есть в питере?",
        "что ты умеешь?",
        "где мои избранные места?",
        "сколько стоит билет на аэроэкспресс?",
    ]

    start_time = time.time()
    results = handler.bulk_predict_intent(example_texts, unique_words, word_embeddings)
    end_time = time.time()
    print(f"Time elapsed for 40 in bulk: {end_time - start_time}")  # noqa: T201
    for result in results:
        print(f"Текст: {result['text']}")  # noqa: T201
        print(f"Intent: {result['intent']}")  # noqa: T201
        print(f"Топ 10 intents: {result['intent_ranking']}")  # noqa: T201
        print(f"Entities: {result['entities']}")  # noqa: T201

    start_time = time.time()
    result = handler.predict_intent(example_texts[0], unique_words, word_embeddings)
    end_time = time.time()

    print(f"Time elapsed for 1 in sample: {end_time - start_time}")  # noqa: T201
    print(f"Текст: {result['text']}")  # noqa: T201
    print(f"Intent: {result['intent']}")  # noqa: T201
    print(f"Топ 10 intents: {result['intent_ranking']}")  # noqa: T201
    print(f"Entities: {result['entities']}")  # noqa: T201
