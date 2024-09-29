from __future__ import annotations
from typing import List, Union
import torch
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output: tuple, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Выполняет усреднение токенов входной последовательности на основе attention mask.

    Parameters:
    - model_output (tuple): Выход модели, включающий токенов эмбеддинги и другие данные.
    - attention_mask (torch.Tensor): Маска внимания для указания значимости токенов.

    Returns:
    - torch.Tensor: Усредненный эмбеддинг.

    Examples:
    >>> embeddings = model_output[0]
    >>> mask = torch.tensor([[1, 1, 1, 0, 0]])
    >>> pooled_embedding = mean_pooling((embeddings,), mask)
    """
    # Получаем эмбеддинги токенов из выхода модели
    token_embeddings = model_output[0]

    # Расширяем маску внимания для умножения с эмбеддингами
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    # Умножаем каждый токен на его маску и суммируем
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

    # Суммируем маски токенов и обрезаем значения, чтобы избежать деления на ноль
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Вычисляем усредненный эмбеддинг
    return sum_embeddings / sum_mask


def txt2embeddings(
    text: Union[str, List[str]], tokenizer, model, device: str = "cpu"
) -> torch.Tensor:
    """
    Преобразует текст в его векторное представление с использованием модели transformer.

    Parameters:
    - text (str): Текст для преобразования в векторное представление.
    - tokenizer: Токенизатор для предобработки текста.
    - model: Модель transformer для преобразования токенов в вектора.
    - device (str): Устройство для вычислений (cpu или cuda).

    Returns:
    - torch.Tensor: Векторное представление текста.

    Examples:
    >>> text = "Пример текста"
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    >>> model = AutoModel.from_pretrained("bert-base-multilingual-cased")
    >>> embeddings = txt2embeddings(text, tokenizer, model, device="cuda")
    """
    # Кодируем входной текст с помощью токенизатора
    if isinstance(text, str):
        text = [text]
    encoded_input = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128,
    )
    # Перемещаем закодированный ввод на указанное устройство
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Получаем выход модели для закодированного ввода
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Преобразуем выход модели в векторное представление текста
    return mean_pooling(model_output, encoded_input["attention_mask"])


def load_text_embeddings(
    model: str, device: str = "cpu", torch_dtype: str = "auto"
) -> tuple:
    """
    Загружает токенизатор и модель для указанной предобученной модели.

    Parameters:
    - model (str): Название предобученной модели, поддерживаемой библиотекой transformers.

    Returns:
    - tuple: Кортеж из токенизатора и модели.

    Examples:
    >>> tokenizer, model = load_text_embeddings("ai-forever/sbert_large_nlu_ru")
    """
    # Загружаем токенизатор для модели
    tokenizer = AutoTokenizer.from_pretrained(
        model, device_map=device, torch_dtype=torch_dtype
    )

    # Загружаем модель
    model = AutoModel.from_pretrained(model, device_map=device, torch_dtype=torch_dtype)

    return tokenizer, model
