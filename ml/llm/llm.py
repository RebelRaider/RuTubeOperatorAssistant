from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from llama_cpp import Llama
from rich.console import Console
from rich.table import Table

from ml.llm.config import ModelKwargs


class BaseLlama3Model(ABC):
    @abstractmethod
    def load_model(self, kwargs: ModelKwargs, model_path: str) -> None: ...

    @abstractmethod
    def inference(self, kwargs: Optional[ModelKwargs] = None) -> str: ...

    @abstractmethod
    def add_message(self, role: str, content: str) -> None: ...

    @abstractmethod
    def clean_chat(self) -> None: ...

    @abstractmethod
    def print_chat(self) -> None: ...

    @abstractmethod
    def get_chat(self) -> List[Dict[str, str]]: ...


class LLama3Quantized(BaseLlama3Model):
    def __init__(self) -> None:
        super().__init__()
        self._llm: Optional[Llama] = None
        self._chat: List[Dict[str, str]] = []
        self._kwargs: ModelKwargs = ModelKwargs()
        self._console = Console()

    def load_model(self, kwargs: ModelKwargs, model_path: str) -> None:
        if not model_path:
            raise ValueError("Путь к модели не может быть пустым.")
        self._llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            n_ctx=kwargs.max_tokens,
            n_parts=1,
        )
        self._kwargs = kwargs

    def _check_model_loaded(self) -> None:
        if not self._llm:
            raise RuntimeError(
                "Модель не загружена. Пожалуйста, загрузите модель перед выполнением вывода."
            )

    def inference(self, kwargs: Optional[ModelKwargs] = None) -> str:
        self._check_model_loaded()
        if not kwargs:
            kwargs = self._kwargs
        if not self._chat:
            raise RuntimeError(
                "Чат пуст. Пожалуйста, добавьте сообщения в чат перед выполнением вывода."
            )
        return self._llm.create_chat_completion(
            self._chat,
            temperature=kwargs.temperature,
            top_k=kwargs.top_k,
            top_p=kwargs.top_p,
            repeat_penalty=kwargs.repeat_penalty,
            stream=False,
        )["choices"][0]["message"]["content"]

    def add_message(self, role: str, content: str) -> None:
        if not role or not content:
            raise ValueError("Роль и содержание не могут быть пустыми.")
        self._chat.append({"role": role, "content": content})

    def clean_chat(self) -> None:
        self._chat = []

    def print_chat(self) -> None:
        if not self._chat:
            self._console.print("Чат пуст.", style="bold red")
            return
        table = Table(title="Чат", show_header=True, header_style="bold magenta")
        table.add_column("Роль", style="dim", width=12)
        table.add_column("Сообщение")
        for message in self._chat:
            for role, content in message.items():
                table.add_row(role.capitalize(), content)
        self._console.print(table)

    def get_chat(self) -> List[Dict[str, str]]:
        return self._chat


if __name__ == "__main__":
    from ml.constants import LLM_PATH

    llm = LLama3Quantized()
    kwargs = ModelKwargs(
        temperature=0.7,
        top_k=30,
        top_p=0.9,
        max_tokens=8192,
        repeat_penalty=1.1,
    )
    try:
        llm.load_model(kwargs, LLM_PATH)
        llm.add_message(
            "system", "Ты Сайга и твоя задача помогать людям в решении вопросов."
        )
        llm.add_message("user", "Привет, в чём смысл вселенной?")
        answer = llm.inference()
        llm.add_message("assistant", answer)
        llm.add_message("user", "Спасибо за такой точный ответ!")
        answer2 = llm.inference()
        llm.print_chat()
        llm.clean_chat()
    except (ValueError, RuntimeError) as e:
        print(f"Ошибка: {e}")
