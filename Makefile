up:
	poetry run uvicorn app:app --reload --port 8000

.PHONY: migrate-rev
migrate-rev:
	poetry run alembic revision --autogenerate -m $(name)

.PHONY: migrate-up
migrate-up:
	poetry run alembic upgrade $(rev)

.PHONY: local
local:
	docker compose -f docker-compose.local.yml up --build

.PHONY: test
test:
	poetry run pytest

load_models:
	mkdir -p ml/models && \
	wget https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q8_0.gguf -O ml/models/model-q8_0.gguf && \
	mkdir -p ml/models/toxic-classifier && \
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/pytorch_model.bin -O ml/models/toxic-classifier/pytorch_model.bin && \
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/config.json -O ml/models/toxic-classifier/config.json && \
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/special_tokens_map.json -O ml/models/toxic-classifier/special_tokens_map.json && \
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/tokenizer.json -O ml/models/toxic-classifier/tokenizer.json && \
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/tokenizer_config.json -O ml/models/toxic-classifier/tokenizer_config.json && \
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/vocab.txt -O ml/models/toxic-classifier/vocab.txt

download-llm-model:
	mkdir -p ml/models
	wget https://huggingface.co/IlyaGusev/saiga_llama3_8b_gguf/resolve/main/model-q8_0.gguf -O ml/models/model-q8_0.gguf

download-toxic-model:
	mkdir -p ml/models/toxic-classifier
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/pytorch_model.bin -O ml/models/toxic-classifier/pytorch_model.bin
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/config.json -O ml/models/toxic-classifier/config.json
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/special_tokens_map.json -O ml/models/toxic-classifier/special_tokens_map.json
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/tokenizer.json -O ml/models/toxic-classifier/tokenizer.json
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/tokenizer_config.json -O ml/models/toxic-classifier/tokenizer_config.json
	wget https://huggingface.co/IlyaGusev/rubertconv_toxic_clf/resolve/main/vocab.txt -O ml/models/toxic-classifier/vocab.txt
