FROM nvidia/cuda:12.5.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update -y && \
    apt upgrade -y && \
    apt install -y python3.12-full python3.12-distutils python3-setuptools python3-pip  libcairo2-dev git software-properties-common ffmpeg

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

RUN python3 -m ensurepip --upgrade

WORKDIR /app

COPY pyproject.toml poetry.lock /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN CUDACXX=/usr/local/cuda-12/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 poetry run pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
RUN poetry run pip install numpy==1.26.4

COPY . /app/

CMD ["poetry", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]