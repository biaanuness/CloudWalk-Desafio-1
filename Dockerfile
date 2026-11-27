# Imagem opcional para reproduzir a análise sem instalar nada além do Docker.
# Construir:  docker build -t desafio-explicabilidade .
# Rodar:      docker run --rm -v "$PWD/saida:/app/saida" desafio-explicabilidade
#             (o padrão é o modo rápido; para o completo, acrescente: --saida saida)
FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.8.17
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON=python3.12

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project
COPY . .

ENTRYPOINT ["uv", "run", "--frozen", "python", "01-explicabilidade/analise_explicabilidade.py"]
CMD ["--rapido", "--saida", "saida"]
