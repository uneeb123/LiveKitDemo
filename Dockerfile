FROM python:3.13-slim
RUN pip install --no-cache-dir uv==0.6.13
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev
COPY src ./src
RUN useradd --uid 10001 --create-home max && chown -R max:max /app
USER max
RUN .venv/bin/python src/agent.py download-files
CMD [".venv/bin/python", "src/agent.py", "start"]
