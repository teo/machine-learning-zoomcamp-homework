FROM agrigorev/zoomcamp-model:2025
COPY question4.py pyproject.toml uv.lock .python-version pipeline_v1.bin /app/
WORKDIR /app
RUN pip install uv
RUN uv sync --locked
EXPOSE 8000
ENTRYPOINT [ "uv", "run", "question4.py", "--host", "0.0.0.0", "--port", "8000"]