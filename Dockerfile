FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY autopilot ./autopilot
COPY config ./config
COPY data ./data
COPY dashboard ./dashboard
COPY scripts ./scripts
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m autopilot.cli train
EXPOSE 8000
CMD ["uvicorn","autopilot.api:app","--host","0.0.0.0","--port","8000"]
