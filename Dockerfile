FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api_service/ api_service/
COPY data_processing/ data_processing/
COPY evaluation/ evaluation/
COPY scripts/generate_questions.py scripts/generate_questions.py
COPY scripts/validate_mcq.py scripts/validate_mcq.py
COPY data/ data/
COPY processed_training_data/ processed_training_data/

ENV PYTHONUNBUFFERED=1
ENV DEFAULT_MODEL=fw/kimi-k2.5
ENV PRELOAD_GRADES=1,2,3,4,5,6,7,8,9,10,11,12

EXPOSE 8000

CMD ["uvicorn", "api_service.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
