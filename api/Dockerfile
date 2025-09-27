#FROM ubuntu:latest
#LABEL authors="evans"
#
#ENTRYPOINT ["top", "-b"]


FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY api/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]