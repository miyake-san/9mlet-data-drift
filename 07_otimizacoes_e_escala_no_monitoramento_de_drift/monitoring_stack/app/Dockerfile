# Dockerfile compartilhado pelo producer e pelo drift-consumer da Aula 7.
# Build context = pasta da Aula 7 (07_otimizacoes_e_escala_no_monitoramento_de_drift)
# para reaproveitar o pacote src/ (DriftMonitor com PSI/KS/Wasserstein).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# wget é usado pelo healthcheck do drift-consumer no docker-compose.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências.
COPY monitoring_stack/app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Pacote src/ original da aula (DriftMonitor, IncrementalPSI, etc.).
COPY src /app/src

# Aplicações de streaming (gerador compartilhado, producer e consumer).
COPY monitoring_stack/app/transaction_generator.py /app/transaction_generator.py
COPY monitoring_stack/app/producer.py /app/producer.py
COPY monitoring_stack/app/drift_consumer.py /app/drift_consumer.py

EXPOSE 8000

# CMD default sobe o consumer; o producer sobrescreve 'command' no compose.
CMD ["python", "/app/drift_consumer.py"]
