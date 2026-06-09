# Dockerfile do exporter de drift da Aula 6
# Build context = pasta da Aula 6 (06_monitoramento_continuo_e_data_slos_em_producao)
# para que possamos reaproveitar o pacote src/ existente.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Pacotes de sistema mínimos (wget é usado pelo healthcheck do compose)
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências
COPY monitoring_stack/app/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copia o pacote src/ original da aula (modelo, SLOMonitor, pipeline, etc.)
COPY src /app/src
COPY generate_dataset.py /app/generate_dataset.py

# Copia o app exporter
COPY monitoring_stack/app/drift_exporter.py /app/drift_exporter.py

EXPOSE 8000

CMD ["python", "/app/drift_exporter.py"]
