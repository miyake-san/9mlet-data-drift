"""
Drift Consumer / Exporter — Aula 7 (Otimizações e Escala no Monitoramento).

Consome o *Redis Stream* alimentado pelo ``producer``, mantém uma **janela
deslizante** das transações mais recentes e, a cada intervalo de coleta,
avalia drift comparando a janela corrente com uma **distribuição de
referência** (baseline estável). As métricas são publicadas no endpoint
``/metrics`` no formato Prometheus.

Esta é a camada de *speed/serving* da arquitetura **Kappa** discutida na
Aula 7: um único caminho de *streaming* unificado processa o fluxo em janelas
e produz sinais de drift de baixa latência (Carbone et al., 2015).

A detecção reutiliza o ``DriftMonitor`` da Aula 7 (``src/model.py``), que
combina PSI, teste K–S e distância de Wasserstein (Kullback & Leibler, 1951;
Kolmogorov, 1933; Smirnov, 1948).

Variáveis de ambiente:
    REDIS_HOST               Host do Redis (default: redis)
    REDIS_PORT               Porta do Redis (default: 6379)
    STREAM_KEY               Nome do stream (default: finbank:transactions)
    CONSUMER_GROUP           Grupo de consumo (default: drift-monitors)
    CONSUMER_NAME            Nome do consumidor (default: exporter-1)
    WINDOW_SIZE              Tamanho da janela deslizante (default: 2000)
    BASELINE_SIZE            Tamanho da amostra de referência (default: 5000)
    SCRAPE_INTERVAL_SECONDS  Intervalo entre avaliações (default: 5)
    EXPORTER_PORT            Porta do /metrics (default: 8000)
    PSI_THRESHOLD            Limiar de PSI para drift crítico (default: 0.25)
    PSI_WARNING              Limiar de PSI para warning (default: 0.10)
    KS_ALPHA                 Significância do teste K–S (default: 0.05)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from typing import Deque, Dict, List

import numpy as np
import redis
from prometheus_client import Counter, Gauge, Info, start_http_server

# Reutiliza o pacote src/ da Aula 7 (DriftMonitor com PSI/KS/Wasserstein).
sys.path.insert(0, "/app")

from src.model import DriftMonitor  # noqa: E402
from transaction_generator import (  # noqa: E402
    MONITORED_FEATURES,
    REGIME_CODE,
    FinBankGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("drift_consumer")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = env_int("REDIS_PORT", 6379)
STREAM_KEY = os.getenv("STREAM_KEY", "finbank:transactions")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "drift-monitors")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "exporter-1")
WINDOW_SIZE = env_int("WINDOW_SIZE", 2000)
BASELINE_SIZE = env_int("BASELINE_SIZE", 5000)
SCRAPE_INTERVAL_SECONDS = env_float("SCRAPE_INTERVAL_SECONDS", 5.0)
EXPORTER_PORT = env_int("EXPORTER_PORT", 8000)
PSI_THRESHOLD = env_float("PSI_THRESHOLD", 0.25)
PSI_WARNING = env_float("PSI_WARNING", 0.10)
KS_ALPHA = env_float("KS_ALPHA", 0.05)
SEED = env_int("SEED", 2025)

# Valor finito usado quando o PSI é NaN/inf. Isso ocorre, por construção do
# DriftMonitor, sob drift extremo: quando a janela corrente cai inteiramente
# fora dos bins do baseline, o histograma fica vazio e o PSI diverge. Nesse
# caso o drift é máximo, então o substituímos por um teto finito alto para
# manter as métricas Prometheus válidas (sem NaN) e o sinal de drift correto.
PSI_CAP = env_float("PSI_CAP", 10.0)


def _finite(value: float, fallback: float = PSI_CAP) -> float:
    """Substitui NaN/inf por um valor finito (drift máximo)."""
    return float(value) if np.isfinite(value) else float(fallback)

# ----------------------------------------------------------------------
# Métricas Prometheus
# ----------------------------------------------------------------------

BUILD_INFO = Info("finbank_drift_build", "Metadados do exporter de drift (Aula 7).")

PSI_GAUGE = Gauge(
    "finbank_drift_psi",
    "Population Stability Index por feature (>0.25 = drift severo).",
    ["feature"],
)
KS_PVALUE_GAUGE = Gauge(
    "finbank_drift_ks_pvalue",
    "P-valor do teste de Kolmogorov-Smirnov por feature (< alpha => drift).",
    ["feature"],
)
KS_STAT_GAUGE = Gauge(
    "finbank_drift_ks_statistic",
    "Estatística D do teste de Kolmogorov-Smirnov por feature.",
    ["feature"],
)
WASSERSTEIN_GAUGE = Gauge(
    "finbank_drift_wasserstein",
    "Distância de Wasserstein por feature (Earth Mover's Distance).",
    ["feature"],
)
DRIFT_DETECTED_GAUGE = Gauge(
    "finbank_drift_detected",
    "Drift detectado por feature (1 = sim, 0 = não).",
    ["feature"],
)

MEAN_PSI_GAUGE = Gauge("finbank_mean_psi", "PSI médio entre as features monitoradas.")
MAX_PSI_GAUGE = Gauge("finbank_max_psi", "PSI máximo entre as features monitoradas.")
DRIFT_FRACTION_GAUGE = Gauge(
    "finbank_drift_fraction", "Fração de features com drift detectado."
)
FEATURES_WITH_DRIFT_GAUGE = Gauge(
    "finbank_features_with_drift", "Número de features com drift detectado."
)

WINDOW_FILL_GAUGE = Gauge(
    "finbank_window_size", "Quantidade de transações na janela deslizante corrente."
)
FRAUD_RATE_GAUGE = Gauge(
    "finbank_window_fraud_rate", "Taxa de fraude observada na janela corrente."
)
REGIME_GAUGE = Gauge(
    "finbank_producer_regime",
    "Regime corrente do producer (0=baseline, 1=gradual, 2=abrupt) — ground truth.",
)
REGIME_LABEL_GAUGE = Gauge(
    "finbank_producer_regime_active",
    "Indicador booleano por rótulo de regime do producer.",
    ["regime"],
)

CONSUMED_TOTAL = Counter(
    "finbank_transactions_consumed_total",
    "Total de transações consumidas do stream desde o start.",
)
MESSAGES_TOTAL = Counter(
    "finbank_messages_consumed_total",
    "Total de mensagens (lotes) consumidas do stream desde o start.",
)
DRIFT_ALERTS_TOTAL = Counter(
    "finbank_drift_alerts_total",
    "Total de alertas de drift por feature desde o start.",
    ["feature"],
)

# Thresholds publicados como gauges fixos (overlay em painéis).
Gauge("finbank_psi_threshold", "Threshold de PSI para drift crítico.").set(PSI_THRESHOLD)
Gauge("finbank_psi_warning", "Threshold de PSI para warning.").set(PSI_WARNING)
Gauge("finbank_ks_alpha", "Nível de significância do teste K–S.").set(KS_ALPHA)


# ----------------------------------------------------------------------
# Broker / janela deslizante
# ----------------------------------------------------------------------


def connect_redis() -> redis.Redis:
    """Conecta ao Redis com retry até o broker ficar disponível."""
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    for attempt in range(1, 31):
        try:
            client.ping()
            logger.info("Conectado ao Redis em %s:%d", REDIS_HOST, REDIS_PORT)
            return client
        except redis.exceptions.ConnectionError:
            logger.warning("Redis indisponível (tentativa %d/30)...", attempt)
            time.sleep(2)
    raise RuntimeError("Não foi possível conectar ao Redis.")


def ensure_group(client: redis.Redis) -> None:
    """Cria o consumer group se ainda não existir (lendo do início)."""
    try:
        client.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("Consumer group '%s' criado.", CONSUMER_GROUP)
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            logger.info("Consumer group '%s' já existe.", CONSUMER_GROUP)
        else:
            raise


def build_baseline() -> DriftMonitor:
    """Constrói o baseline estável e ajusta o DriftMonitor."""
    generator = FinBankGenerator(seed=SEED)
    baseline = generator.baseline_arrays(BASELINE_SIZE, MONITORED_FEATURES)
    monitor = DriftMonitor(
        n_bins=10,
        psi_threshold=PSI_THRESHOLD,
        psi_warning=PSI_WARNING,
        ks_alpha=KS_ALPHA,
    )
    monitor.fit(baseline)
    logger.info(
        "Baseline ajustado (%d amostras, %d features).",
        BASELINE_SIZE,
        len(MONITORED_FEATURES),
    )
    return monitor


def window_to_arrays(window: Deque[dict]) -> Dict[str, np.ndarray]:
    """Converte a janela de transações em arrays por feature."""
    return {
        feat: np.asarray([row[feat] for row in window], dtype=float)
        for feat in MONITORED_FEATURES
    }


def publish_metrics(
    monitor: DriftMonitor,
    window: Deque[dict],
    last_regime: str,
) -> None:
    """Avalia drift na janela corrente e publica as métricas."""
    if len(window) < max(50, monitor.n_bins * 5):
        # Janela ainda pequena: evita estatísticas instáveis.
        WINDOW_FILL_GAUGE.set(len(window))
        return

    current = window_to_arrays(window)
    results = monitor.predict(current)

    drift_count = 0
    psi_values: List[float] = []
    for r in results:
        psi = _finite(r.psi_value)
        psi_values.append(psi)
        PSI_GAUGE.labels(feature=r.feature).set(psi)
        KS_PVALUE_GAUGE.labels(feature=r.feature).set(_finite(r.ks_pvalue, 0.0))
        KS_STAT_GAUGE.labels(feature=r.feature).set(_finite(r.ks_statistic, 0.0))
        WASSERSTEIN_GAUGE.labels(feature=r.feature).set(_finite(r.wasserstein_distance, 0.0))
        DRIFT_DETECTED_GAUGE.labels(feature=r.feature).set(1.0 if r.drift_detected else 0.0)
        if r.drift_detected:
            drift_count += 1
            DRIFT_ALERTS_TOTAL.labels(feature=r.feature).inc()

    mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
    max_psi = float(np.max(psi_values)) if psi_values else 0.0
    drift_fraction = drift_count / len(results) if results else 0.0

    MEAN_PSI_GAUGE.set(mean_psi)
    MAX_PSI_GAUGE.set(max_psi)
    DRIFT_FRACTION_GAUGE.set(drift_fraction)
    FEATURES_WITH_DRIFT_GAUGE.set(drift_count)

    WINDOW_FILL_GAUGE.set(len(window))
    fraud_rate = float(np.mean([row["is_fraud"] for row in window]))
    FRAUD_RATE_GAUGE.set(fraud_rate)

    REGIME_GAUGE.set(REGIME_CODE.get(last_regime, 0))
    for regime, code in REGIME_CODE.items():
        REGIME_LABEL_GAUGE.labels(regime=regime).set(1.0 if regime == last_regime else 0.0)

    logger.info(
        "janela=%d regime=%s mean_psi=%.4f max_psi=%.4f drift_features=%d/%d fraude=%.3f",
        len(window),
        last_regime,
        mean_psi,
        max_psi,
        drift_count,
        len(results),
        fraud_rate,
    )


def run() -> None:
    """Loop principal: consome o stream em janela deslizante e avalia drift."""
    client = connect_redis()
    ensure_group(client)
    monitor = build_baseline()

    BUILD_INFO.info(
        {
            "window_size": str(WINDOW_SIZE),
            "baseline_size": str(BASELINE_SIZE),
            "scrape_interval_s": str(SCRAPE_INTERVAL_SECONDS),
            "psi_threshold": str(PSI_THRESHOLD),
            "ks_alpha": str(KS_ALPHA),
            "architecture": "kappa-streaming",
        }
    )

    window: Deque[dict] = deque(maxlen=WINDOW_SIZE)
    last_regime = "baseline"
    last_eval = time.monotonic()

    logger.info(
        "Consumer iniciado | janela=%d baseline=%d coleta=%.1fs",
        WINDOW_SIZE,
        BASELINE_SIZE,
        SCRAPE_INTERVAL_SECONDS,
    )

    while True:
        # Leitura não-bloqueante curta para manter o loop responsivo.
        response = client.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=CONSUMER_NAME,
            streams={STREAM_KEY: ">"},
            count=50,
            block=1000,
        )

        if response:
            for _stream, messages in response:
                for msg_id, fields in messages:
                    records: List[dict] = json.loads(fields["payload"])
                    window.extend(records)
                    last_regime = fields.get("regime", last_regime)
                    CONSUMED_TOTAL.inc(len(records))
                    MESSAGES_TOTAL.inc()
                    client.xack(STREAM_KEY, CONSUMER_GROUP, msg_id)

        # Avalia drift no intervalo de coleta configurado.
        now = time.monotonic()
        if now - last_eval >= SCRAPE_INTERVAL_SECONDS:
            publish_metrics(monitor, window, last_regime)
            last_eval = now


def main() -> None:
    logger.info("Subindo HTTP server Prometheus em :%d/metrics", EXPORTER_PORT)
    start_http_server(EXPORTER_PORT)
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Consumer encerrado.")


if __name__ == "__main__":
    main()
