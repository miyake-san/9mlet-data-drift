"""
Producer de transações FinBank (Aula 7 — arquitetura Kappa).

Gera transações de forma **contínua e aleatória**, alternando entre
períodos **sem drift** (regime ``baseline``) e períodos **com drift**
(regimes ``gradual`` e ``abrupt``), e publica cada lote em um
*Redis Stream* que funciona como *message broker*.

A alternância entre regimes é estocástica e com persistência: o producer
permanece em um regime por alguns lotes antes de, com probabilidade
``REGIME_SWITCH_PROB``, sortear um novo regime. Isso simula de forma
realista janelas estáveis intercaladas com surtos de drift gradual e
abrupto (Gama et al., 2014).

Variáveis de ambiente:
    REDIS_HOST              Host do Redis (default: redis)
    REDIS_PORT              Porta do Redis (default: 6379)
    STREAM_KEY              Nome do stream (default: finbank:transactions)
    STREAM_MAXLEN           Tamanho máximo aproximado do stream (default: 50000)
    PRODUCE_INTERVAL_SECONDS Intervalo entre lotes (default: 2)
    BATCH_SIZE              Transações por lote (default: 200)
    REGIME_SWITCH_PROB      Probabilidade de troca de regime por lote (default: 0.15)
    SEED                    Semente do gerador (default: 2025)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List

import numpy as np
import redis

from transaction_generator import (
    REGIMES,
    FinBankGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("finbank_producer")


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
STREAM_MAXLEN = env_int("STREAM_MAXLEN", 50_000)
PRODUCE_INTERVAL_SECONDS = env_float("PRODUCE_INTERVAL_SECONDS", 2.0)
BATCH_SIZE = env_int("BATCH_SIZE", 200)
REGIME_SWITCH_PROB = env_float("REGIME_SWITCH_PROB", 0.15)
SEED = env_int("SEED", 2025)

# Pesos de sorteio de regime quando ocorre uma troca. Mais tempo em baseline
# (sem drift) do que em drift, refletindo um cenário de produção saudável que
# ocasionalmente degrada.
REGIME_WEIGHTS = {"baseline": 0.50, "gradual": 0.30, "abrupt": 0.20}


def connect_redis() -> redis.Redis:
    """Conecta ao Redis com retry simples até o broker ficar disponível."""
    client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
    )
    for attempt in range(1, 31):
        try:
            client.ping()
            logger.info("Conectado ao Redis em %s:%d", REDIS_HOST, REDIS_PORT)
            return client
        except redis.exceptions.ConnectionError:
            logger.warning(
                "Redis indisponível (tentativa %d/30). Aguardando...", attempt
            )
            time.sleep(2)
    raise RuntimeError("Não foi possível conectar ao Redis.")


def pick_next_regime(rng: np.random.Generator) -> str:
    """Sorteia o próximo regime conforme os pesos configurados."""
    regimes = list(REGIME_WEIGHTS.keys())
    probs = np.array([REGIME_WEIGHTS[r] for r in regimes], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(regimes, p=probs))


def run() -> None:
    """Loop principal: gera lotes e publica no stream."""
    rng = np.random.default_rng(SEED)
    generator = FinBankGenerator(seed=SEED)
    client = connect_redis()

    current_regime = "baseline"
    drift_factor = 1.0
    batch_id = 0

    logger.info(
        "Producer iniciado | stream=%s batch=%d intervalo=%.1fs switch_prob=%.2f",
        STREAM_KEY,
        BATCH_SIZE,
        PRODUCE_INTERVAL_SECONDS,
        REGIME_SWITCH_PROB,
    )

    while True:
        # Decide se troca de regime neste lote.
        if rng.random() < REGIME_SWITCH_PROB:
            current_regime = pick_next_regime(rng)
            if current_regime == "gradual":
                # Drift gradual com intensidade aleatória.
                drift_factor = float(rng.uniform(0.3, 1.0))
            logger.info(
                "Troca de regime -> %s (drift_factor=%.2f)",
                current_regime,
                drift_factor if current_regime == "gradual" else 1.0,
            )

        records: List[dict] = generator.generate_batch(
            n=BATCH_SIZE,
            regime=current_regime,
            drift_factor=drift_factor,
        )

        fraud_count = sum(r["is_fraud"] for r in records)
        client.xadd(
            STREAM_KEY,
            {
                "payload": json.dumps(records),
                "regime": current_regime,
                "n": str(len(records)),
                "fraud_count": str(fraud_count),
                "batch_id": str(batch_id),
            },
            maxlen=STREAM_MAXLEN,
            approximate=True,
        )

        logger.info(
            "batch_id=%d regime=%s n=%d fraude=%d publicado",
            batch_id,
            current_regime,
            len(records),
            fraud_count,
        )

        batch_id += 1
        time.sleep(PRODUCE_INTERVAL_SECONDS)


def main() -> None:
    try:
        run()
    except KeyboardInterrupt:
        logger.info("Producer encerrado.")


if __name__ == "__main__":
    main()
