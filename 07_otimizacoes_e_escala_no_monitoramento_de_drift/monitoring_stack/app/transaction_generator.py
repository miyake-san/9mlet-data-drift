"""
Gerador de transações FinBank em streaming (Aula 7).

Encapsula a lógica de distribuição por *regime* extraída de
``scripts/generate_dataset.py``, permitindo gerar transações de forma
contínua tanto **sem drift** (regime ``baseline``) quanto **com drift**
(regimes ``gradual`` e ``abrupt``).

Tanto o ``producer`` (que injeta dados no broker) quanto o
``drift_consumer`` (que constrói a distribuição de referência) compartilham
este módulo, garantindo que baseline e tráfego corrente sejam gerados
exatamente pela mesma família de distribuições — apenas com parâmetros
deslocados quando há drift (Gama et al., 2014; Sculley et al., 2015).

Referências:
    Gama, J. et al. (2014). A Survey on Concept Drift Adaptation. ACM.
    Sculley, D. et al. (2015). Hidden Technical Debt in ML Systems. NeurIPS.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

# Regimes de geração suportados. O rótulo é propagado nas mensagens do broker
# para servir de "ground truth" no dashboard (comparação drift real x detectado).
REGIMES: List[str] = ["baseline", "gradual", "abrupt"]

# Codificação numérica do regime para exposição no Prometheus.
REGIME_CODE: Dict[str, int] = {"baseline": 0, "gradual": 1, "abrupt": 2}

# Features numéricas monitoradas para drift (subconjunto de NUMERIC_FEATURES
# do DataPreprocessor da Aula 7).
MONITORED_FEATURES: List[str] = [
    "amount",
    "customer_age",
    "account_age_days",
    "transaction_count_30d",
    "avg_amount_30d",
    "hour_of_day",
]


def _regime_params(regime: str, drift_factor: float = 1.0) -> Dict[str, float]:
    """
    Retorna os parâmetros de distribuição para um dado regime.

    Os valores reproduzem os três cenários do caso FinBank descritos em
    ``generate_dataset.py``:

    - ``baseline``: distribuição estável (meses 1–3 do dataset original).
    - ``gradual`` : drift gradual interpolado por ``drift_factor`` em (0, 1].
    - ``abrupt``  : drift abrupto (meses 7–8 do dataset original).

    Args:
        regime: Um de ``baseline``, ``gradual`` ou ``abrupt``.
        drift_factor: Intensidade do drift gradual em (0, 1].

    Returns:
        Dicionário de parâmetros de distribuição.
    """
    if regime == "baseline":
        return {
            "amount_mean": 250.0,
            "amount_std": 120.0,
            "age_mean": 35.0,
            "age_std": 10.0,
            "acct_age_mean": 365.0,
            "txn_count_mean": 15.0,
            "avg_amount_mean": 240.0,
            "channel_probs": [0.40, 0.35, 0.20, 0.05],
            "intl_prob": 0.08,
            "fraud_rate": 0.03,
            "night_bias": 0.0,
        }

    if regime == "gradual":
        f = float(np.clip(drift_factor, 0.0, 1.0))
        return {
            "amount_mean": 250.0 + 100.0 * f,
            "amount_std": 120.0 + 40.0 * f,
            "age_mean": 35.0 - 3.0 * f,
            "age_std": 10.0 + 2.0 * f,
            "acct_age_mean": 365.0 - 80.0 * f,
            "txn_count_mean": 15.0 + 5.0 * f,
            "avg_amount_mean": 240.0 + 60.0 * f,
            "channel_probs": [
                0.40 + 0.10 * f,
                0.35 - 0.05 * f,
                0.20 - 0.05 * f,
                0.05,
            ],
            "intl_prob": 0.08 + 0.04 * f,
            "fraud_rate": 0.03 + 0.02 * f,
            "night_bias": 0.0,
        }

    # abrupt
    return {
        "amount_mean": 420.0,
        "amount_std": 200.0,
        "age_mean": 28.0,
        "age_std": 14.0,
        "acct_age_mean": 180.0,
        "txn_count_mean": 25.0,
        "avg_amount_mean": 380.0,
        "channel_probs": [0.55, 0.25, 0.10, 0.10],
        "intl_prob": 0.20,
        "fraud_rate": 0.08,
        "night_bias": 3.0,
    }


def _hour_distribution(night_bias: float) -> np.ndarray:
    """
    Distribuição de probabilidade de transações por hora do dia.

    Sob drift abrupto (``night_bias`` > 0) há mais transações em horários
    atípicos (madrugada/noite), conforme o caso FinBank.

    Args:
        night_bias: Peso extra adicionado às horas noturnas.

    Returns:
        Vetor de probabilidades de tamanho 24 (soma = 1).
    """
    base = np.array(
        [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 7,
         7, 7, 7, 7, 6, 5, 4, 4, 3, 2, 2, 1],
        dtype=float,
    )
    if night_bias > 0:
        base[0:6] += night_bias
        base[22:24] += night_bias
    return base / base.sum()


class FinBankGenerator:
    """
    Gera lotes de transações FinBank para um regime configurável.

    Diferentemente de ``generate_dataset.py`` (que materializa um CSV
    estático), esta classe produz lotes sob demanda em tempo de execução,
    o que viabiliza o cenário de *streaming* da arquitetura Kappa discutida
    na Aula 7 (broker + janelas deslizantes).

    Args:
        seed: Semente do gerador para reprodutibilidade.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        self._counter = 0

    def generate_batch(
        self,
        n: int,
        regime: str = "baseline",
        drift_factor: float = 1.0,
    ) -> List[Dict[str, object]]:
        """
        Gera um lote de ``n`` transações para o regime informado.

        Args:
            n: Quantidade de transações a gerar.
            regime: ``baseline`` (sem drift), ``gradual`` ou ``abrupt``.
            drift_factor: Intensidade do drift gradual em (0, 1].

        Returns:
            Lista de dicionários, uma transação por elemento.

        Raises:
            ValueError: Se ``regime`` for inválido.
        """
        if regime not in REGIMES:
            raise ValueError(f"Regime inválido: {regime}. Use um de {REGIMES}.")

        p = _regime_params(regime, drift_factor)
        rng = self._rng

        amounts = np.maximum(rng.normal(p["amount_mean"], p["amount_std"], n), 1.0)
        num_items = rng.poisson(3, n) + 1
        hours = rng.choice(24, n, p=_hour_distribution(p["night_bias"]))
        days_of_week = rng.integers(0, 7, n)
        ages = np.clip(rng.normal(p["age_mean"], p["age_std"], n), 18, 80)
        acct_ages = np.maximum(rng.poisson(p["acct_age_mean"], n), 1)
        txn_counts = np.maximum(rng.poisson(p["txn_count_mean"], n), 0)
        avg_amounts = np.maximum(rng.normal(p["avg_amount_mean"], 80, n), 10.0)
        channels = rng.choice(
            ["app", "web", "physical", "phone"], n, p=p["channel_probs"]
        )
        is_international = rng.binomial(1, p["intl_prob"], n)
        is_fraud = rng.binomial(1, p["fraud_rate"], n)

        now = datetime.utcnow()
        records: List[Dict[str, object]] = []
        for i in range(n):
            self._counter += 1
            ts = now + timedelta(
                hours=int(hours[i]), minutes=int(rng.integers(0, 60))
            )
            records.append(
                {
                    "transaction_id": f"TXN-{regime[:3].upper()}-{self._counter:08d}",
                    "timestamp": ts.isoformat(),
                    "amount": round(float(amounts[i]), 2),
                    "num_items": int(num_items[i]),
                    "hour_of_day": int(hours[i]),
                    "day_of_week": int(days_of_week[i]),
                    "customer_age": round(float(ages[i]), 1),
                    "account_age_days": int(acct_ages[i]),
                    "transaction_count_30d": int(txn_counts[i]),
                    "avg_amount_30d": round(float(avg_amounts[i]), 2),
                    "channel": str(channels[i]),
                    "is_international": int(is_international[i]),
                    "is_fraud": int(is_fraud[i]),
                    "regime": regime,
                }
            )
        return records

    def baseline_arrays(
        self, n: int, features: List[str] | None = None
    ) -> Dict[str, np.ndarray]:
        """
        Gera a distribuição de referência (baseline) como arrays por feature.

        Usado pelo ``drift_consumer`` para chamar ``DriftMonitor.fit`` com a
        distribuição estável de referência.

        Args:
            n: Tamanho da amostra de referência.
            features: Features a extrair. Default: ``MONITORED_FEATURES``.

        Returns:
            Dicionário ``{feature: np.ndarray}``.
        """
        feats = features or MONITORED_FEATURES
        batch = self.generate_batch(n, regime="baseline")
        return {
            feat: np.asarray([row[feat] for row in batch], dtype=float)
            for feat in feats
        }
