"""
Script para geração do dataset sintético de transações de pagamento.

Simula o cenário de detecção de fraude em plataforma de pagamentos global,
conforme descrito no Documento 04 da Aula 8 ('Integração e Revisão').

Gera dois períodos:
    - Referência (baseline): distribuição estável para treinamento.
    - Produção (com drift): covariate drift, prior drift e concept drift
      simulados para demonstrar ferramentas de monitoramento (Evidently,
      NannyML, Great Expectations).

Referências:
    Rabanser, S. et al. (NeurIPS 2019). Failing loudly.
    Müller, R. et al. (2024). Open-source drift detection tools in action.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def generate_reference_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Gera dados de referência (baseline) para o modelo de fraude.

    Simula transações normais de uma plataforma de pagamentos com
    distribuições estáveis, representando o período de treinamento.

    Args:
        n_samples: Número de transações a gerar.
        seed: Semente para reproduzibilidade.

    Returns:
        DataFrame com transações de referência.
    """
    rng = np.random.RandomState(seed)

    # Features numéricas - distribuições de referência
    valor_transacao = rng.lognormal(mean=4.5, sigma=1.0, size=n_samples)
    tempo_conta_cliente = rng.exponential(scale=180, size=n_samples).astype(int)
    num_transacoes_24h = rng.poisson(lam=3, size=n_samples)
    valor_medio_historico = rng.lognormal(mean=4.2, sigma=0.8, size=n_samples)
    distancia_localizacao = rng.exponential(scale=15, size=n_samples)
    hora_transacao = rng.choice(range(24), size=n_samples, p=_hour_distribution_ref())
    score_risco_dispositivo = rng.beta(a=2, b=8, size=n_samples)
    tentativas_senha = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.2, 0.08, 0.02])
    is_weekend = rng.binomial(1, 0.28, size=n_samples)

    # Features categóricas - distribuições de referência
    tipo_cartao = rng.choice(
        ["credito", "debito", "prepago"], size=n_samples, p=[0.55, 0.35, 0.10]
    )
    canal_transacao = rng.choice(
        ["app", "web", "pos", "telefone"], size=n_samples, p=[0.40, 0.30, 0.25, 0.05]
    )
    pais_origem = rng.choice(
        ["BR", "US", "PT", "AR", "OTHER"], size=n_samples, p=[0.70, 0.10, 0.08, 0.07, 0.05]
    )

    # Razão valor/média
    razao_valor_medio = valor_transacao / (valor_medio_historico + 1e-6)

    # Gerar rótulo de fraude (~5% baseline)
    # Fraude correlacionada com: valor alto, distância grande, score risco alto
    fraud_score = (
        0.3 * _normalize(valor_transacao)
        + 0.25 * _normalize(distancia_localizacao)
        + 0.25 * _normalize(score_risco_dispositivo)
        + 0.1 * _normalize(tentativas_senha)
        + 0.1 * _normalize(razao_valor_medio)
    )
    fraud_threshold = np.percentile(fraud_score, 95)
    fraude = (fraud_score >= fraud_threshold).astype(int)

    # Timestamps
    base_date = pd.Timestamp("2025-01-01")
    timestamps = [base_date + pd.Timedelta(hours=int(h)) for h in rng.uniform(0, 90 * 24, n_samples)]

    df = pd.DataFrame(
        {
            "transaction_id": [f"REF_{i:06d}" for i in range(n_samples)],
            "timestamp": sorted(timestamps),
            "valor_transacao": np.round(valor_transacao, 2),
            "tempo_conta_cliente": np.clip(tempo_conta_cliente, 1, 3650),
            "num_transacoes_24h": num_transacoes_24h,
            "valor_medio_historico": np.round(valor_medio_historico, 2),
            "distancia_localizacao": np.round(distancia_localizacao, 2),
            "hora_transacao": hora_transacao,
            "tipo_cartao": tipo_cartao,
            "canal_transacao": canal_transacao,
            "pais_origem": pais_origem,
            "score_risco_dispositivo": np.round(score_risco_dispositivo, 4),
            "tentativas_senha": tentativas_senha,
            "is_weekend": is_weekend,
            "razao_valor_medio": np.round(razao_valor_medio, 4),
            "fraude": fraude,
            "periodo": "referencia",
        }
    )
    return df


def generate_production_data(n_samples: int = 5000, seed: int = 123) -> pd.DataFrame:
    """
    Gera dados de produção COM drift simulado.

    Implementa três tipos de drift conforme discutido no Documento 04:
      - Covariate drift: P_t(X) != P_{t+Δ}(X) — mudanças nas distribuições
        de valor_transacao e distancia_localizacao.
      - Prior drift: P_t(Y) != P_{t+Δ}(Y) — aumento na taxa de fraude.
      - Concept drift: P_t(Y|X) != P_{t+Δ}(Y|X) — novo padrão de fraude
        via canal telefone.

    Args:
        n_samples: Número de transações a gerar.
        seed: Semente para reproduzibilidade.

    Returns:
        DataFrame com transações de produção (com drift).
    """
    rng = np.random.RandomState(seed)

    # === COVARIATE DRIFT ===
    # Valores de transação maiores (shift na distribuição lognormal)
    valor_transacao = rng.lognormal(mean=5.0, sigma=1.2, size=n_samples)

    tempo_conta_cliente = rng.exponential(scale=120, size=n_samples).astype(int)
    num_transacoes_24h = rng.poisson(lam=4, size=n_samples)
    valor_medio_historico = rng.lognormal(mean=4.5, sigma=0.9, size=n_samples)

    # Distâncias maiores (covariate drift)
    distancia_localizacao = rng.exponential(scale=30, size=n_samples)

    hora_transacao = rng.choice(range(24), size=n_samples, p=_hour_distribution_prod())
    score_risco_dispositivo = rng.beta(a=2.5, b=6, size=n_samples)
    tentativas_senha = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.60, 0.22, 0.12, 0.06])
    is_weekend = rng.binomial(1, 0.28, size=n_samples)

    # Mudança na distribuição de canais (concept drift: mais telefone)
    tipo_cartao = rng.choice(
        ["credito", "debito", "prepago"], size=n_samples, p=[0.50, 0.30, 0.20]
    )
    canal_transacao = rng.choice(
        ["app", "web", "pos", "telefone"], size=n_samples, p=[0.30, 0.25, 0.25, 0.20]
    )
    pais_origem = rng.choice(
        ["BR", "US", "PT", "AR", "OTHER"], size=n_samples, p=[0.55, 0.15, 0.10, 0.10, 0.10]
    )

    razao_valor_medio = valor_transacao / (valor_medio_historico + 1e-6)

    # === PRIOR DRIFT + CONCEPT DRIFT ===
    # Taxa de fraude maior (~8%) + novo padrão: fraudes por canal telefone
    fraud_score = (
        0.25 * _normalize(valor_transacao)
        + 0.20 * _normalize(distancia_localizacao)
        + 0.20 * _normalize(score_risco_dispositivo)
        + 0.10 * _normalize(tentativas_senha)
        + 0.10 * _normalize(razao_valor_medio)
        + 0.15 * (canal_transacao == "telefone").astype(float)
    )
    fraud_threshold = np.percentile(fraud_score, 92)
    fraude = (fraud_score >= fraud_threshold).astype(int)

    # Timestamps (período posterior ao de referência)
    base_date = pd.Timestamp("2025-04-01")
    timestamps = [base_date + pd.Timedelta(hours=int(h)) for h in rng.uniform(0, 90 * 24, n_samples)]

    df = pd.DataFrame(
        {
            "transaction_id": [f"PROD_{i:06d}" for i in range(n_samples)],
            "timestamp": sorted(timestamps),
            "valor_transacao": np.round(valor_transacao, 2),
            "tempo_conta_cliente": np.clip(tempo_conta_cliente, 1, 3650),
            "num_transacoes_24h": num_transacoes_24h,
            "valor_medio_historico": np.round(valor_medio_historico, 2),
            "distancia_localizacao": np.round(distancia_localizacao, 2),
            "hora_transacao": hora_transacao,
            "tipo_cartao": tipo_cartao,
            "canal_transacao": canal_transacao,
            "pais_origem": pais_origem,
            "score_risco_dispositivo": np.round(score_risco_dispositivo, 4),
            "tentativas_senha": tentativas_senha,
            "is_weekend": is_weekend,
            "razao_valor_medio": np.round(razao_valor_medio, 4),
            "fraude": fraude,
            "periodo": "producao",
        }
    )
    return df


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normaliza um array para [0, 1]."""
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def _hour_distribution_ref() -> list:
    """Distribuição de horas para período de referência (mais atividade diurna)."""
    probs = np.array([
        0.01, 0.005, 0.005, 0.005, 0.01, 0.02,  # 0-5h
        0.03, 0.05, 0.07, 0.08, 0.09, 0.08,      # 6-11h
        0.07, 0.06, 0.06, 0.06, 0.05, 0.05,       # 12-17h
        0.05, 0.04, 0.04, 0.03, 0.02, 0.015,      # 18-23h
    ])
    return (probs / probs.sum()).tolist()


def _hour_distribution_prod() -> list:
    """Distribuição de horas para produção (mais atividade noturna — drift)."""
    probs = np.array([
        0.03, 0.025, 0.02, 0.02, 0.025, 0.03,    # 0-5h (mais atividade noturna)
        0.03, 0.04, 0.06, 0.07, 0.07, 0.07,       # 6-11h
        0.06, 0.05, 0.05, 0.05, 0.05, 0.05,       # 12-17h
        0.05, 0.04, 0.04, 0.035, 0.03, 0.025,     # 18-23h
    ])
    return (probs / probs.sum()).tolist()


def generate_full_dataset(output_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera o dataset completo (referência + produção) e salva em CSV.

    Args:
        output_dir: Diretório de saída para o CSV.

    Returns:
        Tupla (df_referencia, df_producao).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df_ref = generate_reference_data()
    df_prod = generate_production_data()
    df_full = pd.concat([df_ref, df_prod], ignore_index=True)

    filepath = output_path / "dataset.csv"
    df_full.to_csv(filepath, index=False)
    print(f"Dataset gerado: {filepath} ({len(df_full)} registros)")
    print(f"  - Referência: {len(df_ref)} registros ({df_ref['fraude'].mean():.1%} fraude)")
    print(f"  - Produção:   {len(df_prod)} registros ({df_prod['fraude'].mean():.1%} fraude)")

    return df_ref, df_prod


if __name__ == "__main__":
    generate_full_dataset()
