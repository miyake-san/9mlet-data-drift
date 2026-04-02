"""
Gerador do dataset sintético FinBank Transactions para a Aula 7.

Simula transações financeiras com drift temporal injetado, conforme
o caso FinBank discutido no material da aula (Sculley et al., 2015).

Meses 1–3: distribuição estável (baseline).
Meses 4–6: drift gradual (aumento em amount, mudança em channel).
Meses 7–8: drift abrupto (mudança forte em múltiplas features).

Uso:
    python scripts/generate_dataset.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_finbank_dataset(
    n_samples: int = 10_000,
    seed: int = 42,
    output_path: str = "data/raw/finbank_transactions.csv",
) -> pd.DataFrame:
    """
    Gera dataset sintético de transações FinBank com drift injetado.

    Implementa a criação de dados com drift gradual e abrupto conforme
    descrito no caso FinBank da Aula 7. O drift simula cenários reais
    onde o perfil de transações muda ao longo do tempo, exigindo
    monitoramento contínuo (Gama et al., 2014).

    Args:
        n_samples: Número total de transações a gerar.
        seed: Seed para reprodutibilidade.
        output_path: Caminho de saída do CSV.

    Returns:
        DataFrame com as transações geradas.
    """
    rng = np.random.RandomState(seed)

    # Distribuir amostras por mês (8 meses)
    samples_per_month = n_samples // 8
    remainder = n_samples - samples_per_month * 8
    month_counts = [samples_per_month] * 8
    month_counts[-1] += remainder

    records = []

    for month_idx, n_month in enumerate(month_counts, start=1):
        # ---- Parâmetros base que variam conforme o mês (drift) ----

        # Meses 1–3: baseline estável
        if month_idx <= 3:
            amount_mean, amount_std = 250.0, 120.0
            age_mean, age_std = 35.0, 10.0
            acct_age_mean = 365
            txn_count_mean = 15
            avg_amount_mean = 240.0
            channel_probs = [0.40, 0.35, 0.20, 0.05]  # app, web, physical, phone
            intl_prob = 0.08
            fraud_rate = 0.03

        # Meses 4–6: drift gradual
        elif month_idx <= 6:
            drift_factor = (month_idx - 3) / 3.0  # 0.33, 0.67, 1.0
            amount_mean = 250.0 + 100.0 * drift_factor
            amount_std = 120.0 + 40.0 * drift_factor
            age_mean = 35.0 - 3.0 * drift_factor
            age_std = 10.0 + 2.0 * drift_factor
            acct_age_mean = int(365 - 80 * drift_factor)
            txn_count_mean = int(15 + 5 * drift_factor)
            avg_amount_mean = 240.0 + 60.0 * drift_factor
            # Canal: mais app, menos physical
            channel_probs = [
                0.40 + 0.10 * drift_factor,
                0.35 - 0.05 * drift_factor,
                0.20 - 0.05 * drift_factor,
                0.05,
            ]
            intl_prob = 0.08 + 0.04 * drift_factor
            fraud_rate = 0.03 + 0.02 * drift_factor

        # Meses 7–8: drift abrupto
        else:
            amount_mean, amount_std = 420.0, 200.0
            age_mean, age_std = 28.0, 14.0
            acct_age_mean = 180
            txn_count_mean = 25
            avg_amount_mean = 380.0
            channel_probs = [0.55, 0.25, 0.10, 0.10]
            intl_prob = 0.20
            fraud_rate = 0.08

        # ---- Gerar features ----
        amounts = np.maximum(rng.normal(amount_mean, amount_std, n_month), 1.0)
        num_items = rng.poisson(3, n_month) + 1
        hours = rng.choice(24, n_month, p=_hour_distribution(rng, month_idx))
        days_of_week = rng.randint(0, 7, n_month)
        ages = np.clip(rng.normal(age_mean, age_std, n_month), 18, 80)
        acct_ages = np.maximum(rng.poisson(acct_age_mean, n_month), 1)
        txn_counts = np.maximum(rng.poisson(txn_count_mean, n_month), 0)
        avg_amounts = np.maximum(rng.normal(avg_amount_mean, 80, n_month), 10.0)
        channels = rng.choice(
            ["app", "web", "physical", "phone"], n_month, p=channel_probs
        )
        is_international = rng.binomial(1, intl_prob, n_month)
        is_fraud = rng.binomial(1, fraud_rate, n_month)

        # Timestamps dentro do mês
        base_date = datetime(2025, month_idx, 1)
        days_in_month = 28  # simplificação
        timestamps = [
            base_date + timedelta(days=int(rng.randint(0, days_in_month)),
                                  hours=int(h), minutes=int(rng.randint(0, 60)))
            for h in hours
        ]

        for i in range(n_month):
            records.append({
                "transaction_id": f"TXN-{month_idx:02d}-{i:05d}",
                "timestamp": timestamps[i],
                "month": month_idx,
                "amount": round(amounts[i], 2),
                "num_items": int(num_items[i]),
                "hour_of_day": int(hours[i]),
                "day_of_week": int(days_of_week[i]),
                "customer_age": round(ages[i], 1),
                "account_age_days": int(acct_ages[i]),
                "transaction_count_30d": int(txn_counts[i]),
                "avg_amount_30d": round(avg_amounts[i], 2),
                "channel": channels[i],
                "is_international": int(is_international[i]),
                "is_fraud": int(is_fraud[i]),
            })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Salvar CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset gerado: {output_path} ({len(df)} registros)")
    print(f"Distribuição por mês:\n{df['month'].value_counts().sort_index()}")
    print(f"Taxa de fraude por mês:\n{df.groupby('month')['is_fraud'].mean()}")

    return df


def _hour_distribution(rng: np.random.RandomState, month: int) -> np.ndarray:
    """
    Gera distribuição de probabilidade por hora do dia.
    Meses com drift abrupto têm mais transações noturnas.
    """
    base = np.array([
        1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 7,
        7, 7, 7, 7, 6, 5, 4, 4, 3, 2, 2, 1
    ], dtype=float)

    if month >= 7:
        # Drift abrupto: mais transações em horários atípicos
        base[0:6] += 3
        base[22:24] += 3

    base /= base.sum()
    return base


if __name__ == "__main__":
    # Determinar diretório base do script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    output = os.path.join(base_dir, "data", "raw", "finbank_transactions.csv")

    generate_finbank_dataset(output_path=output)
