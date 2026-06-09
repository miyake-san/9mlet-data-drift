"""
Drift Exporter — Aula 6 (Monitoramento Contínuo e Data SLOs em Produção)

Aplicação Python que:
  1. Gera/carrega o dataset sintético de crédito digital (DataPreprocessor).
  2. Treina um modelo simples de classificação na partição de REFERÊNCIA.
  3. Em loop, simula batches de produção em duas fases alternadas:
        - "pre_drift"  -> amostras vindas da partição de referência
                         (modelo se comporta dentro do SLO).
        - "pos_drift"  -> amostras vindas da partição de produção
                         (drift introduzido em idade/renda/score).
  4. Executa o MonitoringPipeline (Snippets 1-3 da aula) em cada batch
     e expõe as métricas no endpoint /metrics (formato Prometheus).

O dashboard do Grafana lê essas métricas e mostra claramente a transição
entre o cenário "antes do drift" e "depois do drift".
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Info, start_http_server
from sklearn.metrics import brier_score_loss

# Garante import do pacote src/ que já existe na Aula 6
sys.path.insert(0, "/app")

from src.data_preprocessing import DataPreprocessor  # noqa: E402
from src.model import MonitoringPipeline, SLOMonitor  # noqa: E402
from src.training import CreditModelTrainer  # noqa: E402

# ----------------------------------------------------------------------
# Configuração / logging
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
logger = logging.getLogger("drift_exporter")


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


SCRAPE_INTERVAL_SECONDS = env_int("SCRAPE_INTERVAL_SECONDS", 10)
ITERATIONS_PER_PHASE = env_int("ITERATIONS_PER_PHASE", 12)
BATCH_SIZE = env_int("BATCH_SIZE", 500)
EXPORTER_PORT = env_int("EXPORTER_PORT", 8000)

ACCURACY_SLO = env_float("ACCURACY_SLO", 0.85)
DRIFT_P_VALUE_THRESHOLD = env_float("DRIFT_P_VALUE_THRESHOLD", 0.05)
MISSING_RATE_SLO = env_float("MISSING_RATE_SLO", 0.01)
PSI_THRESHOLD = env_float("PSI_THRESHOLD", 0.25)

DATASET_PATH = Path("/app/data/raw/dataset.csv")

# Features monitoradas para drift (idênticas às do MonitoringPipeline default)
FEATURES_TO_MONITOR = [
    "idade",
    "renda_mensal",
    "score_credito",
    "tempo_emprego",
    "valor_emprestimo",
]

# ----------------------------------------------------------------------
# Métricas Prometheus
# ----------------------------------------------------------------------

ML_INFO = Info("ml_monitor_build", "Metadados do exporter de monitoramento da Aula 6")

ACCURACY_GAUGE = Gauge(
    "ml_model_accuracy",
    "Acurácia do modelo no batch corrente (SLI principal de desempenho).",
)
BRIER_GAUGE = Gauge(
    "ml_model_brier_score",
    "Brier Score do batch corrente (calibração probabilística — menor é melhor).",
)

DRIFT_PVALUE_GAUGE = Gauge(
    "ml_drift_ks_pvalue",
    "P-valor do teste de Kolmogorov-Smirnov por feature (drift se p < threshold).",
    ["feature"],
)
DRIFT_STAT_GAUGE = Gauge(
    "ml_drift_ks_statistic",
    "Estatística D do teste de Kolmogorov-Smirnov por feature.",
    ["feature"],
)
PSI_GAUGE = Gauge(
    "ml_drift_psi",
    "Population Stability Index por feature (>0.25 = drift severo).",
    ["feature"],
)
MISSING_RATE_GAUGE = Gauge(
    "ml_data_missing_rate",
    "Taxa de valores ausentes por coluna no batch corrente.",
    ["feature"],
)

CHECK_STATUS_GAUGE = Gauge(
    "ml_slo_check_status",
    "Status de cada check de SLO (1 = passou, 0 = violou).",
    ["check"],
)
CHECKS_PASSED_GAUGE = Gauge(
    "ml_slo_checks_passed",
    "Número de checks de SLO aprovados no batch corrente.",
)
CHECKS_FAILED_GAUGE = Gauge(
    "ml_slo_checks_failed",
    "Número de checks de SLO violados no batch corrente.",
)
OVERALL_STATUS_GAUGE = Gauge(
    "ml_overall_status",
    "Status geral do monitoramento (1 = HEALTHY, 0 = CRITICAL).",
)

PHASE_GAUGE = Gauge(
    "ml_phase",
    "Fase corrente da simulação (0 = pre_drift, 1 = pos_drift).",
)
PHASE_INFO = Gauge(
    "ml_phase_label",
    "Indicador booleano por rótulo de fase (útil para legendas no Grafana).",
    ["phase"],
)

BATCHES_TOTAL = Counter(
    "ml_batches_processed_total",
    "Total de batches simulados desde o start do exporter.",
    ["phase"],
)
ALERTS_TOTAL = Counter(
    "ml_slo_alerts_total",
    "Total de alertas de violação de SLO desde o start do exporter.",
    ["check"],
)

# Thresholds publicados como gauges fixos (útil para overlay em painéis)
Gauge("ml_slo_accuracy_threshold", "SLO de acurácia configurado.").set(ACCURACY_SLO)
Gauge("ml_slo_drift_pvalue_threshold", "Threshold de p-valor para teste KS.").set(
    DRIFT_P_VALUE_THRESHOLD
)
Gauge("ml_slo_missing_rate_threshold", "SLO de taxa máxima de missings.").set(
    MISSING_RATE_SLO
)
Gauge("ml_slo_psi_threshold", "Threshold de PSI para drift severo.").set(PSI_THRESHOLD)


# ----------------------------------------------------------------------
# Helpers de simulação
# ----------------------------------------------------------------------


def ensure_dataset() -> pd.DataFrame:
    """Carrega o dataset sintético, gerando-o se necessário."""
    if not DATASET_PATH.exists():
        logger.info("Dataset não encontrado em %s — gerando...", DATASET_PATH)
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        DataPreprocessor.generate_dataset(
            n_reference=5000,
            n_production=5000,
            output_path=str(DATASET_PATH),
            random_state=42,
        )
    pre = DataPreprocessor()
    df = pre.load_data(filepath=str(DATASET_PATH))
    return df


def train_reference_model(df: pd.DataFrame):
    """Treina um modelo simples nos dados de REFERÊNCIA."""
    df_ref = df[df["is_production"] == 0].copy()
    pre = DataPreprocessor()
    df_ref_clean = pre.clean_data(df_ref)
    X_ref, y_ref = pre.prepare_features(df_ref_clean)

    trainer = CreditModelTrainer(model_type="logistic", random_state=42)
    trainer.train_model(X_ref, y_ref)
    logger.info("Modelo de referência treinado: %s", trainer.model_type)
    return trainer.model


def sample_batch(df: pd.DataFrame, phase: str, rng: np.random.Generator) -> pd.DataFrame:
    """Amostra um batch da partição correspondente à fase."""
    flag = 0 if phase == "pre_drift" else 1
    pool = df[df["is_production"] == flag]
    n = min(BATCH_SIZE, len(pool))
    idx = rng.choice(pool.index.values, size=n, replace=False)
    return pool.loc[idx].copy()


def reset_check_status_gauge(features):
    """Reinicia o gauge de checks (para casos em que checks somem entre batches)."""
    # Cada check terá seu valor reescrito a cada batch; nada a fazer aqui.
    # Mantido como placeholder para evolução futura.
    return


# ----------------------------------------------------------------------
# Loop principal
# ----------------------------------------------------------------------


def publish_report(
    report,
    model,
    X_batch: pd.DataFrame,
    y_true: np.ndarray,
    df_batch: pd.DataFrame,
    phase: str,
):
    """Publica todas as métricas do MonitoringReport no Prometheus."""

    # Probabilidades / Brier
    y_proba = model.predict_proba(X_batch)[:, 1]
    brier = brier_score_loss(y_true, y_proba)
    BRIER_GAUGE.set(float(brier))

    # Missing rate por feature monitorada (atualiza só as features de interesse)
    for feat in FEATURES_TO_MONITOR:
        if feat in df_batch.columns:
            MISSING_RATE_GAUGE.labels(feature=feat).set(
                float(df_batch[feat].isna().mean())
            )

    passed = 0
    failed = 0

    for check in report.checks:
        status_val = 1.0 if check.passed else 0.0
        CHECK_STATUS_GAUGE.labels(check=check.name).set(status_val)

        if check.name == "accuracy_slo":
            ACCURACY_GAUGE.set(float(check.metric_value))
        elif check.name.startswith("drift_ks_"):
            feat = check.name.replace("drift_ks_", "")
            DRIFT_PVALUE_GAUGE.labels(feature=feat).set(float(check.metric_value))
            # Estatística D não está no SLOCheckResult; recomputamos a partir do
            # threshold/p-value não é possível, então só publicamos p-value.
        elif check.name.startswith("psi_"):
            feat = check.name.replace("psi_", "")
            PSI_GAUGE.labels(feature=feat).set(float(check.metric_value))

        if check.passed:
            passed += 1
        else:
            failed += 1
            ALERTS_TOTAL.labels(check=check.name).inc()

    CHECKS_PASSED_GAUGE.set(passed)
    CHECKS_FAILED_GAUGE.set(failed)
    OVERALL_STATUS_GAUGE.set(1 if report.overall_status == "HEALTHY" else 0)

    # Fase corrente
    PHASE_GAUGE.set(0 if phase == "pre_drift" else 1)
    PHASE_INFO.labels(phase="pre_drift").set(1 if phase == "pre_drift" else 0)
    PHASE_INFO.labels(phase="pos_drift").set(1 if phase == "pos_drift" else 0)

    BATCHES_TOTAL.labels(phase=phase).inc()


def run_loop():
    rng = np.random.default_rng(seed=2025)

    df = ensure_dataset()
    df_ref = df[df["is_production"] == 0].copy()
    df_ref_clean = DataPreprocessor().clean_data(df_ref)

    model = train_reference_model(df)

    slo_monitor = SLOMonitor(
        accuracy_slo=ACCURACY_SLO,
        drift_p_value_threshold=DRIFT_P_VALUE_THRESHOLD,
        missing_rate_slo=MISSING_RATE_SLO,
        psi_threshold=PSI_THRESHOLD,
    )
    pipeline = MonitoringPipeline(
        slo_monitor=slo_monitor,
        features_to_monitor=FEATURES_TO_MONITOR,
    )

    ML_INFO.info(
        {
            "phase_iterations": str(ITERATIONS_PER_PHASE),
            "batch_size": str(BATCH_SIZE),
            "scrape_interval_s": str(SCRAPE_INTERVAL_SECONDS),
            "accuracy_slo": str(ACCURACY_SLO),
            "psi_threshold": str(PSI_THRESHOLD),
        }
    )

    phases = ["pre_drift", "pos_drift"]
    iteration = 0

    logger.info(
        "Iniciando loop: %ds entre coletas, %d iterações por fase (pre/pos drift), batch=%d",
        SCRAPE_INTERVAL_SECONDS,
        ITERATIONS_PER_PHASE,
        BATCH_SIZE,
    )

    while True:
        phase = phases[(iteration // ITERATIONS_PER_PHASE) % len(phases)]

        df_batch = sample_batch(df, phase=phase, rng=rng)

        pre = DataPreprocessor()
        df_batch_clean = pre.clean_data(df_batch)
        X_batch, y_batch = pre.prepare_features(df_batch_clean)
        y_pred = model.predict(X_batch)

        report = pipeline.run_full_check(
            y_true=y_batch.values,
            y_pred=y_pred,
            df_reference=df_ref_clean,
            df_production=df_batch_clean,
        )

        publish_report(
            report=report,
            model=model,
            X_batch=X_batch,
            y_true=y_batch.values,
            df_batch=df_batch,
            phase=phase,
        )

        logger.info(
            "iter=%d phase=%s status=%s passed=%d failed=%d",
            iteration,
            phase,
            report.overall_status,
            report.passed_count,
            report.failed_count,
        )

        iteration += 1
        time.sleep(SCRAPE_INTERVAL_SECONDS)


def main():
    logger.info("Subindo HTTP server Prometheus em :%d/metrics", EXPORTER_PORT)
    start_http_server(EXPORTER_PORT)
    try:
        run_loop()
    except KeyboardInterrupt:
        logger.info("Encerrando exporter.")


if __name__ == "__main__":
    main()
