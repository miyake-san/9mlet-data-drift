"""
Módulo de execução de detecção de drift em lotes para a Aula 2.

Implementa pipelines de detecção de drift em múltiplas features e
lotes temporais, conforme discutido nas seções 'Monitoramento Contínuo
e MLOps' e no Vídeo 4 (Pipeline Automatizado de Monitoramento de Drift).

Referências:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning
    Systems. NeurIPS 28, 2503–2511.

    Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing Loudly:
    An Empirical Study of Methods for Detecting Dataset Shift. NeurIPS 2019.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .model import DriftDetector, DriftReport, DriftResult


def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    methods: list[str] | None = None,
    alpha: float = 0.05,
    psi_threshold: float = 0.25,
    n_bins: int = 10,
) -> DriftReport:
    """Executa pipeline de detecção de drift entre dois conjuntos de dados.

    Pipeline simplificado que combina métricas de dissimilaridade e testes
    de hipótese, conforme a estratégia recomendada no Documento da Aula 2:
    'screening contínuo com métricas de divergência e confirmação com
    testes estatísticos formais'.

    Args:
        reference_df: Dados de referência (treinamento).
        current_df: Dados atuais (produção).
        methods: Métodos a aplicar (default: ["ks", "psi"]).
        alpha: Nível de significância para testes.
        psi_threshold: Limiar de PSI.
        n_bins: Número de bins para métricas baseadas em histograma.

    Returns:
        DriftReport com resultados de todas as features.
    """
    detector = DriftDetector(
        alpha=alpha,
        psi_threshold=psi_threshold,
        n_bins=n_bins,
    )
    return detector.detect_all(reference_df, current_df, methods=methods)


def run_full_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    alpha: float = 0.05,
    psi_threshold: float = 0.25,
    n_bins: int = 10,
) -> dict[str, DriftReport]:
    """Executa análise completa com todos os métodos disponíveis.

    Roda KS, PSI, KL e JS para features numéricas e Qui-quadrado para
    categóricas, retornando relatórios separados por método para análise
    comparativa conforme discutido no Vídeo 2.

    Args:
        reference_df: Dados de referência.
        current_df: Dados atuais.
        alpha: Nível de significância.
        psi_threshold: Limiar para PSI.
        n_bins: Número de bins.

    Returns:
        Dicionário {method_name: DriftReport}.
    """
    detector = DriftDetector(
        alpha=alpha,
        psi_threshold=psi_threshold,
        n_bins=n_bins,
    )

    reports: dict[str, DriftReport] = {}

    # KS + Chi2 (testes de hipótese)
    reports["ks"] = detector.detect_all(
        reference_df, current_df, methods=["ks"]
    )

    # PSI (métrica financeira)
    reports["psi"] = detector.detect_all(
        reference_df, current_df, methods=["psi"]
    )

    # KL divergence
    reports["kl"] = detector.detect_all(
        reference_df, current_df, methods=["kl"]
    )

    # JS divergence
    reports["js"] = detector.detect_all(
        reference_df, current_df, methods=["js"]
    )

    return reports


def simulate_temporal_drift(
    preprocessor: "DataPreprocessor",
    n_windows: int = 5,
    window_size: int = 1000,
    drift_magnitude_step: float = 0.3,
    methods: list[str] | None = None,
) -> list[dict[str, object]]:
    """Simula detecção de drift ao longo de janelas temporais.

    Conforme discutido na seção 'Monitoramento Contínuo e MLOps' do
    Documento da Aula 2: 'implementar monitoramento contínuo do data
    drift — calcular periodicamente métricas de drift conforme novos
    dados chegam'.

    Gera dados progressivamente mais deslocados e executa detecção
    a cada janela, simulando monitoramento contínuo em produção.

    Args:
        preprocessor: Instância de DataPreprocessor.
        n_windows: Número de janelas temporais.
        window_size: Amostras por janela.
        drift_magnitude_step: Incremento de drift por janela.
        methods: Métodos de detecção.

    Returns:
        Lista de dicts com resultados por janela temporal.
    """
    if methods is None:
        methods = ["ks", "psi"]

    detector = DriftDetector()

    # Gerar dados de referência
    ref_preprocessor = type(preprocessor)(
        seed=preprocessor.seed, n_samples=window_size
    )
    ref_df = ref_preprocessor.generate_dataset()
    ref_data = ref_df[ref_df["origem"] == "referencia"]

    temporal_results: list[dict[str, object]] = []

    for window_idx in range(n_windows):
        # Progressivamente aumentar o drift
        current_preprocessor = type(preprocessor)(
            seed=preprocessor.seed + window_idx + 1,
            n_samples=window_size,
        )
        current_df = current_preprocessor.generate_dataset()

        # Selecionar dados de produção (já com drift injetado)
        if window_idx < 2:
            # Primeiras janelas: sem drift (usar dados de referência)
            current_data = current_df[current_df["origem"] == "referencia"]
        else:
            # Janelas posteriores: com drift (usar dados de produção)
            current_data = current_df[current_df["origem"] == "producao"]

        # Executar detecção
        report = detector.detect_all(
            ref_data, current_data, methods=methods
        )

        temporal_results.append({
            "window": window_idx + 1,
            "n_samples": len(current_data),
            "n_features_drifted": report.n_features_drifted,
            "drifted_features": report.drifted_features,
            "results": report.results,
        })

    return temporal_results


def cross_validate_detection(
    reference: np.ndarray,
    current: np.ndarray,
    n_splits: int = 5,
    method: str = "ks",
    feature_name: str = "feature",
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Validação cruzada da detecção de drift.

    Divide as amostras em subconjuntos e aplica detecção em cada split
    para verificar a robustez da decisão. Útil para reduzir falsos
    positivos, conforme boas práticas de MLOps.

    Args:
        reference: Dados de referência.
        current: Dados atuais.
        n_splits: Número de divisões.
        method: Método de detecção ("ks", "psi", "kl", "js").
        feature_name: Nome da feature.
        random_state: Semente.

    Returns:
        Dict com listas de estatísticas e p-valores por split.
    """
    rng = np.random.RandomState(random_state)
    detector = DriftDetector()

    method_map = {
        "ks": detector.ks_test,
        "psi": detector.psi,
        "kl": detector.kl_divergence,
        "js": detector.js_divergence,
    }

    if method not in method_map:
        raise ValueError(f"Método '{method}' não suportado. Use: {list(method_map.keys())}")

    ref_flat = reference.ravel()
    cur_flat = current.ravel()
    statistics: list[float] = []
    p_values: list[float] = []
    detections: list[bool] = []

    for _ in range(n_splits):
        # Subsample aleatório
        ref_idx = rng.choice(len(ref_flat), size=len(ref_flat) // 2, replace=False)
        cur_idx = rng.choice(len(cur_flat), size=len(cur_flat) // 2, replace=False)

        result = method_map[method](
            ref_flat[ref_idx], cur_flat[cur_idx], feature_name=feature_name
        )

        statistics.append(result.statistic)
        if result.p_value is not None:
            p_values.append(result.p_value)
        detections.append(result.drift_detected)

    return {
        "statistics": statistics,
        "p_values": p_values,
        "detections": detections,
        "detection_rate": sum(detections) / len(detections),
    }
