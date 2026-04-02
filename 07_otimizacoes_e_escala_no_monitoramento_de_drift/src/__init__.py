"""
Aula 7 — Otimizações e Escala no Monitoramento de Drift.

Módulos:
    - data_preprocessing: Ingestão e pré-processamento de transações FinBank.
    - model: DriftMonitor com PSI, K–S e Wasserstein.
    - training: Pipeline de janelas deslizantes e treinamento.
    - evaluation: Métricas de qualidade de detecção de drift.
    - utils: Funções auxiliares de I/O e configuração.
"""

from src.data_preprocessing import DataPreprocessor
from src.model import DriftMonitor, DriftResult, IncrementalPSI
from src.training import (
    train_model,
    cross_validate,
    hyperparameter_tuning,
    train_sliding_window_pipeline,
    simulate_streaming_pipeline,
)
from src.evaluation import (
    calculate_metrics,
    evaluate_drift_detection,
    plot_psi_timeline,
    plot_distribution_comparison,
    plot_drift_heatmap,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.utils import save_model, load_model, save_metrics, setup_logging, emit_alert

__all__ = [
    "DataPreprocessor",
    "DriftMonitor",
    "DriftResult",
    "IncrementalPSI",
    "train_model",
    "cross_validate",
    "hyperparameter_tuning",
    "train_sliding_window_pipeline",
    "simulate_streaming_pipeline",
    "calculate_metrics",
    "evaluate_drift_detection",
    "plot_psi_timeline",
    "plot_distribution_comparison",
    "plot_drift_heatmap",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "save_model",
    "load_model",
    "save_metrics",
    "setup_logging",
    "emit_alert",
]
