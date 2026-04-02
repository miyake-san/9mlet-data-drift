"""
Pacote src da Aula 6 — Monitoramento Contínuo e Data SLOs em Produção.

Módulos:
    data_preprocessing: Geração e pré-processamento de dados de crédito.
    model: SLOMonitor e MonitoringPipeline para verificação de SLOs.
    training: Treinamento do modelo de classificação de crédito.
    evaluation: Métricas de avaliação, Brier Score e visualizações.
    utils: Utilitários de persistência, logging e configuração.
"""

from src.data_preprocessing import DataPreprocessor
from src.model import MonitoringPipeline, MonitoringReport, SLOCheckResult, SLOMonitor
from src.training import CreditModelTrainer
from src.evaluation import ModelEvaluator
from src.utils import save_model, load_model, save_metrics, setup_logging

__all__ = [
    "DataPreprocessor",
    "SLOMonitor",
    "SLOCheckResult",
    "MonitoringReport",
    "MonitoringPipeline",
    "CreditModelTrainer",
    "ModelEvaluator",
    "save_model",
    "load_model",
    "save_metrics",
    "setup_logging",
]
