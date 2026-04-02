"""
Módulo de avaliação e visualização para a Aula 1.

Implementa métricas de desempenho e gráficos que ilustram o impacto
do data drift, conforme discutido na seção 'Saiba Mais — Consequências
do Data Drift para Modelos de ML'.

Referências:
    Pang, G. et al. (2023). AI Aging: Quantifying Temporal Degradation
    of ML Models. Nature Communications, 14(1), 2165.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Calcula métricas de classificação.

    Conforme discutido na seção 'Saiba Mais — Consequências do Data Drift',
    monitorar acurácia, F1 e AUC ao longo do tempo é essencial para
    detectar degradação do modelo.

    Args:
        y_true: Labels verdadeiras.
        y_pred: Predições do modelo.
        y_proba: Probabilidades da classe positiva (para AUC).

    Returns:
        Dicionário com accuracy, precision, recall, f1 e auc.
    """
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def calculate_metrics_per_period(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    periods: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Calcula métricas por período temporal.

    Permite visualizar a degradação do modelo ao longo do tempo,
    conforme a Figura 1 do material ('Desempenho de um modelo de ML
    decaindo ao longo do tempo em um cenário de data drift').

    Args:
        y_true: Labels verdadeiras.
        y_pred: Predições do modelo.
        periods: Vetor de períodos por amostra.

    Returns:
        Dicionário {período: métricas}.
    """
    unique_periods = sorted(np.unique(periods))
    result: dict[int, dict[str, float]] = {}

    for p in unique_periods:
        mask = periods == p
        result[int(p)] = calculate_metrics(y_true[mask], y_pred[mask])

    return result


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de Confusão",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Gera heatmap da matriz de confusão.

    Args:
        y_true: Labels verdadeiras.
        y_pred: Predições.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura (opcional).

    Returns:
        Objeto Figure do matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Classe 0", "Classe 1"],
        yticklabels=["Classe 0", "Classe 1"],
        ax=ax,
    )
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "Curva ROC",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Gera curva ROC.

    Args:
        y_true: Labels verdadeiras.
        y_proba: Probabilidades da classe positiva.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos (FPR)")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_accuracy_over_time(
    metrics_per_period: dict[int, dict[str, float]],
    metric_name: str = "accuracy",
    title: str = "Acurácia ao Longo do Tempo (por Período)",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Gera gráfico de degradação de métrica ao longo dos períodos.

    Reproduz a análise da Figura 1 do material da aula: 'Desempenho
    de um modelo de ML decaindo ao longo do tempo em um cenário de
    data drift' (Elaborado pelo autor, 2026).

    Args:
        metrics_per_period: Dicionário {período: métricas}.
        metric_name: Nome da métrica a plotar.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    periods = sorted(metrics_per_period.keys())
    values = [metrics_per_period[p][metric_name] for p in periods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, values, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Período")
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(title)
    ax.set_xticks(periods)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Anota valores
    for p, v in zip(periods, values):
        ax.annotate(f"{v:.3f}", (p, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_distribution_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "Feature",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Compara histogramas de distribuição de referência e produção.

    Implementa a visualização de histogramas comparativos conforme
    sugerido no Vídeo 3 (Detectando Drift na Prática com Python).

    Args:
        reference: Dados de referência.
        current: Dados atuais.
        feature_name: Nome da feature.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    if title is None:
        title = f"Comparação de Distribuição — {feature_name}"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(reference, bins=40, alpha=0.5, label="Referência (Treino)",
            color="steelblue", density=True)
    ax.hist(current, bins=40, alpha=0.5, label="Atual (Produção)",
            color="salmon", density=True)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Densidade")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
