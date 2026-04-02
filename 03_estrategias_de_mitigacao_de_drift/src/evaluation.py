"""
Métricas de avaliação, visualizações e comparação de estratégias de mitigação.

Implementa as funções de avaliação discutidas na seção 'Saiba Mais' do documento
acadêmico da Aula 03, incluindo:
  - Cálculo de métricas de classificação (accuracy, F1, AUC-ROC)
  - Curva ROC e confusion matrix
  - Limiar ótimo sensível a custo (Fawcett, 2006)
  - Comparação entre estratégias de mitigação ao longo do tempo
  - Monitoramento em lote com relatório de drift (inspirado no Snippet 2)

Referências:
    Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition
    Letters, 27(8), 861-874.

    Provost, F., & Fawcett, T. (2013). Data Science for Business. O'Reilly Media.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
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
    """Calcula métricas de classificação para avaliação de modelos de churn.

    Implementa as métricas discutidas na seção 'Saiba Mais': accuracy, F1,
    precision, recall e AUC-ROC, que são essenciais para avaliar degradação
    sob drift (Provost & Fawcett, 2013).

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições do modelo (classes).
        y_proba: Probabilidades preditas para a classe positiva (opcional).

    Returns:
        Dicionário com métricas calculadas.
    """
    metrics: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc_roc"] = float("nan")
    return metrics


def optimal_cost_sensitive_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> dict[str, float]:
    """Calcula o limiar ótimo sensível a custo.

    Implementa a fórmula da seção 'Saiba Mais':
        t* = argmin_t [ C_FN · π · (1 - TPR(t)) + C_FP · (1 - π) · FPR(t) ]

    No caso de churn telecom, o custo de perder um cliente valioso (FN)
    é substancialmente maior do que contatar um cliente que ficaria (FP),
    justificando redução do limiar (Fawcett, 2006; Provost & Fawcett, 2013).

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas para a classe positiva.
        cost_fn: Custo de falso negativo (cliente perdido).
        cost_fp: Custo de falso positivo (contato desnecessário).

    Returns:
        Dicionário com limiar ótimo, custo mínimo, TPR e FPR no ponto ótimo.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    prevalence = np.mean(y_true)

    # Custo esperado para cada limiar
    costs = (
        cost_fn * prevalence * (1 - tpr)
        + cost_fp * (1 - prevalence) * fpr
    )

    best_idx = int(np.argmin(costs))
    return {
        "optimal_threshold": float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5,
        "min_cost": float(costs[best_idx]),
        "tpr_at_optimal": float(tpr[best_idx]),
        "fpr_at_optimal": float(fpr[best_idx]),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plota confusion matrix com anotações.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições do modelo.
        labels: Nomes das classes.
        title: Título do gráfico.
        save_path: Caminho para salvar a figura (opcional).
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    if labels is None:
        labels = ["Não Churn (0)", "Churn (1)"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predição")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str = "Modelo",
    title: str = "Curva ROC",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Plota curva ROC com AUC anotado.

    Conforme discutido na seção 'Saiba Mais', a curva ROC é fundamental
    para avaliar degradação sob drift e para decidir o limiar ótimo
    (Fawcett, 2006).

    Args:
        y_true: Labels verdadeiros.
        y_proba: Probabilidades preditas.
        label: Label para a legenda.
        title: Título do gráfico.
        save_path: Caminho para salvar.
        figsize: Tamanho da figura.
        ax: Axes existente (para composição de múltiplas curvas).

    Returns:
        Objeto Figure do matplotlib.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falso Positivo (FPR)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_strategies_over_time(
    results: dict[str, list[dict[str, float]]],
    metric: str = "f1",
    title: str = "Comparação de Estratégias ao Longo do Tempo",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Compara métricas de diferentes estratégias de mitigação ao longo do tempo.

    Conforme discutido na seção 'Saiba Mais' e na Tabela 1 do documento
    acadêmico: re-treinamento, aprendizado contínuo, ensembles adaptativos
    e ajuste de limiar têm tempos de resposta e riscos diferentes.

    Args:
        results: Dicionário {nome_estratégia: lista_de_métricas_por_janela}.
        metric: Métrica a plotar ('f1', 'accuracy', 'auc_roc').
        title: Título do gráfico.
        save_path: Caminho para salvar.
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for strategy_name, metrics_list in results.items():
        values = [m.get(metric, float("nan")) for m in metrics_list]
        ax.plot(range(len(values)), values, marker="o", label=strategy_name)

    ax.set_xlabel("Janela Temporal")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_impact_timeline(
    metrics_by_period: dict[int, dict[str, float]],
    title: str = "Impacto do Drift no Desempenho do Modelo (T0→T3)",
    save_path: Optional[str] = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plota o impacto do drift no desempenho ao longo dos períodos T0-T3.

    Implementa a visualização da linha do tempo de impacto discutida no
    Vídeo 1: o modelo treinado no regime estável (T0) degrada
    progressivamente nos períodos de choque (T1) e adaptação (T2-T3).

    Args:
        metrics_by_period: Dicionário {período: métricas}.
        title: Título do gráfico.
        save_path: Caminho para salvar.
        figsize: Tamanho da figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    periods = sorted(metrics_by_period.keys())
    metric_names = list(metrics_by_period[periods[0]].keys())

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(periods))
    width = 0.2
    offsets = np.linspace(-width, width, len(metric_names))

    for i, metric_name in enumerate(metric_names):
        values = [metrics_by_period[p][metric_name] for p in periods]
        ax.bar(x + offsets[i], values, width, label=metric_name, alpha=0.8)

    ax.set_xlabel("Período")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Período {p}" for p in periods])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_evaluation_report(
    metrics_per_strategy: dict[str, dict[str, float]],
    cost_analysis: Optional[dict[str, float]] = None,
) -> str:
    """Gera relatório textual de avaliação comparativa.

    Conforme Snippet 2 da seção 'Hands On': antes de re-treinar cegamente,
    a equipe precisa saber se a mudança afetou variáveis, predições ou
    segmentos específicos da base.

    Args:
        metrics_per_strategy: {nome_estratégia: {métrica: valor}}.
        cost_analysis: Resultado de optimal_cost_sensitive_threshold (opcional).

    Returns:
        Relatório formatado como string.
    """
    lines = [
        "=" * 60,
        "RELATÓRIO DE AVALIAÇÃO — ESTRATÉGIAS DE MITIGAÇÃO DE DRIFT",
        "=" * 60,
        "",
    ]

    for strategy, metrics in metrics_per_strategy.items():
        lines.append(f"--- {strategy} ---")
        for metric_name, value in metrics.items():
            lines.append(f"  {metric_name:>12s}: {value:.4f}")
        lines.append("")

    if cost_analysis:
        lines.extend([
            "--- Análise de Custo (Limiar Sensível a Custo) ---",
            f"  Limiar ótimo: {cost_analysis['optimal_threshold']:.4f}",
            f"  Custo mínimo: {cost_analysis['min_cost']:.4f}",
            f"  TPR no ótimo: {cost_analysis['tpr_at_optimal']:.4f}",
            f"  FPR no ótimo: {cost_analysis['fpr_at_optimal']:.4f}",
            f"  C_FN / C_FP : {cost_analysis['cost_fn']:.1f} / {cost_analysis['cost_fp']:.1f}",
            "",
        ])

    lines.append("=" * 60)
    return "\n".join(lines)
