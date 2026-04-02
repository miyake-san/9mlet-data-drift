"""
Módulo de avaliação e visualização de resultados de detecção de drift.

Implementa gráficos comparativos de distribuições, heatmaps de métricas
e relatórios visuais, conforme discutido nas seções 'Saiba Mais —
Como identificar mudanças de distribuição?' e nos Vídeos 2 e 3 do
Documento da Aula 2.

Referências:
    Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing Loudly:
    An Empirical Study of Methods for Detecting Dataset Shift. NeurIPS 2019.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .model import DriftReport, DriftResult


def plot_distribution_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "feature",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compara histogramas de referência e produção lado a lado.

    Implementa a visualização descrita na seção 'Saiba Mais — Como
    identificar mudanças de distribuição?' do Documento da Aula 2:
    'colocar histogramas ou gráficos de densidade lado a lado ajuda
    a enxergar deslocamentos na média, dispersão ou formato.'

    Args:
        reference: Dados de referência.
        current: Dados atuais.
        feature_name: Nome da feature (usado no eixo x).
        title: Título do gráfico (auto-gerado se None).
        save_path: Caminho para salvar a figura.

    Returns:
        Objeto Figure do matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        reference.ravel(), bins=50, alpha=0.5, density=True,
        label="Referência (treino)", color="steelblue", edgecolor="white",
    )
    ax.hist(
        current.ravel(), bins=50, alpha=0.5, density=True,
        label="Produção (atual)", color="darkorange", edgecolor="white",
    )

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Densidade")
    ax.set_title(title or f"Comparação de Distribuições — {feature_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cdf_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "feature",
    ks_statistic: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compara CDFs empíricas (visualização do teste KS).

    A estatística KS corresponde à máxima distância vertical entre as
    duas CDFs: D = sup_x |F_ref(x) - F_current(x)|, conforme a seção
    'Testes estatísticos de hipóteses' do Documento.

    Args:
        reference: Dados de referência.
        current: Dados atuais.
        feature_name: Nome da feature.
        ks_statistic: Valor D do teste KS (anotado no gráfico se fornecido).
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure do matplotlib.
    """
    ref_sorted = np.sort(reference.ravel())
    cur_sorted = np.sort(current.ravel())

    ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    cur_cdf = np.arange(1, len(cur_sorted) + 1) / len(cur_sorted)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ref_sorted, ref_cdf, label="CDF Referência", color="steelblue", linewidth=2)
    ax.plot(cur_sorted, cur_cdf, label="CDF Produção", color="darkorange", linewidth=2)

    if ks_statistic is not None:
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.annotate(
            f"KS = {ks_statistic:.4f}",
            xy=(0.65, 0.15),
            xycoords="axes fraction",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )

    ax.set_xlabel(feature_name)
    ax.set_ylabel("Probabilidade Acumulada")
    ax.set_title(f"CDFs Empíricas — {feature_name} (Teste KS)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drift_heatmap(
    report: DriftReport,
    metric: str = "statistic",
    title: str = "Heatmap de Drift por Feature e Método",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Gera heatmap de métricas de drift (features × métodos).

    Visualização matricial que permite comparar rapidamente quais features
    apresentam drift em cada método, conforme a estratégia de 'screening'
    discutida no Documento.

    Args:
        report: DriftReport com resultados.
        metric: "statistic" ou "p_value".
        title: Título do gráfico.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure do matplotlib.
    """
    # Construir matrix features × methods
    data: dict[str, dict[str, float]] = {}
    for r in report.results:
        if r.feature_name not in data:
            data[r.feature_name] = {}
        value = r.statistic if metric == "statistic" else (r.p_value or 0.0)
        data[r.feature_name][r.method] = value

    df = pd.DataFrame(data).T
    df = df.fillna(0)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))
    sns.heatmap(
        df,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Método")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_psi_bars(
    report: DriftReport,
    title: str = "PSI por Feature",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Gráfico de barras de PSI com limiares de alerta.

    Sobrepõe linhas nos limiares 0.10 (moderado) e 0.25 (severo)
    conforme os padrões do setor bancário discutidos no Documento.

    Args:
        report: DriftReport com resultados.
        title: Título do gráfico.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure do matplotlib.
    """
    psi_results = [r for r in report.results if r.method == "psi"]
    if not psi_results:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sem resultados PSI", ha="center", va="center")
        return fig

    features = [r.feature_name for r in psi_results]
    values = [r.statistic for r in psi_results]

    colors = []
    for v in values:
        if v < 0.10:
            colors.append("forestgreen")
        elif v < 0.25:
            colors.append("darkorange")
        else:
            colors.append("firebrick")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, values, color=colors, edgecolor="white")

    # Limiares conforme Documento da Aula 2
    ax.axvline(x=0.10, color="orange", linestyle="--", linewidth=1.5, label="Moderado (0.10)")
    ax.axvline(x=0.25, color="red", linestyle="--", linewidth=1.5, label="Severo (0.25)")

    ax.set_xlabel("PSI")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_categorical_comparison(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "feature",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compara distribuições categóricas com barras lado a lado.

    Permite visualizar mudanças na proporção de categorias, complementando
    o teste Qui-quadrado discutido na seção 'Testes estatísticos' do
    Documento.

    Args:
        reference: Valores categóricos da referência.
        current: Valores categóricos da produção.
        feature_name: Nome da feature.
        save_path: Caminho para salvar.

    Returns:
        Objeto Figure do matplotlib.
    """
    ref_flat = reference.ravel()
    cur_flat = current.ravel()

    categories = np.union1d(np.unique(ref_flat), np.unique(cur_flat))

    ref_counts = np.array([np.sum(ref_flat == c) for c in categories])
    cur_counts = np.array([np.sum(cur_flat == c) for c in categories])

    ref_props = ref_counts / ref_counts.sum()
    cur_props = cur_counts / cur_counts.sum()

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ref_props, width, label="Referência", color="steelblue")
    ax.bar(x + width / 2, cur_props, width, label="Produção", color="darkorange")

    ax.set_xlabel("Categoria")
    ax.set_ylabel("Proporção")
    ax.set_title(f"Distribuição Categórica — {feature_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def generate_drift_summary(
    report: DriftReport,
) -> pd.DataFrame:
    """Gera tabela resumo com resultados de todos os testes.

    Consolida estatísticas, p-valores, limiares e decisões em um
    DataFrame para fácil inspeção e exportação.

    Args:
        report: DriftReport com resultados.

    Returns:
        DataFrame com colunas: feature, method, statistic, p_value,
        threshold, drift_detected.
    """
    rows = []
    for r in report.results:
        rows.append({
            "feature": r.feature_name,
            "method": r.method,
            "statistic": r.statistic,
            "p_value": r.p_value,
            "threshold": r.threshold,
            "drift_detected": r.drift_detected,
        })

    return pd.DataFrame(rows)


def calculate_metrics(
    report: DriftReport,
) -> dict[str, float]:
    """Calcula métricas agregadas do relatório de drift.

    Args:
        report: DriftReport.

    Returns:
        Dicionário com métricas:
        - pct_features_drifted: % de features com drift
        - avg_ks_statistic: média da estatística KS
        - avg_psi: média do PSI
        - max_psi: máximo PSI
    """
    ks_stats = [r.statistic for r in report.results if r.method == "ks"]
    psi_stats = [r.statistic for r in report.results if r.method == "psi"]

    metrics: dict[str, float] = {
        "pct_features_drifted": (
            report.n_features_drifted / report.n_features_total * 100
            if report.n_features_total > 0
            else 0.0
        ),
    }

    if ks_stats:
        metrics["avg_ks_statistic"] = float(np.mean(ks_stats))
        metrics["max_ks_statistic"] = float(np.max(ks_stats))

    if psi_stats:
        metrics["avg_psi"] = float(np.mean(psi_stats))
        metrics["max_psi"] = float(np.max(psi_stats))

    return metrics
