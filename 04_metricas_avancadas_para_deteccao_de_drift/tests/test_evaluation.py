# -*- coding: utf-8 -*-
"""
test_evaluation.py – Testes unitários para funções de avaliação e visualização.

Cobre calculate_metrics, plot_distributions, plot_scatter_drift,
plot_metric_comparison e plot_correlation_heatmaps (Documento 04).
"""

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt

# Usa backend não-interativo para evitar janelas durante testes
matplotlib.use("Agg")

from src.evaluation import (
    calculate_metrics,
    plot_correlation_heatmaps,
    plot_distributions,
    plot_metric_comparison,
    plot_scatter_drift,
)


@pytest.fixture
def rng() -> np.random.RandomState:
    """Gerador com seed fixa."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng: np.random.RandomState):
    """Dados de referência e atual para testes."""
    n = 200
    idade = rng.normal(40, 10, n)
    renda_ref = 3000 + 500 * idade + rng.normal(0, 10000, n)
    renda_cur = 3000 - 500 * idade + rng.normal(0, 10000, n)
    X_ref = np.column_stack([idade, renda_ref])
    X_cur = np.column_stack([idade, renda_cur])
    return X_ref, X_cur


class TestCalculateMetrics:
    """Testes para calculate_metrics (Documento 04, Vídeos 3-4)."""

    def test_returns_all_metric_keys(self, sample_data) -> None:
        """Retorna dicionário com psi, mmd, wasserstein, energy_distance."""
        X_ref, X_cur = sample_data
        metrics = calculate_metrics(X_ref, X_cur)
        assert "psi" in metrics
        assert "mmd" in metrics
        assert "wasserstein" in metrics
        assert "energy_distance" in metrics

    def test_psi_per_feature_correct_count(self, sample_data) -> None:
        """PSI per-feature tem tamanho correto."""
        X_ref, X_cur = sample_data
        metrics = calculate_metrics(
            X_ref, X_cur, feature_names=["idade", "renda"]
        )
        assert len(metrics["psi"]["per_feature"]) == 2
        assert "idade" in metrics["psi"]["per_feature"]
        assert "renda" in metrics["psi"]["per_feature"]

    def test_mmd_is_float(self, sample_data) -> None:
        """MMD retorna float >= 0."""
        X_ref, X_cur = sample_data
        metrics = calculate_metrics(X_ref, X_cur)
        assert isinstance(metrics["mmd"], float)
        assert metrics["mmd"] >= 0.0

    def test_energy_distance_positive_for_drift(self, sample_data) -> None:
        """Energy distance > 0 para dados com drift."""
        X_ref, X_cur = sample_data
        metrics = calculate_metrics(X_ref, X_cur)
        assert metrics["energy_distance"] > 0.0

    def test_1d_input(self, rng: np.random.RandomState) -> None:
        """Funciona com entrada unidimensional."""
        a = rng.normal(0, 1, 500)
        b = rng.normal(1, 1, 500)
        metrics = calculate_metrics(a, b)
        assert metrics["mmd"] >= 0.0


class TestPlotFunctions:
    """Testes para funções de visualização (Documento 04, Figura 1)."""

    def test_plot_distributions_returns_figure(self, sample_data) -> None:
        """plot_distributions retorna objeto Figure do matplotlib."""
        X_ref, X_cur = sample_data
        fig = plot_distributions(X_ref, X_cur, feature_names=["Idade", "Renda"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_scatter_drift_returns_figure(self, sample_data) -> None:
        """plot_scatter_drift retorna Figure."""
        X_ref, X_cur = sample_data
        fig = plot_scatter_drift(
            X_ref, X_cur, labels=("Idade", "Renda")
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_metric_comparison_returns_figure(self, sample_data) -> None:
        """plot_metric_comparison retorna Figure."""
        X_ref, X_cur = sample_data
        metrics = calculate_metrics(
            X_ref, X_cur, feature_names=["idade", "renda"]
        )
        fig = plot_metric_comparison(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_correlation_heatmaps_returns_figure(self, sample_data) -> None:
        """plot_correlation_heatmaps retorna Figure."""
        X_ref, X_cur = sample_data
        fig = plot_correlation_heatmaps(
            X_ref, X_cur, feature_names=["Idade", "Renda"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
