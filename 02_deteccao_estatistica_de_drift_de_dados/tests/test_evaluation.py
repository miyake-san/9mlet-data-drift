"""
Testes unitários para o módulo evaluation.

Valida funções de visualização e métricas de avaliação de drift,
garantindo que gráficos são gerados corretamente e que métricas
agregadas estão consistentes.
"""

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")  # backend não-interativo para testes
import matplotlib.pyplot as plt

from src.model import DriftDetector, DriftReport, DriftResult
from src.evaluation import (
    calculate_metrics,
    generate_drift_summary,
    plot_distribution_comparison,
    plot_cdf_comparison,
    plot_psi_bars,
    plot_categorical_comparison,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def sample_report(rng: np.random.RandomState) -> DriftReport:
    """Gera um DriftReport de exemplo para testes."""
    detector = DriftDetector(alpha=0.05)
    ref = pd.DataFrame({
        "f1": rng.normal(0, 1, 500),
        "f2": rng.normal(5, 2, 500),
        "cat": rng.choice(["A", "B", "C"], 500),
    })
    cur = pd.DataFrame({
        "f1": rng.normal(2, 1, 500),
        "f2": rng.normal(5, 2, 500),
        "cat": rng.choice(["A", "B", "C"], 500, p=[0.6, 0.2, 0.2]),
    })
    return detector.detect_all(ref, cur, methods=["ks", "psi"])


# ---------------------------------------------------------------
# Tests: calculate_metrics
# ---------------------------------------------------------------

class TestCalculateMetrics:
    """Testes de métricas agregadas."""

    def test_returns_dict(self, sample_report: DriftReport) -> None:
        """Verifica que retorna dicionário."""
        metrics = calculate_metrics(sample_report)
        assert isinstance(metrics, dict)

    def test_pct_features_drifted(self, sample_report: DriftReport) -> None:
        """Verifica que % features drifted está entre 0 e 100."""
        metrics = calculate_metrics(sample_report)
        assert 0 <= metrics["pct_features_drifted"] <= 100

    def test_contains_expected_keys(self, sample_report: DriftReport) -> None:
        """Verifica presença das chaves esperadas."""
        metrics = calculate_metrics(sample_report)
        assert "pct_features_drifted" in metrics


# ---------------------------------------------------------------
# Tests: generate_drift_summary
# ---------------------------------------------------------------

class TestGenerateDriftSummary:
    """Testes da tabela resumo."""

    def test_returns_dataframe(self, sample_report: DriftReport) -> None:
        """Verifica que retorna DataFrame."""
        summary = generate_drift_summary(sample_report)
        assert isinstance(summary, pd.DataFrame)

    def test_summary_columns(self, sample_report: DriftReport) -> None:
        """Verifica colunas do resumo."""
        summary = generate_drift_summary(sample_report)
        expected = {"feature", "method", "statistic", "p_value", "threshold", "drift_detected"}
        assert expected.issubset(set(summary.columns))

    def test_summary_rows_match_results(self, sample_report: DriftReport) -> None:
        """Verifica que número de linhas corresponde aos resultados."""
        summary = generate_drift_summary(sample_report)
        assert len(summary) == len(sample_report.results)


# ---------------------------------------------------------------
# Tests: Plot functions
# ---------------------------------------------------------------

class TestPlotFunctions:
    """Testes de funções de visualização."""

    def test_plot_distribution_comparison(self, rng: np.random.RandomState) -> None:
        """Verifica que gera Figure do matplotlib."""
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(2, 1, 500)
        fig = plot_distribution_comparison(ref, cur, feature_name="test")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_cdf_comparison(self, rng: np.random.RandomState) -> None:
        """Verifica geração de figura CDF."""
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(2, 1, 500)
        fig = plot_cdf_comparison(ref, cur, feature_name="test", ks_statistic=0.35)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_psi_bars(self, sample_report: DriftReport) -> None:
        """Verifica geração de barras PSI."""
        fig = plot_psi_bars(sample_report)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_categorical_comparison(self) -> None:
        """Verifica geração de gráfico categórico."""
        ref = np.array(["A", "B", "C", "A", "B", "A"])
        cur = np.array(["A", "C", "C", "C", "B", "C"])
        fig = plot_categorical_comparison(ref, cur, feature_name="cat")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_saves_to_file(self, rng: np.random.RandomState, tmp_path) -> None:
        """Verifica que save_path persiste o gráfico."""
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(2, 1, 500)
        save_path = tmp_path / "test_fig.png"
        fig = plot_distribution_comparison(ref, cur, save_path=str(save_path))
        assert save_path.exists()
        plt.close(fig)
