"""
Testes unitários para o módulo evaluation.py.

Testa cálculo de métricas agregadas de drift, avaliação de qualidade
de detecção e funções de visualização (sem renderização).
"""

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation import (
    calculate_metrics,
    evaluate_drift_detection,
    plot_distribution_comparison,
    plot_psi_timeline,
)
from src.model import DriftResult


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def timeline_results() -> list:
    """Cria timeline simulada de resultados de drift."""
    return [
        {
            "window_months": [4],
            "n_samples": 1250,
            "alert_level": "OK",
            "aggregate_score": {
                "mean_psi": 0.05,
                "max_psi": 0.08,
                "drift_fraction": 0.0,
            },
            "drift_results": [
                DriftResult("amount", 0.05, 0.04, 0.30, 5.0, False, "none"),
                DriftResult("customer_age", 0.08, 0.06, 0.15, 3.0, False, "none"),
            ],
        },
        {
            "window_months": [5],
            "n_samples": 1250,
            "alert_level": "WARNING",
            "aggregate_score": {
                "mean_psi": 0.12,
                "max_psi": 0.18,
                "drift_fraction": 0.5,
            },
            "drift_results": [
                DriftResult("amount", 0.18, 0.09, 0.02, 15.0, True, "warning"),
                DriftResult("customer_age", 0.06, 0.04, 0.25, 2.0, False, "none"),
            ],
        },
        {
            "window_months": [7],
            "n_samples": 1250,
            "alert_level": "CRITICAL",
            "aggregate_score": {
                "mean_psi": 0.35,
                "max_psi": 0.50,
                "drift_fraction": 1.0,
            },
            "drift_results": [
                DriftResult("amount", 0.50, 0.22, 0.001, 40.0, True, "critical"),
                DriftResult("customer_age", 0.20, 0.15, 0.01, 8.0, True, "warning"),
            ],
        },
    ]


# ======================================================================
# Testes
# ======================================================================


class TestCalculateMetrics:
    """Testes para calculate_metrics()."""

    def test_returns_expected_keys(self, timeline_results: list) -> None:
        """Verifica que métricas agregadas contêm chaves esperadas."""
        metrics = calculate_metrics(timeline_results)
        assert "n_windows" in metrics
        assert "critical_windows" in metrics
        assert "warning_windows" in metrics
        assert "ok_windows" in metrics
        assert "critical_rate" in metrics

    def test_counts_are_correct(self, timeline_results: list) -> None:
        """Verifica contagem correta de janelas por nível."""
        metrics = calculate_metrics(timeline_results)
        assert metrics["n_windows"] == 3
        assert metrics["critical_windows"] == 1
        assert metrics["warning_windows"] == 1
        assert metrics["ok_windows"] == 1

    def test_empty_timeline(self) -> None:
        """Verifica retorno para timeline vazia."""
        metrics = calculate_metrics([])
        assert metrics["n_windows"] == 0


class TestEvaluateDriftDetection:
    """Testes para evaluate_drift_detection()."""

    def test_without_ground_truth(self, timeline_results: list) -> None:
        """Verifica avaliação sem ground truth."""
        result = evaluate_drift_detection(timeline_results)
        assert "n_detected" in result
        assert result["n_detected"] == 2  # WARNING + CRITICAL

    def test_with_ground_truth_perfect(self, timeline_results: list) -> None:
        """Verifica métricas com ground truth perfeito."""
        # Janelas 1 e 2 (índices) têm alertas; se ground truth = {1, 2}
        result = evaluate_drift_detection(
            timeline_results, expected_drift_windows=[1, 2]
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1_score"] == 1.0

    def test_with_ground_truth_partial(self, timeline_results: list) -> None:
        """Verifica métricas com ground truth parcial (false negatives)."""
        # Ground truth diz que janela 0 também tinha drift (mas não foi detectada)
        result = evaluate_drift_detection(
            timeline_results, expected_drift_windows=[0, 1, 2]
        )
        assert result["recall"] < 1.0
        assert result["false_negatives"] == 1


class TestPlotFunctions:
    """Testa que funções de plot executam sem erros."""

    def test_plot_psi_timeline_runs(
        self, timeline_results: list, tmp_path: Path
    ) -> None:
        """Verifica que plot_psi_timeline roda sem erro."""
        import matplotlib
        matplotlib.use("Agg")
        fig = plot_psi_timeline(
            timeline_results,
            save_path=str(tmp_path / "psi.png"),
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_distribution_comparison_runs(self, tmp_path: Path) -> None:
        """Verifica que plot_distribution_comparison roda sem erro."""
        import matplotlib
        matplotlib.use("Agg")
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(1, 1, 500)
        fig = plot_distribution_comparison(
            ref, cur,
            feature_name="amount",
            save_path=str(tmp_path / "dist.png"),
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
