"""
Testes unitários para o módulo evaluation da Aula 03.

Valida funções de métricas, limiar sensível a custo e visualizações,
conforme discutido na seção 'Saiba Mais' do documento acadêmico.

Referências:
    Fawcett, T. (2006). An introduction to ROC analysis. Pattern
    Recognition Letters, 27(8), 861-874.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    calculate_metrics,
    generate_evaluation_report,
    optimal_cost_sensitive_threshold,
)


@pytest.fixture
def binary_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria predições binárias sintéticas para testes."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, size=200), 0, 1)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


class TestCalculateMetrics:
    """Testes para calculate_metrics()."""

    def test_returns_all_metrics(
        self, binary_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred, y_proba = binary_predictions
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "auc_roc" in metrics

    def test_metrics_range(
        self, binary_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        y_true, y_pred, y_proba = binary_predictions
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} fora do intervalo: {value}"

    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        metrics = calculate_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_without_proba(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        metrics = calculate_metrics(y_true, y_pred)
        assert "auc_roc" not in metrics


class TestOptimalCostSensitiveThreshold:
    """Testes para optimal_cost_sensitive_threshold()."""

    def test_returns_expected_keys(
        self, binary_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        y_true, _, y_proba = binary_predictions
        result = optimal_cost_sensitive_threshold(y_true, y_proba)
        assert "optimal_threshold" in result
        assert "min_cost" in result
        assert "tpr_at_optimal" in result
        assert "fpr_at_optimal" in result

    def test_threshold_in_valid_range(
        self, binary_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        y_true, _, y_proba = binary_predictions
        result = optimal_cost_sensitive_threshold(y_true, y_proba)
        assert 0 <= result["optimal_threshold"] <= 1

    def test_higher_cost_fn_lowers_threshold(
        self, binary_predictions: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Conforme seção 'Saiba Mais': custo FN maior → limiar menor."""
        y_true, _, y_proba = binary_predictions
        result_low = optimal_cost_sensitive_threshold(
            y_true, y_proba, cost_fn=1.0, cost_fp=1.0
        )
        result_high = optimal_cost_sensitive_threshold(
            y_true, y_proba, cost_fn=50.0, cost_fp=1.0
        )
        # Higher cost_fn should favor lower threshold (more recall)
        assert result_high["tpr_at_optimal"] >= result_low["tpr_at_optimal"] - 0.1


class TestGenerateEvaluationReport:
    """Testes para generate_evaluation_report()."""

    def test_report_contains_strategy_names(self) -> None:
        metrics = {
            "Baseline": {"accuracy": 0.85, "f1": 0.80},
            "Re-treinamento": {"accuracy": 0.90, "f1": 0.88},
        }
        report = generate_evaluation_report(metrics)
        assert "Baseline" in report
        assert "Re-treinamento" in report
        assert "0.8500" in report

    def test_report_with_cost_analysis(self) -> None:
        metrics = {"Baseline": {"accuracy": 0.85}}
        cost = {
            "optimal_threshold": 0.35,
            "min_cost": 1.23,
            "tpr_at_optimal": 0.92,
            "fpr_at_optimal": 0.15,
            "cost_fn": 10.0,
            "cost_fp": 1.0,
        }
        report = generate_evaluation_report(metrics, cost_analysis=cost)
        assert "Limiar" in report
        assert "0.3500" in report
