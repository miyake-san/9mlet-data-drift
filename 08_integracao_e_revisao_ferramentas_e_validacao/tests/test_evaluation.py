"""
Testes unitários para o módulo evaluation.

Valida cálculo de métricas de performance, métricas de drift e
funções de visualização para o pipeline da Aula 8 (Documento 04).
"""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calculate_all_features_drift,
    calculate_drift_metrics,
    calculate_metrics,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_predictions() -> tuple:
    """Gera predições binárias e probabilidades de exemplo."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0])
    y_proba = np.clip(
        np.where(y_true == 1, np.random.uniform(0.5, 0.95, len(y_true)),
                 np.random.uniform(0.05, 0.5, len(y_true))),
        0, 1,
    )
    return y_true, y_pred, y_proba


@pytest.fixture
def reference_array() -> np.ndarray:
    """Distribuição de referência (normal padrão)."""
    np.random.seed(42)
    return np.random.normal(0, 1, 500)


@pytest.fixture
def production_array_drift() -> np.ndarray:
    """Distribuição de produção COM drift (média deslocada)."""
    np.random.seed(99)
    return np.random.normal(2, 1.5, 500)


@pytest.fixture
def production_array_no_drift() -> np.ndarray:
    """Distribuição de produção SEM drift (mesma distribuição)."""
    np.random.seed(123)
    return np.random.normal(0, 1, 500)


# ---------------------------------------------------------------------------
# Testes — Métricas de Performance
# ---------------------------------------------------------------------------


class TestCalculateMetrics:
    def test_returns_expected_keys(self, binary_predictions: tuple) -> None:
        """Verifica que calculate_metrics retorna todas as métricas."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

    def test_metrics_range_zero_to_one(self, binary_predictions: tuple) -> None:
        """Verifica que todas as métricas estão em [0, 1]."""
        y_true, y_pred, y_proba = binary_predictions
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} fora de [0,1]: {value}"

    def test_without_proba(self, binary_predictions: tuple) -> None:
        """Verifica que funciona sem probabilidades (sem ROC AUC)."""
        y_true, y_pred, _ = binary_predictions
        metrics = calculate_metrics(y_true, y_pred)
        assert "roc_auc" not in metrics
        assert "accuracy" in metrics

    def test_perfect_predictions(self) -> None:
        """Verifica métricas perfeitas para predições corretas."""
        y = np.array([0, 0, 1, 1])
        metrics = calculate_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0


# ---------------------------------------------------------------------------
# Testes — Métricas de Drift
# ---------------------------------------------------------------------------


class TestCalculateDriftMetrics:
    def test_drift_detected_shifted_distributions(
        self, reference_array: np.ndarray, production_array_drift: np.ndarray
    ) -> None:
        """Verifica detecção de drift em distribuições deslocadas."""
        result = calculate_drift_metrics(reference_array, production_array_drift)
        assert result["ks_drift_detected"]
        assert result["ks_pvalue"] < 0.05
        assert result["wasserstein_distance"] > 0

    def test_no_drift_same_distribution(
        self, reference_array: np.ndarray, production_array_no_drift: np.ndarray
    ) -> None:
        """Verifica que drift NÃO é detectado em mesma distribuição."""
        result = calculate_drift_metrics(reference_array, production_array_no_drift)
        # p-value deve ser alto (sem drift significativo)
        assert result["ks_pvalue"] > 0.01

    def test_returns_all_metrics(
        self, reference_array: np.ndarray, production_array_drift: np.ndarray
    ) -> None:
        """Verifica que todas as métricas de drift são retornadas."""
        result = calculate_drift_metrics(reference_array, production_array_drift)
        expected = {"ks_statistic", "ks_pvalue", "ks_drift_detected",
                    "wasserstein_distance", "jensen_shannon_divergence"}
        assert expected == set(result.keys())


class TestCalculateAllFeaturesDrift:
    def test_returns_dataframe(self) -> None:
        """Verifica que retorna DataFrame com features como índice."""
        np.random.seed(42)
        df_ref = pd.DataFrame({"a": np.random.normal(0, 1, 200), "b": np.random.normal(5, 2, 200)})
        df_prod = pd.DataFrame({"a": np.random.normal(2, 1, 200), "b": np.random.normal(5, 2, 200)})
        result = calculate_all_features_drift(df_ref, df_prod, ["a", "b"])
        assert isinstance(result, pd.DataFrame)
        assert "a" in result.index
        assert "b" in result.index


# ---------------------------------------------------------------------------
# Testes — Visualizações
# ---------------------------------------------------------------------------


class TestPlotFunctions:
    def test_plot_confusion_matrix_returns_figure(
        self, binary_predictions: tuple
    ) -> None:
        """Verifica que plot_confusion_matrix retorna Figure."""
        import matplotlib.pyplot as plt
        y_true, y_pred, _ = binary_predictions
        fig = plot_confusion_matrix(y_true, y_pred)
        assert fig is not None
        plt.close(fig)

    def test_plot_roc_curve_returns_figure(self, binary_predictions: tuple) -> None:
        """Verifica que plot_roc_curve retorna Figure."""
        import matplotlib.pyplot as plt
        y_true, _, y_proba = binary_predictions
        fig = plot_roc_curve(y_true, y_proba)
        assert fig is not None
        plt.close(fig)
