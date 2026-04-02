"""
Testes unitários para os módulos model.py e training.py.

Testa DriftMonitor, IncrementalPSI e funções de treinamento,
incluindo PSI, K–S, Wasserstein e pipeline de janelas deslizantes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import DriftMonitor, DriftResult, IncrementalPSI
from src.training import train_model, cross_validate, train_sliding_window_pipeline


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def rng() -> np.random.RandomState:
    """Random state com seed fixa."""
    return np.random.RandomState(42)


@pytest.fixture
def monitor() -> DriftMonitor:
    """Cria DriftMonitor com thresholds padrão."""
    return DriftMonitor(
        n_bins=10,
        psi_threshold=0.25,
        psi_warning=0.10,
        ks_alpha=0.05,
    )


@pytest.fixture
def baseline_data(rng: np.random.RandomState) -> dict:
    """Dados de referência (baseline) para monitoramento."""
    n = 1000
    return {
        "amount": rng.normal(250, 120, n),
        "customer_age": rng.normal(35, 10, n),
        "transaction_count_30d": rng.poisson(15, n).astype(float),
    }


@pytest.fixture
def no_drift_data(rng: np.random.RandomState) -> dict:
    """Dados correntes SEM drift (mesma distribuição que baseline)."""
    n = 1000
    rng2 = np.random.RandomState(99)
    return {
        "amount": rng2.normal(250, 120, n),
        "customer_age": rng2.normal(35, 10, n),
        "transaction_count_30d": rng2.poisson(15, n).astype(float),
    }


@pytest.fixture
def drift_data(rng: np.random.RandomState) -> dict:
    """Dados correntes COM drift significativo."""
    n = 1000
    rng2 = np.random.RandomState(99)
    return {
        "amount": rng2.normal(420, 200, n),       # drift forte
        "customer_age": rng2.normal(28, 14, n),    # drift moderado
        "transaction_count_30d": rng2.poisson(25, n).astype(float),  # drift
    }


@pytest.fixture
def train_dataframe() -> pd.DataFrame:
    """DataFrame com estrutura FinBank para testes de treinamento."""
    rng = np.random.RandomState(42)
    n = 400
    return pd.DataFrame({
        "month": np.repeat([1, 2, 3, 4, 5, 6, 7, 8], n // 8),
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="6h"),
        "amount": rng.normal(300, 100, n),
        "customer_age": rng.normal(35, 10, n),
        "transaction_count_30d": rng.poisson(15, n).astype(float),
        "is_fraud": rng.binomial(1, 0.1, n),
    })


# ======================================================================
# Testes — DriftMonitor
# ======================================================================


class TestDriftMonitorInit:
    """Testes de inicialização do DriftMonitor."""

    def test_default_params(self) -> None:
        """Verifica parâmetros padrão."""
        m = DriftMonitor()
        assert m.n_bins == 10
        assert m.psi_threshold == 0.25
        assert m.ks_alpha == 0.05

    def test_custom_params(self) -> None:
        """Verifica parâmetros customizados."""
        m = DriftMonitor(n_bins=20, psi_threshold=0.30, ks_alpha=0.01)
        assert m.n_bins == 20
        assert m.psi_threshold == 0.30
        assert m.ks_alpha == 0.01


class TestDriftMonitorFitPredict:
    """Testes de fit/predict do DriftMonitor."""

    def test_predict_without_fit_raises(
        self, monitor: DriftMonitor, no_drift_data: dict
    ) -> None:
        """Verifica que predict sem fit levanta ValueError."""
        with pytest.raises(ValueError, match="fit"):
            monitor.predict(no_drift_data)

    def test_no_drift_detected(
        self, monitor: DriftMonitor, baseline_data: dict, no_drift_data: dict
    ) -> None:
        """Verifica que dados similares não geram drift crítico."""
        monitor.fit(baseline_data)
        results = monitor.predict(no_drift_data)
        critical = [r for r in results if r.severity == "critical"]
        assert len(critical) == 0

    def test_drift_detected(
        self, monitor: DriftMonitor, baseline_data: dict, drift_data: dict
    ) -> None:
        """Verifica que dados com drift são detectados."""
        monitor.fit(baseline_data)
        results = monitor.predict(drift_data)
        any_drift = any(r.drift_detected for r in results)
        assert any_drift

    def test_result_has_correct_fields(
        self, monitor: DriftMonitor, baseline_data: dict, no_drift_data: dict
    ) -> None:
        """Verifica campos do DriftResult."""
        monitor.fit(baseline_data)
        results = monitor.predict(no_drift_data)
        assert len(results) > 0
        r = results[0]
        assert isinstance(r, DriftResult)
        assert hasattr(r, "feature")
        assert hasattr(r, "psi_value")
        assert hasattr(r, "ks_statistic")
        assert hasattr(r, "ks_pvalue")
        assert hasattr(r, "wasserstein_distance")

    def test_score_returns_dict(
        self, monitor: DriftMonitor, baseline_data: dict, no_drift_data: dict
    ) -> None:
        """Verifica que score retorna dicionário com métricas."""
        monitor.fit(baseline_data)
        score = monitor.score(no_drift_data)
        assert "mean_psi" in score
        assert "drift_fraction" in score
        assert 0 <= score["drift_fraction"] <= 1


class TestComputePSI:
    """Testes para DriftMonitor.compute_psi()."""

    def test_psi_identical_is_near_zero(self, monitor: DriftMonitor) -> None:
        """PSI de distribuições idênticas deve ser ≈ 0."""
        data = np.random.RandomState(42).normal(0, 1, 1000)
        psi = monitor.compute_psi(data, data)
        assert psi < 0.01

    def test_psi_different_is_positive(self, monitor: DriftMonitor) -> None:
        """PSI de distribuições diferentes deve ser > 0."""
        ref = np.random.RandomState(42).normal(0, 1, 1000)
        cur = np.random.RandomState(42).normal(5, 1, 1000)
        psi = monitor.compute_psi(ref, cur)
        assert psi > 0.1


class TestComputeKS:
    """Testes para DriftMonitor.compute_ks()."""

    def test_ks_identical_not_significant(self, monitor: DriftMonitor) -> None:
        """K–S de amostras da mesma distribuição: p-valor alto."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        _, p = monitor.compute_ks(a, b)
        assert p > 0.01

    def test_ks_different_is_significant(self, monitor: DriftMonitor) -> None:
        """K–S de distribuições diferentes: p-valor baixo."""
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(5, 1, 500)
        _, p = monitor.compute_ks(a, b)
        assert p < 0.05


# ======================================================================
# Testes — IncrementalPSI
# ======================================================================


class TestIncrementalPSI:
    """Testes para IncrementalPSI (cálculo incremental)."""

    def test_incremental_update(self) -> None:
        """Verifica que update acumula contagens."""
        bins = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        inc = IncrementalPSI(bins=bins)
        inc.update(np.array([0.5, 1.5, 2.5]))
        assert inc._total == 3

    def test_incremental_reset(self) -> None:
        """Verifica que reset zera contadores."""
        bins = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        inc = IncrementalPSI(bins=bins)
        inc.update(np.array([0.5, 1.5]))
        inc.reset()
        assert inc._total == 0


# ======================================================================
# Testes — training.py
# ======================================================================


class TestTrainModel:
    """Testes para train_model()."""

    def test_train_returns_classifier(self) -> None:
        """Verifica que train_model retorna classificador treinado."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = rng.binomial(1, 0.3, 100)
        clf = train_model(X, y)
        assert hasattr(clf, "predict")
        assert hasattr(clf, "predict_proba")

    def test_predictions_shape(self) -> None:
        """Verifica shape das predições."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = rng.binomial(1, 0.3, 100)
        clf = train_model(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)


class TestSlidingWindowPipeline:
    """Testes para train_sliding_window_pipeline()."""

    def test_pipeline_returns_results(
        self, monitor: DriftMonitor, train_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que pipeline retorna lista de resultados."""
        features = ["amount", "customer_age", "transaction_count_30d"]
        results = train_sliding_window_pipeline(
            train_dataframe, features, monitor,
            window_size=1, baseline_months=(1, 2, 3)
        )
        assert isinstance(results, list)
        assert len(results) > 0

    def test_pipeline_results_have_alert_level(
        self, monitor: DriftMonitor, train_dataframe: pd.DataFrame
    ) -> None:
        """Verifica que cada resultado tem alert_level."""
        features = ["amount", "customer_age", "transaction_count_30d"]
        results = train_sliding_window_pipeline(
            train_dataframe, features, monitor
        )
        for r in results:
            assert "alert_level" in r
            assert r["alert_level"] in ("OK", "WARNING", "CRITICAL")
