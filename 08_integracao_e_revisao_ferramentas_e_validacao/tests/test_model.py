"""
Testes unitários para o módulo model (FraudDetector).

Valida inicialização, treinamento, predição e detecção de drift do
FraudDetector, conforme o pipeline da Aula 8 (Documento 04).
"""

import numpy as np
import pandas as pd
import pytest

from src.model import FraudDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_data() -> tuple:
    """Gera dados de treino sintéticos para o FraudDetector."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame({
        "valor_transacao": np.random.lognormal(4.5, 1.0, n),
        "tempo_conta_cliente": np.random.exponential(180, n),
        "num_transacoes_24h": np.random.poisson(3, n).astype(float),
        "distancia_localizacao": np.random.exponential(15, n),
        "score_risco_dispositivo": np.random.beta(2, 8, n),
    })
    y = pd.Series(np.random.choice([0, 1], n, p=[0.93, 0.07]), name="fraude")
    return X, y


@pytest.fixture
def production_data_no_drift(training_data: tuple) -> pd.DataFrame:
    """Gera dados de produção SEM drift (mesma distribuição)."""
    np.random.seed(42)
    X, _ = training_data
    return X.sample(frac=1, replace=True, random_state=99).reset_index(drop=True)


@pytest.fixture
def production_data_with_drift() -> pd.DataFrame:
    """Gera dados de produção COM drift (distribuições diferentes)."""
    np.random.seed(99)
    n = 300
    return pd.DataFrame({
        "valor_transacao": np.random.lognormal(6.0, 1.5, n),
        "tempo_conta_cliente": np.random.exponential(50, n),
        "num_transacoes_24h": np.random.poisson(10, n).astype(float),
        "distancia_localizacao": np.random.exponential(60, n),
        "score_risco_dispositivo": np.random.beta(5, 3, n),
    })


@pytest.fixture
def fitted_detector(training_data: tuple) -> FraudDetector:
    """Retorna FraudDetector já treinado."""
    X, y = training_data
    detector = FraudDetector(n_estimators=20, max_depth=5, random_state=42)
    detector.fit(X, y, store_reference=True)
    return detector


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_initialization(self) -> None:
        """Verifica valores padrão do FraudDetector."""
        det = FraudDetector()
        assert det.drift_threshold == 0.05
        assert not det.is_fitted
        assert det.reference_distributions == {}

    def test_custom_parameters(self) -> None:
        """Verifica que parâmetros personalizados são aceitos."""
        det = FraudDetector(n_estimators=50, max_depth=7, drift_threshold=0.01)
        assert det.model.n_estimators == 50
        assert det.model.max_depth == 7
        assert det.drift_threshold == 0.01


class TestFit:
    def test_fit_sets_fitted_flag(self, training_data: tuple) -> None:
        """Verifica que fit() marca o modelo como treinado."""
        X, y = training_data
        det = FraudDetector(n_estimators=10, random_state=42)
        det.fit(X, y)
        assert det.is_fitted

    def test_fit_stores_reference(self, training_data: tuple) -> None:
        """Verifica que fit() armazena distribuições de referência."""
        X, y = training_data
        det = FraudDetector(n_estimators=10, random_state=42)
        det.fit(X, y, store_reference=True)
        assert len(det.reference_distributions) > 0
        for col in X.select_dtypes(include=[np.number]).columns:
            assert col in det.reference_distributions

    def test_fit_returns_self(self, training_data: tuple) -> None:
        """Verifica encadeamento (fit retorna self)."""
        X, y = training_data
        det = FraudDetector(n_estimators=10, random_state=42)
        result = det.fit(X, y)
        assert result is det


class TestPredict:
    def test_predict_shape(self, fitted_detector: FraudDetector, training_data: tuple) -> None:
        """Verifica shape correto das predições."""
        X, _ = training_data
        preds = fitted_detector.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_binary_values(self, fitted_detector: FraudDetector, training_data: tuple) -> None:
        """Verifica que predições são binárias (0 ou 1)."""
        X, _ = training_data
        preds = fitted_detector.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, fitted_detector: FraudDetector, training_data: tuple) -> None:
        """Verifica shape das probabilidades (n_samples, 2)."""
        X, _ = training_data
        proba = fitted_detector.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_raises_if_not_fitted(self, training_data: tuple) -> None:
        """Verifica exceção se modelo não treinado."""
        X, _ = training_data
        det = FraudDetector()
        with pytest.raises(RuntimeError, match="Modelo não treinado"):
            det.predict(X)


class TestCheckDrift:
    def test_drift_detected_with_shifted_data(
        self, fitted_detector: FraudDetector, production_data_with_drift: pd.DataFrame
    ) -> None:
        """Verifica que drift é detectado em dados com distribuição deslocada."""
        results = fitted_detector.check_drift(production_data_with_drift)
        drifted = [k for k, v in results.items() if v["drift_detected"]]
        assert len(drifted) > 0, "Deveria detectar drift em pelo menos 1 feature"

    def test_drift_returns_expected_keys(
        self, fitted_detector: FraudDetector, production_data_with_drift: pd.DataFrame
    ) -> None:
        """Verifica que resultado do drift tem chaves esperadas."""
        results = fitted_detector.check_drift(production_data_with_drift)
        for feat, info in results.items():
            assert "ks_statistic" in info
            assert "p_value" in info
            assert "drift_detected" in info


class TestScore:
    def test_score_returns_all_metrics(
        self, fitted_detector: FraudDetector, training_data: tuple
    ) -> None:
        """Verifica que score() retorna todas as métricas esperadas."""
        X, y = training_data
        metrics = fitted_detector.score(X, y)
        expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert expected_keys == set(metrics.keys())
        for v in metrics.values():
            assert 0.0 <= v <= 1.0
