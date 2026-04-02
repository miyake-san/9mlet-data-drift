"""
Testes unitários para o módulo model.

Valida DriftDetector (teste KS) e AdaptiveClassifier (fit, partial_fit,
predict, score), garantindo que as implementações são consistentes com
os conceitos de detecção de drift e aprendizado incremental da Aula 1.
"""

import numpy as np
import pytest

from src.model import AdaptiveClassifier, DriftDetector, DriftResult


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector(alpha=0.05)


@pytest.fixture
def classifier() -> AdaptiveClassifier:
    return AdaptiveClassifier(random_state=42)


@pytest.fixture
def simple_data(rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Dados linerarmente separáveis para testes rápidos."""
    X = rng.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------
# Testes DriftDetector
# ---------------------------------------------------------------

class TestDriftDetector:
    """Testes do DriftDetector (teste KS)."""

    def test_no_drift_same_distribution(self, detector: DriftDetector, rng: np.random.RandomState) -> None:
        """Sem drift quando as distribuições são iguais."""
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(0, 1, 1000)
        result = detector.detect_univariate(ref, cur, "feat")
        # Com mesma distribuição, é improvável rejeitar H0
        assert result.p_value > 0.01

    def test_drift_detected_different_distribution(self, detector: DriftDetector, rng: np.random.RandomState) -> None:
        """Drift detectado quando distribuições diferem significativamente."""
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(3, 1, 1000)  # média deslocada
        result = detector.detect_univariate(ref, cur, "feat")
        assert result.drift_detected == True  # noqa: E712
        assert result.p_value < 0.05

    def test_detect_multivariate_returns_list(self, detector: DriftDetector, rng: np.random.RandomState) -> None:
        """detect_multivariate retorna lista com um resultado por feature."""
        ref = rng.randn(500, 3)
        cur = rng.randn(500, 3)
        results = detector.detect_multivariate(ref, cur)
        assert len(results) == 3
        assert all(isinstance(r, DriftResult) for r in results)

    def test_summary_conta_features(self, detector: DriftDetector, rng: np.random.RandomState) -> None:
        """Summary conta corretamente features com drift."""
        ref = rng.randn(500, 3)
        # Desloca apenas a primeira coluna
        cur = ref.copy()
        cur[:, 0] += 5.0
        results = detector.detect_multivariate(ref, cur)
        summary = detector.summary(results)
        assert summary["features_with_drift"] >= 1

    def test_alpha_invalido_levanta_erro(self) -> None:
        """Alpha fora de (0,1) deve levantar ValueError."""
        with pytest.raises(ValueError):
            DriftDetector(alpha=0.0)
        with pytest.raises(ValueError):
            DriftDetector(alpha=1.0)


# ---------------------------------------------------------------
# Testes AdaptiveClassifier
# ---------------------------------------------------------------

class TestAdaptiveClassifier:
    """Testes do AdaptiveClassifier."""

    def test_fit_and_predict(self, classifier: AdaptiveClassifier, simple_data: tuple) -> None:
        """Modelo treina e gera predições do tamanho correto."""
        X, y = simple_data
        classifier.fit(X, y)
        preds = classifier.predict(X)
        assert preds.shape == y.shape

    def test_score_range(self, classifier: AdaptiveClassifier, simple_data: tuple) -> None:
        """Score (acurácia) deve estar entre 0 e 1."""
        X, y = simple_data
        classifier.fit(X, y)
        score = classifier.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_partial_fit_updates_history(self, classifier: AdaptiveClassifier, simple_data: tuple) -> None:
        """partial_fit deve registrar acurácia no histórico."""
        X, y = simple_data
        classifier.partial_fit(X[:50], y[:50])
        classifier.partial_fit(X[50:100], y[50:100])
        assert len(classifier.history) == 2
        assert "batch_accuracy" in classifier.history[0]

    def test_predict_before_fit_raises(self, classifier: AdaptiveClassifier) -> None:
        """Predict sem fit deve levantar RuntimeError."""
        with pytest.raises(RuntimeError):
            classifier.predict(np.array([[1, 2, 3, 4, 5]]))

    def test_partial_fit_improves_or_maintains(self, simple_data: tuple) -> None:
        """Após vários partial_fit no mesmo dado, acurácia não deve degradar."""
        X, y = simple_data
        clf = AdaptiveClassifier(random_state=42)
        # Warmup
        clf.partial_fit(X[:100], y[:100])
        score_before = clf.score(X, y)
        # Mais batches
        for i in range(0, 200, 50):
            clf.partial_fit(X[i:i+50], y[i:i+50])
        score_after = clf.score(X, y)
        # Deve manter razoável (tolerância)
        assert score_after >= score_before - 0.15
