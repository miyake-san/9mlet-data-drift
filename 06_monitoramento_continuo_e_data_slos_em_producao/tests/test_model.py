"""
Testes unitários para os módulos model.py e training.py.

Testa SLOMonitor, MonitoringPipeline e CreditModelTrainer,
incluindo verificações de acurácia, drift KS, qualidade de dados e PSI.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MonitoringPipeline, MonitoringReport, SLOCheckResult, SLOMonitor
from src.training import CreditModelTrainer


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def slo_monitor() -> SLOMonitor:
    """Cria SLOMonitor com thresholds padrão."""
    return SLOMonitor(
        accuracy_slo=0.85,
        drift_p_value_threshold=0.05,
        missing_rate_slo=0.01,
        psi_threshold=0.25,
    )


@pytest.fixture
def rng() -> np.random.RandomState:
    """Random state com seed fixa."""
    return np.random.RandomState(42)


@pytest.fixture
def reference_data(rng: np.random.RandomState) -> pd.DataFrame:
    """Dados de referência (sem drift)."""
    n = 500
    return pd.DataFrame({
        "idade": rng.normal(35, 10, n),
        "renda_mensal": rng.lognormal(9.0, 0.8, n),
        "score_credito": rng.normal(600, 150, n),
        "tempo_emprego": rng.exponential(5, n),
        "valor_emprestimo": rng.lognormal(9.5, 0.6, n),
    })


@pytest.fixture
def production_data_with_drift(rng: np.random.RandomState) -> pd.DataFrame:
    """Dados de produção com drift introduzido."""
    n = 500
    return pd.DataFrame({
        "idade": rng.normal(45, 10, n),           # +10 anos de drift
        "renda_mensal": rng.lognormal(8.5, 0.8, n),  # redução de renda
        "score_credito": rng.normal(500, 150, n),  # -100 pontos
        "tempo_emprego": rng.exponential(5, n),    # sem drift
        "valor_emprestimo": rng.lognormal(9.5, 0.6, n),  # sem drift
    })


@pytest.fixture
def train_data(rng: np.random.RandomState):
    """Dados de treino para CreditModelTrainer."""
    n = 300
    X = pd.DataFrame({
        "idade": rng.normal(35, 10, n),
        "renda_mensal": rng.lognormal(9.0, 0.8, n),
        "score_credito": rng.normal(600, 150, n),
        "tempo_emprego": rng.exponential(5, n),
        "valor_emprestimo": rng.lognormal(9.5, 0.6, n),
        "taxa_utilizacao_credito": rng.beta(2, 5, n),
        "num_parcelas": rng.choice([12, 24, 36], n),
        "historico_atrasos": rng.poisson(1, n),
        "saldo_conta": rng.lognormal(8.0, 1.0, n),
        "qtd_dependentes": rng.poisson(1, n),
    })
    y = pd.Series(rng.choice([0, 1], n, p=[0.7, 0.3]), name="target")
    return X, y


# ======================================================================
# Testes — SLOMonitor
# ======================================================================


class TestSLOMonitor:
    """Testes para o SLOMonitor."""

    def test_check_accuracy_passes(self, slo_monitor: SLOMonitor) -> None:
        """Verifica check de acurácia com valor acima do SLO."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])  # 100% acerto
        result = slo_monitor.check_accuracy(y_true, y_pred)
        assert result.passed is True
        assert result.metric_value == 1.0

    def test_check_accuracy_fails(self, slo_monitor: SLOMonitor) -> None:
        """Verifica check de acurácia com valor abaixo do SLO."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])  # 0% acerto
        result = slo_monitor.check_accuracy(y_true, y_pred)
        assert result.passed is False
        assert result.metric_value == 0.0

    def test_check_drift_ks_no_drift(self, slo_monitor: SLOMonitor, rng) -> None:
        """Verifica KS sem drift (mesma distribuição)."""
        data = rng.normal(0, 1, 500)
        ref = pd.Series(data[:250])
        prod = pd.Series(data[250:])
        result = slo_monitor.check_drift_ks(ref, prod, "feature_test")
        # Mesma distribuição → p-value alto → passed
        assert result.passed == True

    def test_check_drift_ks_with_drift(self, slo_monitor: SLOMonitor, rng) -> None:
        """Verifica KS com drift (distribuições diferentes)."""
        ref = pd.Series(rng.normal(0, 1, 500))
        prod = pd.Series(rng.normal(5, 1, 500))  # drift forte
        result = slo_monitor.check_drift_ks(ref, prod, "feature_drift")
        assert result.passed == False  # Drift detectado

    def test_check_missing_rate_passes(self, slo_monitor: SLOMonitor) -> None:
        """Verifica missing rate abaixo do SLO."""
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        results = slo_monitor.check_missing_rate(df)
        assert all(r.passed for r in results)

    def test_check_missing_rate_fails(self, slo_monitor: SLOMonitor) -> None:
        """Verifica missing rate acima do SLO (1%)."""
        # 20% missings → deve falhar
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        results = slo_monitor.check_missing_rate(df)
        failed = [r for r in results if not r.passed]
        assert len(failed) >= 1

    def test_calculate_psi_no_drift(self, slo_monitor: SLOMonitor, rng) -> None:
        """Verifica PSI baixo para distribuições idênticas."""
        data = rng.normal(0, 1, 1000)
        psi = SLOMonitor.calculate_psi(data[:500], data[500:])
        assert psi < 0.10

    def test_calculate_psi_with_drift(self, slo_monitor: SLOMonitor, rng) -> None:
        """Verifica PSI alto para distribuições diferentes."""
        ref = rng.normal(0, 1, 500)
        prod = rng.normal(5, 1, 500)
        psi = SLOMonitor.calculate_psi(ref, prod)
        assert psi > 0.25  # Drift severo

    def test_check_psi_returns_result(self, slo_monitor: SLOMonitor, rng) -> None:
        """Verifica que check_psi retorna SLOCheckResult."""
        ref = rng.normal(0, 1, 500)
        prod = rng.normal(0, 1, 500)
        result = slo_monitor.check_psi(ref, prod, "test_feature")
        assert isinstance(result, SLOCheckResult)
        assert result.name == "psi_test_feature"


# ======================================================================
# Testes — MonitoringPipeline
# ======================================================================


class TestMonitoringPipeline:
    """Testes para o MonitoringPipeline."""

    def test_run_full_check_returns_report(
        self, reference_data, production_data_with_drift
    ) -> None:
        """Verifica que run_full_check retorna MonitoringReport."""
        pipeline = MonitoringPipeline(
            features_to_monitor=["idade", "score_credito"],
        )
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 500)
        y_pred = rng.choice([0, 1], 500)

        report = pipeline.run_full_check(
            y_true, y_pred, reference_data, production_data_with_drift,
        )
        assert isinstance(report, MonitoringReport)
        assert len(report.checks) > 0

    def test_report_detects_drift(
        self, reference_data, production_data_with_drift
    ) -> None:
        """Verifica que pipeline detecta drift nos dados com drift."""
        pipeline = MonitoringPipeline(
            features_to_monitor=["idade", "score_credito"],
        )
        rng = np.random.RandomState(42)
        y_true = rng.choice([0, 1], 500)
        y_pred = rng.choice([0, 1], 500)

        report = pipeline.run_full_check(
            y_true, y_pred, reference_data, production_data_with_drift,
        )
        # Com drift forte, deve haver checks falhando
        assert report.failed_count > 0


# ======================================================================
# Testes — CreditModelTrainer
# ======================================================================


class TestCreditModelTrainer:
    """Testes para o CreditModelTrainer."""

    def test_init_valid_model_type(self) -> None:
        """Verifica inicialização com tipos válidos."""
        for model_type in ["logistic", "rf", "gbm"]:
            trainer = CreditModelTrainer(model_type=model_type)
            assert trainer.model_type == model_type

    def test_init_invalid_model_type(self) -> None:
        """Verifica erro com tipo de modelo inválido."""
        with pytest.raises(ValueError, match="não suportado"):
            CreditModelTrainer(model_type="invalid")

    def test_train_model(self, train_data) -> None:
        """Verifica que modelo pode ser treinado."""
        X, y = train_data
        trainer = CreditModelTrainer(model_type="logistic", random_state=42)
        model = trainer.train_model(X, y)
        assert model is not None
        assert trainer.model is not None

    def test_predict_after_train(self, train_data) -> None:
        """Verifica previsões após treinamento."""
        X, y = train_data
        trainer = CreditModelTrainer(model_type="logistic", random_state=42)
        trainer.train_model(X, y)
        predictions = trainer.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})

    def test_predict_before_train_raises(self) -> None:
        """Verifica erro ao prever sem treinar."""
        trainer = CreditModelTrainer(model_type="logistic")
        with pytest.raises(RuntimeError, match="não treinado"):
            trainer.predict(pd.DataFrame({"a": [1, 2]}))

    def test_cross_validate(self, train_data) -> None:
        """Verifica validação cruzada."""
        X, y = train_data
        trainer = CreditModelTrainer(model_type="logistic", random_state=42)
        scores = trainer.cross_validate(X, y, cv=3)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_score(self, train_data) -> None:
        """Verifica cálculo de score."""
        X, y = train_data
        trainer = CreditModelTrainer(model_type="logistic", random_state=42)
        trainer.train_model(X, y)
        score = trainer.score(X, y)
        assert 0 <= score <= 1
