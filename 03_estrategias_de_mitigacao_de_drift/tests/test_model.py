"""
Testes unitários para o módulo model da Aula 03.

Valida BatchChurnModel, OnlineChurnModel e AdaptiveEnsembleModel,
incluindo inicialização, treinamento e predição conforme discutido
na seção 'Saiba Mais' do documento acadêmico.

Referências:
    Krawczyk, B., et al. (2017). Ensemble learning for data stream
    analysis: A survey. Information Fusion, 37, 132-156.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import DataPreprocessor
from src.model import AdaptiveEnsembleModel, BatchChurnModel, OnlineChurnModel


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    return DataPreprocessor(seed=42, n_per_period=200)


@pytest.fixture
def train_data(preprocessor: DataPreprocessor) -> tuple[pd.DataFrame, pd.Series]:
    df = preprocessor.generate_dataset()
    df = preprocessor.clean_data(df)
    periods = preprocessor.split_by_period(df)
    X, y = preprocessor.prepare_features(periods[1])
    return X, y


@pytest.fixture
def stream_data(preprocessor: DataPreprocessor) -> list[tuple[dict, int]]:
    """Stream de dados para modelos online."""
    df = preprocessor.generate_dataset()
    df = preprocessor.clean_data(df)
    feature_cols = [
        "tenure_months", "monthly_charges", "total_charges",
        "data_usage_gb", "call_duration_min", "num_complaints",
        "payment_delay_days", "contract_type", "has_premium_support",
        "competitor_offer_exposure", "customer_value_segment",
    ]
    records = df[feature_cols].to_dict(orient="records")
    labels = df["churn"].tolist()
    return list(zip(records, labels))


class TestBatchChurnModel:
    """Testes para BatchChurnModel."""

    def test_init_logistic(self) -> None:
        model = BatchChurnModel(model_type="logistic")
        assert model.model_type == "logistic"

    def test_init_random_forest(self) -> None:
        model = BatchChurnModel(model_type="random_forest")
        assert model.model_type == "random_forest"

    def test_init_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="não suportado"):
            BatchChurnModel(model_type="invalid")

    def test_fit_predict(
        self, train_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = train_data
        model = BatchChurnModel(model_type="logistic")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(
        self, train_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = train_data
        model = BatchChurnModel(model_type="logistic")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_score_returns_metrics(
        self, train_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = train_data
        model = BatchChurnModel(model_type="logistic")
        model.fit(X, y)
        scores = model.score(X, y)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "auc_roc" in scores
        assert 0 <= scores["accuracy"] <= 1


class TestOnlineChurnModel:
    """Testes para OnlineChurnModel."""

    def test_init(self) -> None:
        model = OnlineChurnModel(seed=42)
        assert model.n_samples_seen == 0
        assert model.drift_points == []

    def test_learn_one(self, stream_data: list[tuple[dict, int]]) -> None:
        model = OnlineChurnModel(seed=42)
        for x, y in stream_data[:50]:
            model.learn_one(x, y)
        assert model.n_samples_seen == 50

    def test_predict_one(self, stream_data: list[tuple[dict, int]]) -> None:
        model = OnlineChurnModel(seed=42)
        # Train on some data first
        for x, y in stream_data[:100]:
            model.learn_one(x, y)
        # Predict
        pred = model.predict_one(stream_data[100][0])
        assert pred in (0, 1, None)

    def test_accuracy_property(
        self, stream_data: list[tuple[dict, int]]
    ) -> None:
        model = OnlineChurnModel(seed=42)
        for x, y in stream_data[:100]:
            model.learn_one(x, y)
        assert 0 <= model.accuracy <= 1


class TestAdaptiveEnsembleModel:
    """Testes para AdaptiveEnsembleModel."""

    def test_init(self) -> None:
        model = AdaptiveEnsembleModel(n_models=3, seed=42)
        assert model.n_models == 3
        assert model.n_samples_seen == 0

    def test_learn_one(self, stream_data: list[tuple[dict, int]]) -> None:
        model = AdaptiveEnsembleModel(n_models=3, seed=42)
        for x, y in stream_data[:50]:
            model.learn_one(x, y)
        assert model.n_samples_seen == 50

    def test_predict_one(self, stream_data: list[tuple[dict, int]]) -> None:
        model = AdaptiveEnsembleModel(n_models=3, seed=42)
        for x, y in stream_data[:100]:
            model.learn_one(x, y)
        pred = model.predict_one(stream_data[100][0])
        assert pred in (0, 1, None)

    def test_accuracy_property(
        self, stream_data: list[tuple[dict, int]]
    ) -> None:
        model = AdaptiveEnsembleModel(n_models=3, seed=42)
        for x, y in stream_data[:100]:
            model.learn_one(x, y)
        assert 0 <= model.accuracy <= 1
