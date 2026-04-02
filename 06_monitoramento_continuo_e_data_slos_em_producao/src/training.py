"""
Módulo de treinamento do modelo de crédito digital.

Implementa treinamento, validação cruzada e ajuste de hiperparâmetros
para um classificador de risco de crédito, conforme cenário da fintech
descrito no DOCUMENTO_AULA_6.md.

O modelo treinado é utilizado posteriormente pelo MonitoringPipeline
para simulação de monitoramento contínuo em produção.

Referências:
    Breck, E. et al. (2017). The ML Test Score: A Rubric for ML
    Production Readiness and Technical Debt Reduction. IEEE BigData.

    Sculley, D. et al. (2015). Hidden Technical Debt in Machine
    Learning Systems. NIPS 2015.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class CreditModelTrainer:
    """Treinador de modelo de classificação de risco de crédito.

    Implementa o fluxo de treinamento para o modelo de aprovação
    de crédito da fintech, conforme descrito no DOCUMENTO_AULA_6.md:
    'uma fintech acaba de lançar um modelo de ML para aprovação
    instantânea de crédito'.

    O modelo treinado é posteriormente monitorado pelo SLOMonitor
    e MonitoringPipeline.

    Attributes:
        model_type: Tipo de modelo ('logistic', 'rf', 'gbm').
        random_state: Seed para reproduzibilidade.
        model: Pipeline treinado (scaler + classificador).
        cv_scores: Scores de validação cruzada.
    """

    SUPPORTED_MODELS = {
        "logistic": LogisticRegression,
        "rf": RandomForestClassifier,
        "gbm": GradientBoostingClassifier,
    }

    def __init__(
        self,
        model_type: str = "gbm",
        random_state: int = 42,
    ) -> None:
        """Inicializa o treinador.

        Args:
            model_type: Tipo de modelo ('logistic', 'rf', 'gbm').
            random_state: Seed para reproduzibilidade.

        Raises:
            ValueError: Se model_type não for suportado.
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"model_type '{model_type}' não suportado. "
                f"Opções: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_type = model_type
        self.random_state = random_state
        self.model: Optional[Pipeline] = None
        self.cv_scores: Optional[np.ndarray] = None

    def _create_pipeline(self, **model_params: Any) -> Pipeline:
        """Cria pipeline com scaler e classificador.

        Args:
            **model_params: Parâmetros do classificador.

        Returns:
            Pipeline sklearn.
        """
        clf_class = self.SUPPORTED_MODELS[self.model_type]

        default_params: Dict[str, Any] = {"random_state": self.random_state}
        if self.model_type == "logistic":
            default_params["max_iter"] = 1000
            default_params["solver"] = "lbfgs"
        elif self.model_type == "rf":
            default_params["n_estimators"] = 100
            default_params["max_depth"] = 10
        elif self.model_type == "gbm":
            default_params["n_estimators"] = 100
            default_params["max_depth"] = 5
            default_params["learning_rate"] = 0.1

        default_params.update(model_params)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf_class(**default_params)),
        ])

        return pipeline

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **model_params: Any,
    ) -> Pipeline:
        """Treina o modelo de crédito.

        Conforme discutido na seção 'O QUE VEM POR AÍ?' do
        DOCUMENTO_AULA_6.md, o modelo precisa ser treinado para
        depois ser monitorado continuamente em produção.

        Args:
            X_train: Features de treino.
            y_train: Target de treino.
            **model_params: Parâmetros adicionais do classificador.

        Returns:
            Pipeline treinado.
        """
        self.model = self._create_pipeline(**model_params)
        self.model.fit(X_train, y_train)

        train_accuracy = self.model.score(X_train, y_train)
        logger.info(
            f"Modelo treinado ({self.model_type}). "
            f"Acurácia treino: {train_accuracy:.4f}"
        )

        return self.model

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> np.ndarray:
        """Executa validação cruzada.

        Implementa validação cruzada para avaliar a robustez do modelo
        antes do deploy, conforme boas práticas discutidas no doc 04.

        Args:
            X: Features.
            y: Target.
            cv: Número de folds.
            scoring: Métrica de scoring.

        Returns:
            Array com scores de cada fold.
        """
        pipeline = self._create_pipeline()
        self.cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)

        logger.info(
            f"Validação cruzada ({cv} folds): "
            f"média={self.cv_scores.mean():.4f} ± {self.cv_scores.std():.4f}"
        )

        return self.cv_scores

    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        cv: int = 3,
        scoring: str = "accuracy",
    ) -> Dict[str, Any]:
        """Realiza ajuste de hiperparâmetros via GridSearch.

        Args:
            X: Features.
            y: Target.
            param_grid: Dicionário com hiperparâmetros a testar.
                        Se None, usa grid padrão para o model_type.
            cv: Número de folds.
            scoring: Métrica de scoring.

        Returns:
            Dicionário com melhores parâmetros encontrados.
        """
        if param_grid is None:
            param_grid = self._default_param_grid()

        pipeline = self._create_pipeline()

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_

        logger.info(
            f"Melhor score: {grid_search.best_score_:.4f} "
            f"com parâmetros: {grid_search.best_params_}"
        )

        return grid_search.best_params_

    def _default_param_grid(self) -> Dict[str, List[Any]]:
        """Retorna grid de hiperparâmetros padrão.

        Returns:
            Dicionário com parâmetros para GridSearchCV.
        """
        if self.model_type == "logistic":
            return {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
            }
        elif self.model_type == "rf":
            return {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [5, 10],
            }
        elif self.model_type == "gbm":
            return {
                "classifier__n_estimators": [50, 100],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
            }
        return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gera previsões com o modelo treinado.

        Args:
            X: Features de entrada.

        Returns:
            Array com previsões.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute train_model() primeiro.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Gera probabilidades de previsão.

        Necessário para cálculo do Brier Score, conforme
        discutido na seção 'SLIs e Data SLOs em ML' do doc 04.

        Args:
            X: Features de entrada.

        Returns:
            Array (n_samples, 2) com probabilidades.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute train_model() primeiro.")
        return self.model.predict_proba(X)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calcula acurácia no conjunto fornecido.

        Usado pelo MonitoringPipeline para comparar com o SLO de
        acurácia mínima (Snippet 1 do DOCUMENTO_AULA_6.md).

        Args:
            X: Features.
            y: Target verdadeiro.

        Returns:
            Acurácia.

        Raises:
            RuntimeError: Se o modelo não foi treinado.
        """
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Execute train_model() primeiro.")
        return self.model.score(X, y)


# ======================================================================
# Execução direta: treina modelo e salva
# ======================================================================


if __name__ == "__main__":
    from src.data_preprocessing import DataPreprocessor
    from src.utils import save_model, setup_logging

    setup_logging()

    # Carregar dados
    preprocessor = DataPreprocessor(random_state=42)
    try:
        data = preprocessor.load_data()
    except FileNotFoundError:
        print("Dataset não encontrado. Gerando...")
        DataPreprocessor.generate_dataset()
        data = preprocessor.load_data()

    # Usar apenas dados de referência para treino
    df_ref, _ = preprocessor.split_reference_production(data)
    df_ref_clean = preprocessor.clean_data(df_ref)
    X, y = preprocessor.prepare_features(df_ref_clean)

    # Treinar
    trainer = CreditModelTrainer(model_type="gbm", random_state=42)
    trainer.train_model(X, y)

    # Validação cruzada
    cv_scores = trainer.cross_validate(X, y, cv=5)
    print(f"\nCV Scores: {cv_scores}")
    print(f"Média: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Salvar modelo
    output_dir = Path(__file__).resolve().parent.parent / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model(trainer.model, str(output_dir / "credit_model.joblib"))

    print("\nTreinamento concluído!")
