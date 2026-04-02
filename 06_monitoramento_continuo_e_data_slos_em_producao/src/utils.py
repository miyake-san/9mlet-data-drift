"""
Módulo de utilitários para monitoramento contínuo de ML.

Implementa funções auxiliares de persistência de modelos, métricas,
configuração de logging e carregamento de configurações.

Referências:
    Sculley, D. et al. (2015). Hidden Technical Debt in Machine
    Learning Systems. NIPS 2015.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def save_model(model: Any, filepath: str) -> None:
    """Salva modelo treinado em disco via joblib.

    Args:
        model: Modelo ou pipeline sklearn treinado.
        filepath: Caminho de saída (.joblib).
    """
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(output_path))
    logger.info(f"Modelo salvo: {output_path}")


def load_model(filepath: str) -> Any:
    """Carrega modelo salvo do disco.

    Args:
        filepath: Caminho do modelo (.joblib).

    Returns:
        Modelo ou pipeline carregado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")

    model = joblib.load(str(path))
    logger.info(f"Modelo carregado: {path}")
    return model


def save_metrics(
    metrics: Dict[str, Any],
    filepath: Optional[str] = None,
) -> None:
    """Salva métricas de avaliação em JSON.

    Args:
        metrics: Dicionário com métricas.
        filepath: Caminho de saída. Se None, usa logs/metrics_<timestamp>.json.
    """
    if filepath is None:
        logs_dir = Path(__file__).resolve().parent.parent / "outputs" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = str(logs_dir / f"metrics_{timestamp}.json")

    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Converter numpy types para tipos nativos Python
    serializable = _make_serializable(metrics)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    logger.info(f"Métricas salvas: {output_path}")


def load_metrics(filepath: str) -> Dict[str, Any]:
    """Carrega métricas salvas de um arquivo JSON.

    Args:
        filepath: Caminho do arquivo JSON.

    Returns:
        Dicionário com métricas.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return metrics


def load_config(filepath: Optional[str] = None) -> Dict[str, Any]:
    """Carrega configuração de monitoramento.

    Args:
        filepath: Caminho do arquivo de configuração JSON.
                  Se None, retorna configuração padrão.

    Returns:
        Dicionário com configuração.
    """
    if filepath is not None:
        path = Path(filepath)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    # Configuração padrão baseada nos SLOs do DOCUMENTO_AULA_6.md
    return {
        "slo": {
            "accuracy_min": 0.85,
            "drift_p_value": 0.05,
            "missing_rate_max": 0.01,
            "psi_threshold": 0.25,
        },
        "features_to_monitor": [
            "idade", "renda_mensal", "score_credito",
            "tempo_emprego", "valor_emprestimo",
        ],
        "model_type": "gbm",
        "random_state": 42,
    }


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configura logging para o projeto.

    Args:
        level: Nível de logging.
        log_file: Caminho para arquivo de log. Se None, apenas console.
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _make_serializable(obj: Any) -> Any:
    """Converte objetos numpy/pandas para tipos Python nativos.

    Args:
        obj: Objeto a converter.

    Returns:
        Objeto serializado em tipos nativos.
    """
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
