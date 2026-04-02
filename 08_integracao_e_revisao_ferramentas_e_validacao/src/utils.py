"""
Módulo de utilitários para o pipeline de monitoramento de fraude.

Fornece funções auxiliares para serialização de modelos, armazenamento
de métricas, configuração de reproduzibilidade e logging, conforme
as boas práticas de MLOps discutidas no Documento 04 da Aula 8.

Referências:
    Sculley, D. et al. (NIPS 2015). Hidden technical debt in
        machine learning systems.
    Polyzotis, N. et al. (2019). Data lifecycle challenges in
        production machine learning. ACM SIGMOD.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Define seed global para reproduzibilidade dos experimentos.

    Conforme boas práticas de MLOps (Sculley et al., 2015), a
    reproduzibilidade é essencial para auditoria e governança
    de modelos em produção.

    Args:
        seed: Valor inteiro da semente.
    """
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Serialização de modelos
# ---------------------------------------------------------------------------

def save_model(model: Any, filepath: str) -> str:
    """
    Salva modelo treinado em disco usando joblib.

    Implementa persistência de artefatos conforme recomendado em
    pipelines de MLOps (Documento 04, seção 'Pipeline Integrado').

    Args:
        model: Objeto do modelo treinado.
        filepath: Caminho para salvar (extensão .pkl recomendada).

    Returns:
        Caminho absoluto do arquivo salvo.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path.resolve())


def load_model(filepath: str) -> Any:
    """
    Carrega modelo previamente salvo com joblib.

    Args:
        filepath: Caminho do arquivo .pkl.

    Returns:
        Objeto do modelo carregado.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Métricas e relatórios
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict[str, Any], filepath: str) -> str:
    """
    Salva dicionário de métricas em arquivo JSON.

    Adiciona timestamp para rastreabilidade, conforme práticas de
    governança e auditoria discutidas no Documento 04 (seção
    'Validação, Governança e Tendências Futuras').

    Args:
        metrics: Dicionário com métricas (chave → valor).
        filepath: Caminho para o arquivo JSON de saída.

    Returns:
        Caminho absoluto do arquivo salvo.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": _make_json_serializable(metrics),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return str(path.resolve())


def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Carrega métricas de um arquivo JSON.

    Args:
        filepath: Caminho do arquivo JSON.

    Returns:
        Dicionário com métricas e timestamp.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de métricas não encontrado: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Configuração e logging
# ---------------------------------------------------------------------------

def load_config(filepath: str) -> Dict[str, Any]:
    """
    Carrega configuração de um arquivo JSON.

    Args:
        filepath: Caminho do arquivo de configuração.

    Returns:
        Dicionário com configurações.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrado: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Configura logger para o pipeline de monitoramento.

    Conforme Sculley et al. (2015), o logging adequado é essencial
    para rastrear comportamento de sistemas de ML em produção.

    Args:
        log_dir: Diretório para salvar logs.
        level: Nível de logging (padrão: INFO).
        name: Nome do logger. Se None, usa 'aula8_pipeline'.

    Returns:
        Logger configurado.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger_name = name or "aula8_pipeline"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Evitar handlers duplicados
    if not logger.handlers:
        # Handler para arquivo
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_path / f"{logger_name}_{ts}.log")
        fh.setLevel(level)

        # Handler para console
        ch = logging.StreamHandler()
        ch.setLevel(level)

        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _make_json_serializable(obj: Any) -> Any:
    """Converte tipos numpy para tipos Python nativos."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
