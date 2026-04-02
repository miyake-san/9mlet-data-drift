"""
Utilitários gerais para a Aula 7.

Funções auxiliares para I/O, configuração, logging e serialização
de resultados de monitoramento de drift.

Referências:
    Sculley, D. et al. (2015). Hidden Technical Debt in ML Systems.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def save_model(model: Any, filepath: str) -> None:
    """
    Salva modelo treinado em disco via pickle.

    Args:
        model: Objeto do modelo a salvar.
        filepath: Caminho do arquivo de saída.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """
    Carrega modelo salvo do disco.

    Args:
        filepath: Caminho do arquivo.

    Returns:
        Objeto do modelo.

    Raises:
        FileNotFoundError: Se arquivo não existir.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)  # noqa: S301


def save_metrics(
    metrics: Dict[str, Any],
    filepath: str,
    append: bool = False,
) -> None:
    """
    Salva métricas de drift em arquivo JSON.

    Conforme seção 'Saiba Mais' — observabilidade: registrar
    séries temporais de métricas com contexto (versão do modelo,
    janela, amostragem) para rastreabilidade (Sculley et al., 2015).

    Args:
        metrics: Dicionário de métricas.
        filepath: Caminho do arquivo JSON.
        append: Se True, adiciona a um array existente.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Converter tipos numpy para JSON serializáveis
    clean_metrics = _make_json_serializable(metrics)
    clean_metrics["saved_at"] = datetime.now().isoformat()

    if append and os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(clean_metrics)
        else:
            existing = [existing, clean_metrics]
        data = existing
    else:
        data = clean_metrics

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Carrega configuração de monitoramento de um arquivo JSON.

    Args:
        filepath: Caminho do arquivo de configuração.

    Returns:
        Dicionário de configuração.

    Raises:
        FileNotFoundError: Se arquivo não existir.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuração não encontrada: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configura logger para o pipeline de monitoramento.

    Args:
        log_dir: Diretório para arquivos de log.
        level: Nível de logging.

    Returns:
        Logger configurado.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("drift_monitor")
    logger.setLevel(level)

    if not logger.handlers:
        # Handler de arquivo
        fh = logging.FileHandler(
            os.path.join(log_dir, f"drift_{datetime.now():%Y%m%d_%H%M%S}.log"),
            encoding="utf-8",
        )
        fh.setLevel(level)

        # Handler de console
        ch = logging.StreamHandler()
        ch.setLevel(level)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def emit_alert(
    message: str,
    level: str = "WARNING",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Emite alerta de drift conforme Snippet 4 do Hands On.

    Em produção, isso corresponderia ao envio para sistema
    de alertas (PagerDuty, Slack, etc.). Aqui, registra
    no logger e retorna dicionário do alerta.

    Args:
        message: Mensagem do alerta.
        level: Nível do alerta (WARNING, CRITICAL).
        logger: Logger opcional.

    Returns:
        Dicionário com detalhes do alerta.
    """
    alert = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
    }

    if logger:
        log_fn = logger.warning if level == "WARNING" else logger.critical
        log_fn(f"[DRIFT ALERT] {message}")
    else:
        print(f"[{level}] DRIFT ALERT: {message}")

    return alert


def _make_json_serializable(obj: Any) -> Any:
    """Converte tipos numpy/pandas para tipos nativos Python."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def format_drift_report(
    drift_results: List[Dict[str, Any]],
) -> str:
    """
    Formata resultados de drift em relatório textual.

    Args:
        drift_results: Lista de resultados de janelas.

    Returns:
        String formatada com resumo do relatório.
    """
    lines = ["=" * 60, "RELATÓRIO DE MONITORAMENTO DE DRIFT", "=" * 60]

    for r in drift_results:
        months = r.get("window_months", [])
        alert = r.get("alert_level", "?")
        score = r.get("aggregate_score", {})

        lines.append(f"\nJanela: Mês(es) {months}")
        lines.append(f"  Amostras: {r.get('n_samples', '?')}")
        lines.append(f"  PSI médio: {score.get('mean_psi', 0):.4f}")
        lines.append(f"  PSI máximo: {score.get('max_psi', 0):.4f}")
        lines.append(f"  Features com drift: {score.get('n_features_with_drift', 0)}/{score.get('n_features_total', 0)}")
        lines.append(f"  Alerta: {alert}")
        if r.get("should_retrain"):
            lines.append("  >>> TRIGGER DE RETRAINING ATIVADO <<<")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
