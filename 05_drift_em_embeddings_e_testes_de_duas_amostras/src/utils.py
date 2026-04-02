"""
Módulo de utilitários para a Aula 5: Drift em Embeddings e Testes de Duas Amostras.

Inclui funções para persistência de modelos, configuração de reproduzibilidade
e geração de datasets sintéticos para experimentação com detecção de drift.

Referências:
    Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers. NAACL.
    Gretton, A. et al. (2012). A kernel two-sample test. JMLR, 13, 723-773.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configura logging para o módulo.

    Args:
        log_dir: Diretório para arquivos de log.
        level: Nível de logging.

    Returns:
        Logger configurado.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"drift_detection_{timestamp}.log"

    logger = logging.getLogger("drift_detection")
    logger.setLevel(level)

    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Reproduzibilidade
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Define seed global para reproduzibilidade.

    Conforme boas práticas de experimentação em ML, fixar a seed garante
    que os resultados dos testes estatísticos e da geração de dados
    sintéticos sejam reproduzíveis entre execuções.

    Args:
        seed: Valor da seed (default: 42).
    """
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Persistência de Modelos
# ---------------------------------------------------------------------------

def save_model(model: Any, filepath: str) -> None:
    """
    Salva modelo ou detector em disco usando joblib.

    Args:
        model: Objeto do modelo/detector a salvar.
        filepath: Caminho de destino (ex: 'outputs/models/detector.joblib').
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(filepath: str) -> Any:
    """
    Carrega modelo ou detector salvo com joblib.

    Args:
        filepath: Caminho do arquivo salvo.

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
# Persistência de Métricas
# ---------------------------------------------------------------------------

def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Salva métricas de drift em formato JSON.

    Args:
        metrics: Dicionário com métricas calculadas.
        filepath: Caminho de destino (ex: 'outputs/logs/metrics.json').
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Converte tipos numpy para tipos nativos Python
    serializable = _make_serializable(metrics)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def load_config(filepath: str) -> Dict[str, Any]:
    """
    Carrega configuração JSON.

    Args:
        filepath: Caminho do arquivo JSON de configuração.

    Returns:
        Dicionário com a configuração.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {filepath}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_serializable(obj: Any) -> Any:
    """Converte objetos numpy/pandas para tipos nativos Python."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# Geração de Dataset Sintético
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_reference: int = 5000,
    n_production_stable: int = 2500,
    n_production_drift: int = 2500,
    embedding_dim: int = 64,
    drift_magnitude: float = 1.5,
    seed: int = 42,
    output_path: Optional[str] = "data/raw/dataset.csv",
) -> pd.DataFrame:
    """
    Gera dataset sintético de embeddings simulando drift em dados textuais.

    Implementa o cenário descrito na seção 'O que vem por aí' do documento da
    Aula 5 (caso TrendCast): um sistema de ML que monitora tópicos de redes
    sociais e enfrenta mudanças na distribuição de dados quando eventos mundiais
    alteram os padrões de postagens.

    Conforme discutido na seção 'Saiba Mais', o drift é simulado como mudança
    na distribuição conjunta P(X) dos embeddings, onde:
    - P_ref(X) ≠ P_novo(X) indica drift nos dados de entrada (Gama et al., 2014)
    - Tanto a posição dos clusters (drift semântico) quanto as proporções das
      categorias (drift de proporção) são alteradas

    Args:
        n_reference: Número de amostras de referência.
        n_production_stable: Número de amostras de produção estável (sem drift).
        n_production_drift: Número de amostras de produção com drift.
        embedding_dim: Dimensionalidade dos embeddings sintéticos.
        drift_magnitude: Magnitude do deslocamento para simular drift.
        seed: Seed para reproduzibilidade.
        output_path: Caminho para salvar o CSV (None para não salvar).

    Returns:
        DataFrame com colunas: sample_id, period, category, timestamp, emb_0..emb_N.

    Referência:
        Gama, J. et al. (2014). A survey on concept drift adaptation.
        ACM Computing Surveys, 46(4), 44.
    """
    rng = np.random.RandomState(seed)

    categories = ["esportes", "clima", "entretenimento", "tecnologia", "politica"]
    n_cats = len(categories)

    # Centros dos clusters por categoria no espaço de embedding
    # Cada categoria ocupa uma região distinta do espaço vetorial,
    # análogo a como embeddings BERT posicionam textos semanticamente
    # similares próximos (Devlin et al., 2019)
    centers = {cat: rng.randn(embedding_dim) * 0.5 for cat in categories}

    all_embeddings = []
    all_periods = []
    all_categories = []

    # ---- Dados de referência: distribuição balanceada ----
    per_cat_ref = n_reference // n_cats
    for cat in categories:
        embs = rng.multivariate_normal(
            centers[cat], np.eye(embedding_dim) * 0.3, per_cat_ref
        )
        all_embeddings.append(embs)
        all_periods.extend(["reference"] * per_cat_ref)
        all_categories.extend([cat] * per_cat_ref)

    # ---- Produção estável: mesma distribuição da referência ----
    per_cat_stable = n_production_stable // n_cats
    for cat in categories:
        embs = rng.multivariate_normal(
            centers[cat], np.eye(embedding_dim) * 0.3, per_cat_stable
        )
        all_embeddings.append(embs)
        all_periods.extend(["production_stable"] * per_cat_stable)
        all_categories.extend([cat] * per_cat_stable)

    # ---- Produção com drift: centros deslocados e proporções alteradas ----
    # Simula drift semântico: a categoria "política" e "tecnologia" mudam
    # de significado (seus centros no espaço de embedding se deslocam),
    # análogo ao exemplo da palavra "máscara" pré e pós-pandemia
    shifted_centers = {k: v.copy() for k, v in centers.items()}
    shifted_centers["politica"] = centers["politica"] + drift_magnitude
    shifted_centers["tecnologia"] = centers["tecnologia"] + drift_magnitude * 0.5

    # Simula drift de proporção: política domina as postagens (50%)
    drift_proportions = {
        "esportes": 0.10,
        "clima": 0.10,
        "entretenimento": 0.10,
        "tecnologia": 0.20,
        "politica": 0.50,
    }

    for cat, prop in drift_proportions.items():
        n_cat = int(n_production_drift * prop)
        embs = rng.multivariate_normal(
            shifted_centers[cat],
            np.eye(embedding_dim) * 0.35,  # Variância ligeiramente maior
            n_cat,
        )
        all_embeddings.append(embs)
        all_periods.extend(["production_drift"] * n_cat)
        all_categories.extend([cat] * n_cat)

    # ---- Montar DataFrame ----
    embeddings_array = np.vstack(all_embeddings)
    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]

    df = pd.DataFrame(embeddings_array, columns=emb_cols)
    df.insert(0, "sample_id", range(len(df)))
    df.insert(1, "period", all_periods)
    df.insert(2, "category", all_categories)

    # Timestamps simulados: referência (semanas 1-4), estável (5-6), drift (7-8)
    timestamps = []
    for period in all_periods:
        if period == "reference":
            day_offset = rng.randint(0, 28)
            timestamps.append(pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(day_offset)))
        elif period == "production_stable":
            day_offset = rng.randint(0, 14)
            timestamps.append(pd.Timestamp("2024-02-01") + pd.Timedelta(days=int(day_offset)))
        else:
            day_offset = rng.randint(0, 14)
            timestamps.append(pd.Timestamp("2024-02-15") + pd.Timedelta(days=int(day_offset)))

    df.insert(3, "timestamp", timestamps)

    # Salvar em CSV
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    return df


def generate_text_samples(n_per_category: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Gera amostras de textos simulados para demonstração de extração de embeddings.

    Útil para os notebooks que demonstram extração de embeddings com BERT
    conforme o Snippet 1 da seção Hands On do documento da Aula 5.

    Args:
        n_per_category: Número de textos por categoria.
        seed: Seed para reproduzibilidade.

    Returns:
        DataFrame com colunas: text_id, text, category, period.
    """
    rng = np.random.RandomState(seed)

    reference_templates = {
        "esportes": [
            "O campeonato de futebol teve resultados surpreendentes nesta rodada.",
            "O time local venceu a partida por 3 a 1 no estádio municipal.",
            "Atleta brasileiro conquista medalha de ouro nos jogos internacionais.",
            "A temporada de basquete promete ser a mais disputada dos últimos anos.",
            "Novo recorde mundial foi estabelecido na maratona internacional.",
        ],
        "clima": [
            "A previsão indica sol com poucas nuvens para o fim de semana.",
            "Temperaturas devem cair significativamente a partir de quinta-feira.",
            "Alerta de chuvas fortes emitido para a região metropolitana.",
            "O verão deste ano registrou temperaturas acima da média histórica.",
            "Frente fria chega ao sudeste trazendo ventos de até 60km/h.",
        ],
        "entretenimento": [
            "Novo filme de ficção científica estreia com bilheteria recorde.",
            "Festival de música reúne artistas nacionais e internacionais.",
            "A série mais assistida da plataforma ganha nova temporada.",
            "Premiação de cinema destaca produções independentes neste ano.",
            "Shows ao vivo voltam a atrair grandes públicos após período de restrições.",
        ],
        "tecnologia": [
            "Nova atualização do sistema operacional promete melhor desempenho.",
            "Startup brasileira lança aplicativo inovador de produtividade.",
            "Avanços em inteligência artificial transformam o setor de saúde.",
            "Lançamento de smartphone traz câmera com resolução sem precedentes.",
            "Empresa de software anuncia integração com serviços de nuvem.",
        ],
        "politica": [
            "Discussões sobre o orçamento dominam a sessão parlamentar.",
            "Nova proposta de lei visa regulamentar o uso de tecnologia.",
            "Eleições municipais registram alta participação dos eleitores.",
            "Acordo internacional sobre mudanças climáticas é firmado.",
            "Debates sobre educação pública ganham destaque nas redes sociais.",
        ],
    }

    # Textos de produção com drift (temas politizados dominam)
    drift_templates = {
        "politica": [
            "Crise política gera protestos em diversas capitais do país.",
            "Resultados eleitorais surpreendem analistas e pesquisas de opinião.",
            "Debates políticos acalorados tomam conta das redes sociais.",
            "Nova política econômica gera controvérsias entre especialistas.",
            "Tensões geopolíticas globais afetam mercados financeiros locais.",
        ],
        "tecnologia": [
            "Regulamentação de IA se torna pauta política prioritária.",
            "Governo anuncia investimentos em infraestrutura tecnológica.",
            "Política de dados pessoais é tema de audiência pública.",
            "Censura digital é debatida em comissões parlamentares.",
            "Empresa de tecnologia é investigada por práticas monopolistas.",
        ],
    }

    rows = []
    text_id = 0

    for cat, templates in reference_templates.items():
        for _ in range(n_per_category):
            text = rng.choice(templates)
            rows.append({
                "text_id": text_id,
                "text": text,
                "category": cat,
                "period": "reference",
            })
            text_id += 1

    for cat, templates in drift_templates.items():
        for _ in range(n_per_category):
            text = rng.choice(templates)
            rows.append({
                "text_id": text_id,
                "text": text,
                "category": cat,
                "period": "production_drift",
            })
            text_id += 1

    return pd.DataFrame(rows)
