# Datasets – Aula 05: Drift em Embeddings e Testes de Duas Amostras

## Dataset Principal

### Descrição

Dataset **sintético** que simula o cenário TrendCast descrito no Documento 04:
um sistema de ML que monitora tópicos populares em redes sociais e enfrenta
**drift semântico** quando eventos mundiais alteram os padrões de postagens.

Os dados representam **embeddings pré-computados** de 64 dimensões para textos
de cinco categorias (esportes, clima, entretenimento, tecnologia, política),
distribuídos em três períodos temporais:

- **Referência** (semanas 1–4): distribuição balanceada entre categorias.
- **Produção estável** (semanas 5–6): mesma distribuição da referência (sem drift).
- **Produção com drift** (semanas 7–8): drift semântico e proporcional introduzido.

O drift é simulado conforme a seção "Saiba Mais" do Documento 04:
- **Drift semântico**: os centros dos clusters de "política" e "tecnologia" são
  deslocados no espaço de embedding (análogo ao exemplo da palavra "máscara"
  pré e pós-pandemia – Devlin et al., 2019; Feldhans et al., 2021).
- **Drift proporcional**: a categoria "política" passa a dominar 50% das postagens,
  simulando um evento político de grande impacto.

### Estrutura

| Coluna | Tipo | Descrição |
|---|---|---|
| sample_id | int | Identificador único da amostra |
| period | str | Período: `reference`, `production_stable`, `production_drift` |
| category | str | Categoria do texto: esportes, clima, entretenimento, tecnologia, política |
| timestamp | datetime | Data simulada da postagem |
| emb_0 ... emb_63 | float64 | Componentes do embedding de 64 dimensões |

### Estatísticas

- **Instâncias totais:** 10.000
  - Referência: 5.000 (1.000 por categoria)
  - Produção estável: 2.500 (500 por categoria)
  - Produção com drift: 2.500 (proporções alteradas)
- **Features (embedding):** 64 dimensões
- **Categorias:** 5
- **Tipo de drift:** Semântico (deslocamento de centros, magnitude=1.5) + proporcional

### Fonte

Sintético – gerado programaticamente com seed fixa (42) para reprodutibilidade.

### Reprodução

```bash
# Via script CLI
python generate_dataset.py

# Com parâmetros customizados
python generate_dataset.py --output data/raw/dataset.csv --seed 42 \
    --n-reference 5000 --n-stable 2500 --n-drift 2500 \
    --embedding-dim 64 --drift-magnitude 1.5
```

Ou via código Python:

```python
from src.utils import generate_synthetic_dataset

df = generate_synthetic_dataset(
    n_reference=5000,
    n_production_stable=2500,
    n_production_drift=2500,
    embedding_dim=64,
    drift_magnitude=1.5,
    seed=42,
    output_path="data/raw/dataset.csv",
)
```

## Dados Processados

A pasta `processed/` armazena dados gerados durante a execução dos notebooks:

- Embeddings normalizados (StandardScaler)
- Projeções UMAP/t-SNE em 2D
- Arrays de referência e produção separados

Estes arquivos são gerados automaticamente e podem ser recriados executando
os notebooks na ordem sequencial (01 → 02 → 03).
