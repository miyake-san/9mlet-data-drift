# Aula 05 – Drift em Embeddings e Testes de Duas Amostras

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, você será capaz de:

1. **Compreender o que são embeddings** e como modelos como BERT (Devlin et al., 2019) geram representações vetoriais densas que capturam o significado semântico de textos.
2. **Diferenciar drift superficial de drift semântico** em dados não estruturados – sabendo que mudanças de formatação diferem de mudanças de significado (ex.: a palavra "máscara" pré e pós-pandemia).
3. **Aplicar testes de duas amostras** (KS e MMD) para comparar distribuições de embeddings de referência vs. produção e decidir estatisticamente se houve drift.
4. **Implementar a Discrepância de Máxima Média (MMD)** com kernel RBF usando `sklearn.metrics.pairwise.rbf_kernel` conforme Gretton et al. (2012).
5. **Utilizar classificadores adversários** (Lopez-Paz & Oquab, 2017) como detectores de drift – onde accuracy > 50% indica que as distribuições são distinguíveis.
6. **Projetar embeddings em 2D** com t-SNE e UMAP para visualização contínua de drift em pipelines de MLOps.
7. **Construir um pipeline de monitoramento** de drift em embeddings de texto/imagem com alertas baseados em limiares de MMD.

## 📖 Teoria-Chave

### Drift Superficial vs. Drift Semântico

Conforme discutido na seção "Saiba Mais" do Documento 04, nem toda mudança nos dados é igual (Feldhans et al., 2021):

- **Drift superficial**: alterações visíveis nos dados (formatação, estilo) que não modificam o significado. O modelo pode tolerá-las.
- **Drift semântico**: alterações de sentido ou contexto subjacente. Ex.: "máscara" pré-2020 (fantasias) vs. pós-2020 (proteção facial). Passa despercebido por métodos tradicionais mas se reflete nas coordenadas dos embeddings.

### Formalização do Drift

Há drift entre referência e produção quando a distribuição conjunta muda:

$$P_{\text{ref}}(X, y) \neq P_{\text{novo}}(X, y)$$

Três manifestações (Lu et al., 2019):
1. **Covariate shift**: $P(X)$ muda
2. **Concept drift**: $P(y|X)$ muda
3. **Prior probability shift**: $P(y)$ muda

### MMD – Maximum Mean Discrepancy

$$
\mathrm{MMD}^2(P_{\text{ref}}, P_{\text{novo}}) = \mathbb{E}_{x,x'}[k(x,x')] + \mathbb{E}_{y,y'}[k(y,y')] - 2\,\mathbb{E}_{x,y}[k(x,y)]
$$

Onde $k(u,v) = \exp(-\gamma \|u-v\|^2)$ é o kernel RBF. Se as distribuições forem idênticas, MMD ≈ 0 (Gretton et al., 2012).

### Teste de Kolmogorov-Smirnov

$$D_{n,m} = \sup_x \left| F_{\text{ref}}(x) - F_{\text{novo}}(x) \right|$$

Mede a máxima distância entre CDFs empíricas. É univariado; para embeddings multivariados, aplica-se por dimensão com correção de Bonferroni (Massey, 1951).

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| NumPy | ≥1.26.0 | Operações numéricas, álgebra linear |
| Pandas | ≥2.2.0 | Manipulação de DataFrames |
| SciPy | ≥1.12.0 | Teste KS (`ks_2samp`) |
| Scikit-learn | ≥1.5.0 | Kernel RBF, GBM adversário, cross-validation |
| Matplotlib | ≥3.9.0 | Visualizações e gráficos |
| Seaborn | ≥0.13.0 | Heatmaps e gráficos estatísticos |
| UMAP-learn | ≥0.5.5 | Redução de dimensionalidade (UMAP) |
| Joblib | ≥1.4.0 | Persistência de modelos |
| Pytest | ≥8.3.0 | Testes unitários |
| Transformers | ≥4.40.0 | Extração de embeddings BERT (opcional, vídeo 3) |
| Torch | ≥2.2.0 | Backend para transformers (opcional) |

## 📂 Estrutura de Arquivos

```
05_drift_em_embeddings_e_testes_de_duas_amostras/
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências específicas da aula
├── generate_dataset.py                 # Script CLI para gerar dataset sintético
├── notebooks/
│   ├── 01_exploracao.ipynb             # Exploração: embeddings, EDA, visualização
│   ├── 02_treinamento.ipynb            # Treinamento: MMD, KS, classificador adversário
│   └── 03_avaliacao.ipynb              # Avaliação: métricas, relatórios, pipeline
├── src/
│   ├── __init__.py                     # Exports públicos do pacote
│   ├── data_preprocessing.py           # DataPreprocessor: carga, limpeza, split ref/prod
│   ├── model.py                        # EmbeddingDriftDetector: MMD, KS, adversarial
│   ├── training.py                     # Treino do classificador adversário e calibração
│   ├── evaluation.py                   # Métricas de drift, visualizações UMAP/t-SNE
│   └── utils.py                        # I/O, seeds, geração de dataset sintético
├── data/
│   ├── raw/
│   │   └── dataset.csv                 # Dataset sintético de embeddings
│   ├── processed/                      # Dados processados (gerados em runtime)
│   └── README.md                       # Descrição dos dados
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py           # Testes do DataPreprocessor (5 classes)
│   ├── test_model.py                   # Testes do EmbeddingDriftDetector (5 classes)
│   └── test_evaluation.py              # Testes de métricas e relatórios (3 classes)
└── outputs/
    ├── models/                         # Detectores e classificadores salvos
    ├── figures/                        # Gráficos gerados (UMAP, heatmaps, timeseries)
    └── logs/                           # Logs de execução e métricas JSON
```

## 🚀 Como Executar

### Instalação de dependências

```bash
cd mlet_data-drift/05_drift_em_embeddings_e_testes_de_duas_amostras
uv pip install -r requirements.txt
```

### Geração do dataset sintético

```bash
python generate_dataset.py
# Ou com parâmetros customizados:
python generate_dataset.py --n-reference 5000 --n-stable 2500 --n-drift 2500 --seed 42
```

### Execução dos notebooks (sequencial)

```bash
jupyter lab notebooks/01_exploracao.ipynb
jupyter lab notebooks/02_treinamento.ipynb
jupyter lab notebooks/03_avaliacao.ipynb
```

### Execução via scripts Python

```python
from src.data_preprocessing import DataPreprocessor
from src.model import EmbeddingDriftDetector
from src.evaluation import calculate_drift_metrics

# Carregar e preparar dados
preprocessor = DataPreprocessor(embedding_dim=64)
preprocessor.load_data("data/raw/dataset.csv")
preprocessor.clean_data()
emb_ref, emb_prod, df_ref, df_prod = preprocessor.split_data()

# Detectar drift com MMD
detector = EmbeddingDriftDetector(method="mmd", gamma=1.0)
detector.fit(emb_ref)
result = detector.predict(emb_prod)
print(f"MMD² = {result['score']:.4f}, Drift = {result['drift_detected']}")
```

## 📊 Datasets

Dataset sintético simulando embeddings de postagens em redes sociais (caso TrendCast) com drift semântico. Detalhes completos em [data/README.md](data/README.md).

- **Instâncias totais:** 10.000 (5.000 ref + 2.500 estável + 2.500 com drift)
- **Embedding dim:** 64
- **Categorias:** 5 (esportes, clima, entretenimento, tecnologia, política)
- **Tipo de drift:** Semântico (deslocamento de centros) + proporcional (política domina 50%)

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes por módulo
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v

# Com cobertura
pytest tests/ -v --cov=src --cov-report=term-missing
```

## 📚 Referências

1. **Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. & Smola, A.** (2012). A kernel two-sample test. *JMLR*, 13(Mar), 723–773.
2. **Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K.** (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, pp. 4171–4186.
3. **Massey Jr., F.J.** (1951). The Kolmogorov-Smirnov Test for Goodness of Fit. *JASA*, 46(253), 68–78.
4. **Lopez-Paz, D. & Oquab, M.** (2017). Revisiting classifier two-sample tests. *ICLR*.
5. **Feldhans, R. et al.** (2021). Drift Detection in Text Data with Document Embeddings. *IDEAL 2021*, LNCS 13113, pp. 107–118.
6. **Mikolov, T. et al.** (2013). Efficient estimation of word representations in vector space. *ICLR* (arXiv:1301.3781).
7. **Lu, J. et al.** (2019). Learning under concept drift: A review. *IEEE TKDE*, 31(12), 2346–2363.
8. **Gama, J. et al.** (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 44.
9. **Greco, S. et al.** (2024). Unsupervised concept drift detection from deep learning representations in real-time. *arXiv:2406.17813*.
10. **Vaswani, A. et al.** (2017). Attention Is All You Need. *NeurIPS*, 30, 5998–6008.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Duração |
|---|---|---|
| 1 | Drift em Embeddings e Testes de Duas Amostras | 15 min |
| 2 | Drift Semântico e Estratégias de Detecção | 18 min |
| 3 | Extração de Embeddings e Detecção com BERT e MMD | 20 min |
| 4 | Pipeline de Monitoramento de Embeddings em Produção | 18 min |

### Mapeamento Vídeos → Código

| Vídeo | Notebook | Módulos `src/` |
|---|---|---|
| 1 | `01_exploracao.ipynb` (conceitos, EDA) | `data_preprocessing.py`, `utils.py` |
| 2 | `02_treinamento.ipynb` (MMD, adversarial) | `model.py`, `training.py` |
| 3 | `02_treinamento.ipynb` + `03_avaliacao.ipynb` (BERT, MMD) | `model.py`, `evaluation.py` |
| 4 | `03_avaliacao.ipynb` (pipeline, alertas) | `evaluation.py`, `utils.py` |

## 🔗 Links Úteis

- [BERT – Hugging Face](https://huggingface.co/bert-base-uncased)
- [Gretton et al., 2012 – MMD Paper](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
- [Scikit-learn – RBF Kernel](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [SciPy – ks_2samp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)
- [Evidently AI – Embedding Drift](https://www.evidentlyai.com/)
- [Alibi Detect – Drift Detection](https://docs.seldon.io/projects/alibi-detect/en/stable/)
- [DriftLens – Greco et al.](https://arxiv.org/abs/2406.17813)
