# Aula 04 – Métricas Avançadas para Detecção de Drift

## 🎯 Objetivos de Aprendizagem

Ao final desta aula, você será capaz de:

1. **Compreender os limites** dos testes univariados (KS, PSI) em dados de alta dimensionalidade e identificar cenários onde eles falham.
2. **Aplicar métricas multivariadas** de detecção de drift, incluindo MMD (Maximum Mean Discrepancy), distância de Wasserstein e Energy Distance.
3. **Implementar do zero** o cálculo de PSI e MMD com kernel RBF em Python usando NumPy/SciPy.
4. **Detectar drifts multivariados sutis**, como inversão de correlação entre variáveis, que escapam de testes univariados.
5. **Utilizar bibliotecas de produção** (Alibi Detect) para monitoramento de drift em pipelines de MLOps.
6. **Comparar sensibilidades** entre PSI, MMD, Wasserstein e Energy Distance em cenários controlados.

## 📖 Teoria-Chave

### Definição Formal de Drift

Há *concept drift* entre tempos $t_0$ e $t_1$ se $\exists X, y: P_{t_0}(X,y) \neq P_{t_1}(X,y)$, onde $P_t(X,y)$ é a distribuição conjunta das variáveis preditoras $X$ e da variável-alvo $y$ no tempo $t$ (Gama et al., 2014).

### Limitações de Métricas Univariadas

Testes como KS e PSI analisam variáveis individualmente. Quando a alteração ocorre na **relação entre variáveis** (drift multivariado sutil), as distribuições marginais permanecem similares e esses métodos falham em detectar o drift (Documento 04, Seção "Limites de Métricas Simples").

### Máxima Discrepância de Médias (MMD)

$$
\text{MMD}^2(P,Q) = E_{x,x' \sim P}[k(x,x')] + E_{y,y' \sim Q}[k(y,y')] - 2\,E_{x \sim P,\,y \sim Q}[k(x,y)]
$$

Onde $k$ é uma função kernel (tipicamente RBF gaussiano). A MMD é zero se e somente se $P = Q$ (Gretton et al., 2012).

### PSI (Population Stability Index)

$$
\text{PSI} = \sum_{i}(p_i - q_i)\ln\frac{p_i}{q_i}
$$

Valores de PSI > 0.25 sugerem mudança significativa na distribuição. Limitação: análise univariada apenas.

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| NumPy | ≥1.26.0 | Cálculos numéricos, PSI, MMD |
| SciPy | ≥1.12.0 | Distância Wasserstein, Energy Distance |
| Pandas | ≥2.2.0 | Manipulação de dados |
| Scikit-learn | ≥1.5.0 | Kernel RBF, métricas |
| Matplotlib | ≥3.9.0 | Visualizações |
| Seaborn | ≥0.13.0 | Gráficos estatísticos |
| Alibi-Detect | ≥0.12.0 | Drift multivariado em produção |
| Pytest | ≥8.3.0 | Testes unitários |

## 📂 Estrutura de Arquivos

```
04_metricas_avancadas_para_deteccao_de_drift/
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências específicas da aula
├── notebooks/
│   ├── 01_exploracao.ipynb             # Exploração: limites de métricas univariadas
│   ├── 02_treinamento.ipynb            # Implementação: MMD, Wasserstein, Energy Distance
│   └── 03_avaliacao.ipynb              # Avaliação: comparação de métricas e produção
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Geração e pré-processamento de dados sintéticos
│   ├── model.py                        # Detectores de drift (PSI, MMD, Wasserstein, Energy)
│   ├── training.py                     # Calibração de limiares e busca de hiperparâmetros
│   ├── evaluation.py                   # Métricas de avaliação e visualização
│   └── utils.py                        # Utilitários (I/O, configuração, logging)
├── data/
│   ├── raw/
│   │   └── dataset.csv                 # Dataset sintético de referência
│   ├── processed/                      # Dados processados (gerados em runtime)
│   └── README.md                       # Descrição dos dados
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py           # Testes de geração e processamento
│   ├── test_model.py                   # Testes dos detectores de drift
│   └── test_evaluation.py             # Testes de métricas e avaliação
└── outputs/
    ├── models/                         # Resultados de calibração
    ├── figures/                        # Gráficos gerados
    └── logs/                           # Logs de execução
```

## 🚀 Como Executar

### Instalação de dependências

```bash
cd 04_metricas_avancadas_para_deteccao_de_drift
pip install -r requirements.txt
```

### Execução dos notebooks (sequencial)

```bash
jupyter lab notebooks/01_exploracao.ipynb
jupyter lab notebooks/02_treinamento.ipynb
jupyter lab notebooks/03_avaliacao.ipynb
```

### Execução via scripts Python

```bash
# Gerar dataset sintético
python -c "from src.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.generate_and_save()"

# Executar detecção de drift
python -c "from src.model import DriftDetector; print('Módulos carregados com sucesso')"
```

## 📊 Datasets

Dataset sintético simulando cenário de fintech com drift multivariado sutil. Detalhes completos em [data/README.md](data/README.md).

- **Instâncias (referência):** 5.000
- **Instâncias (atual):** 5.000
- **Features:** 6 (idade, renda, divida, score_credito, tempo_emprego, num_parcelas)
- **Tipo de drift:** Inversão de correlação entre variáveis

## 🧪 Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar testes específicos
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
pytest tests/test_evaluation.py -v
```

## 📚 Referências

1. **Gretton, A. et al.** (2012). A Kernel Two-Sample Test. *JMLR*, 13, 723–773.
2. **Gama, J. et al.** (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), art. 44.
3. **Sugiyama, M. et al.** (2013). Density-difference estimation. *Neural Computation*, 25(10), 2734–2775.
4. **Székely, G. J.; Rizzo, M. L.** (2013). Energy statistics: A class of statistics based on distances. *JSPI*, 143(8), 1249–1272.
5. **Arjovsky, M.; Chintala, S.; Bottou, L.** (2017). Wasserstein GAN. *ICML*, PMLR v.70, 214–223.
6. **Feldhans, R. et al.** (2021). Drift Detection in Text Data with Document Embeddings. *IDEAL 2021*, LNCS 12873.
7. **Müller, R. et al.** (2024). Open-Source Drift Detection Tools in Action. *arXiv:2404.18673*.
8. **Bellman, R.** (1961). Adaptive Control Processes: A Guided Tour. Princeton University Press.

## 🎥 Vídeos Relacionados

| Vídeo | Título | Duração |
|---|---|---|
| 1 | Métricas Avançadas para Detecção de Drift | 15 min |
| 2 | MMD, Wasserstein e Testes Multivariados | 18 min |
| 3 | PSI e MMD na Prática com Python | 20 min |
| 4 | Detecção de Drift Multivariado em Produção | 18 min |

## 🔗 Links Úteis

- [Alibi Detect – Documentação](https://docs.seldon.io/projects/alibi-detect/en/stable/)
- [Gretton et al., 2012 – MMD Paper](http://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf)
- [Concept Drift – Fast Forward Labs](https://concept-drift.fastforwardlabs.com/)
- [SciPy – Wasserstein Distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html)
- [Evidently AI – Drift Detection](https://www.evidentlyai.com/)
- [NannyML – ML Monitoring](https://www.nannyml.com/)
