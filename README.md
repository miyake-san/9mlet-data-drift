# DATA DRIFT

Repositorio academico para a disciplina de pos-graduacao em engenharia de machine learning focada em deteccao, monitoramento e mitigacao de drift em sistemas de ML em producao.

## 📋 Sobre

Data drift e concept drift estao entre os principais fatores de degradacao de modelos de machine learning apos o deploy. Quando a distribuicao dos dados de entrada, das classes ou da relacao entre variaveis muda ao longo do tempo, a performance do modelo pode cair de forma gradual ou abrupta, gerando perdas financeiras, aumento de risco operacional e perda de confianca do negocio.

Esta disciplina foi estruturada para cobrir o ciclo completo do problema. O percurso vai dos fundamentos teoricos de drift e testes estatisticos classicos ate metricas multivariadas avancadas, drift em embeddings, monitoramento continuo, Data SLOs, estrategias de mitigacao, arquiteturas escalaveis e integracao com ferramentas open source de observabilidade de modelos.

O repositorio organiza esse conteudo em uma estrutura reutilizavel para ensino, estudo e geracao de codigo. Alem das oito pastas de aula, ele inclui documentacao de arquitetura, ADRs, diagramas Mermaid, casos de negocio e um guia de contribuicao que conectam teoria, implementacao e governanca tecnica.

A stack proposta privilegia ferramentas open source e aderentes ao material da disciplina: Python, bibliotecas cientificas para analise estatistica, componentes para aprendizado em fluxo, frameworks para embeddings e ferramentas especializadas para monitoramento e validacao de dados em producao.

## 🎯 Objetivos de Aprendizagem

- Compreender as diferencas entre data drift, prior shift e concept drift em sistemas de ML.
- Aplicar testes estatisticos e metricas de distancia para detectar mudancas em dados tabulares.
- Identificar drifts sutis e multivariados com metricas avancadas e testes baseados em modelos.
- Monitorar drift em embeddings e dados nao estruturados usando representacoes vetoriais.
- Definir SLIs e Data SLOs para transformar sinais tecnicos em objetivos operacionais.
- Escolher estrategias de resposta, como recalibracao, regras temporarias, re-treino e aprendizado online.
- Projetar pipelines batch e streaming para observabilidade de modelos em escala.
- Integrar validacao de dados, monitoramento de drift e governanca em fluxos de MLOps.

## 📚 Conteudo das Aulas

### Aula 1. Fundamentos do Data Drift em ML

- Objetivo especifico: introduzir o fenomeno do drift e mostrar por que modelos em producao perdem acuracia quando o mundo muda.
- Teoria-chave: data drift altera a distribuicao de entrada $P(X)$, enquanto concept drift altera a relacao $P(Y|X)$. Sem monitoramento e manutencao, o modelo entra em processo de model decay e deixa de atender aos objetivos originais do negocio.
- Tecnologias utilizadas: Python, NumPy, SciPy, scikit-learn, Matplotlib.
- Pasta da aula: [01_fundamentos_do_data_drift_em_ml](01_fundamentos_do_data_drift_em_ml/)

### Aula 2. Deteccao Estatistica de Drift de Dados

- Objetivo especifico: formalizar a deteccao de drift por meio de testes de hipotese e metricas estatisticas.
- Teoria-chave: testes como Kolmogorov-Smirnov e metricas como PSI ajudam a comparar amostras historicas e recentes. A interpretacao correta de p-values, thresholds e distribuicoes de referencia e essencial para reduzir falsos alarmes e operacionalizar alertas.
- Tecnologias utilizadas: Python, NumPy, Pandas, SciPy, Matplotlib, JupyterLab.
- Pasta da aula: [02_deteccao_estatistica_de_drift_de_dados](02_deteccao_estatistica_de_drift_de_dados/)

### Aula 3. Estrategias de Mitigacao de Drift

- Objetivo especifico: transformar sinais de drift em acoes taticas e estruturais para preservar performance.
- Teoria-chave: nem todo drift exige a mesma resposta. Regras temporarias, ajuste de limiar, aprendizado online, ensembles adaptativos e re-treino periodico compoem um portifolio de mitigacoes com trade-offs distintos de custo, latencia e risco.
- Tecnologias utilizadas: Python, River, Evidently, scikit-learn.
- Pasta da aula: [03_estrategias_de_mitigacao_de_drift](03_estrategias_de_mitigacao_de_drift/)

### Aula 4. Metricas Avancadas para Deteccao de Drift

- Objetivo especifico: detectar drifts sutis e multivariados que escapam de metricas univariadas tradicionais.
- Teoria-chave: metricas como MMD, Wasserstein e abordagens baseadas em kernels aumentam a sensibilidade para mudancas em alta dimensao. Elas sao especialmente uteis quando relacoes entre variaveis mudam sem alterar de forma visivel as distribuicoes marginais.
- Tecnologias utilizadas: Python, NumPy, SciPy, scikit-learn, Alibi-Detect.
- Pasta da aula: [04_metricas_avancadas_para_deteccao_de_drift](04_metricas_avancadas_para_deteccao_de_drift/)

### Aula 5. Drift em Embeddings e Testes de Duas Amostras

- Objetivo especifico: aplicar tecnicas de monitoramento a dados textuais e representacoes vetoriais.
- Teoria-chave: embeddings permitem observar mudancas semanticas que nao aparecem em contagens simples de palavras ou features tabulares. Testes de duas amostras, classifier drift e projecoes em UMAP ou t-SNE ajudam a diagnosticar drifts em espacos latentes.
- Tecnologias utilizadas: Hugging Face Transformers, PyTorch, scikit-learn, UMAP-learn, spaCy, NLTK.
- Pasta da aula: [05_drift_em_embeddings_e_testes_de_duas_amostras](05_drift_em_embeddings_e_testes_de_duas_amostras/)

### Aula 6. Monitoramento Continuo e Data SLOs em Producao

- Objetivo especifico: ligar metricas tecnicas de observabilidade a objetivos operacionais claros.
- Teoria-chave: monitorar nao basta; e necessario definir SLIs, thresholds, janelas temporais e Data SLOs que indiquem quando agir. O foco passa a ser governanca continua, priorizacao de alertas e reducao do tempo de resposta a degradacao.
- Tecnologias utilizadas: Python, Pandas, scikit-learn, Evidently, NannyML.
- Pasta da aula: [06_monitoramento_continuo_e_data_slos_em_producao](06_monitoramento_continuo_e_data_slos_em_producao/)

### Aula 7. Otimizacoes e Escala no Monitoramento de Drift

- Objetivo especifico: projetar arquiteturas para detectar drift com alto volume de dados e baixa latencia.
- Teoria-chave: em cenarios de streaming, a deteccao precisa equilibrar throughput, custo e qualidade estatistica. Isso envolve janela temporal, brokers de eventos, processamento incremental, pipelines paralelos e, quando necessario, otimizacoes de baixo nivel em componentes criticos.
- Tecnologias utilizadas: Python, kafka-python, River, streaming batch/near-real-time, Rust opcional para aceleracao.
- Pasta da aula: [07_otimizacoes_e_escala_no_monitoramento_de_drift](07_otimizacoes_e_escala_no_monitoramento_de_drift/)

### Aula 8. Integracao e Revisao (Ferramentas e Validacao)

- Objetivo especifico: consolidar um pipeline completo de monitoramento e resposta a drift em producao.
- Teoria-chave: a melhor pratica nao e uma ferramenta isolada, mas a combinacao de validacao de dados, deteccao de drift, estimativa de impacto no modelo e tomada de acao. Evidently, NannyML e Great Expectations atuam de forma complementar para tornar a operacao do modelo observavel e resiliente.
- Tecnologias utilizadas: Evidently, NannyML, Great Expectations, Python, scikit-learn.
- Pasta da aula: [08_integracao_e_revisao_ferramentas_e_validacao](08_integracao_e_revisao_ferramentas_e_validacao/)

## 🚀 Como Usar Este Repositorio

### Pre-requisitos

- Python 3.12+
- Git
- JupyterLab
- Ambiente virtual Python (`venv` ou equivalente)
- Opcional: Docker ou Podman para encapsular experimentos

### Instalacao

```bash
git clone https://github.com/<org>/mlet_data-drift.git
cd mlet_data-drift
python -m venv .venv
```

Ativacao do ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

Instalacao das dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencias opcionais para NLP:

```bash
python -m nltk.downloader punkt stopwords
python -m spacy download pt_core_news_sm
```

### Execucao

Enquanto as pastas de aula estiverem vazias, o fluxo recomendado e:

1. Ler este README e o documento de arquitetura.
2. Revisar os ADRs para entender as decisoes tecnicas.
3. Estudar os business cases para conectar os conceitos a cenarios reais.
4. Popular cada pasta de aula com notebooks, scripts e datasets por meio do PROMPT B.

Quando o conteudo pratico estiver disponivel, os comandos padrao do repositorio serao:

```bash
task lint
task format
task test
```

## 📂 Estrutura do Repositorio

```text
mlet_data-drift/
├── 01_fundamentos_do_data_drift_em_ml/
├── 02_deteccao_estatistica_de_drift_de_dados/
├── 03_estrategias_de_mitigacao_de_drift/
├── 04_metricas_avancadas_para_deteccao_de_drift/
├── 05_drift_em_embeddings_e_testes_de_duas_amostras/
├── 06_monitoramento_continuo_e_data_slos_em_producao/
├── 07_otimizacoes_e_escala_no_monitoramento_de_drift/
├── 08_integracao_e_revisao_ferramentas_e_validacao/
├── docs/
│   ├── architecture/
│   │   ├── adr/
│   │   ├── diagrams/
│   │   └── overview.md
│   ├── business/
│   │   ├── process-diagrams/
│   │   ├── README.md
│   │   └── use-cases.md
│   └── contributing.md
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## 📖 Documentacao

- Visao geral da arquitetura: [docs/architecture/overview.md](docs/architecture/overview.md)
- Diagramas de arquitetura: [docs/architecture/diagrams/system-architecture.md](docs/architecture/diagrams/system-architecture.md)
- ADRs: [docs/architecture/adr/](docs/architecture/adr/)
- Casos de negocio: [docs/business/README.md](docs/business/README.md)
- Guia de contribuicao: [docs/contributing.md](docs/contributing.md)

## 🤝 Contribuindo

As regras de contribuicao, padroes de codigo e fluxo de PR estao descritos em [docs/contributing.md](docs/contributing.md).

## 📄 Licenca

Este repositorio adota a [MIT License](LICENSE).