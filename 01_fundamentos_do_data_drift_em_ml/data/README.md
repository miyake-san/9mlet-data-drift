# Datasets — Aula 1: Fundamentos do Data Drift em ML

## Dataset Principal

### Descrição

Dataset sintético que simula o cenário motivador da loja online descrito no
material da aula.  Os dados representam atributos demográficos e comportamentais
de clientes, divididos em 5 períodos temporais.  A partir do período 3,
**data drift (covariate shift)** é injetado: a distribuição de P(X) muda
enquanto P(Y|X) permanece constante, ilustrando a definição formal de
*feature drift* apresentada na seção *Saiba Mais*.

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `idade` | float | Idade normalizada do cliente |
| `renda_mensal` | float | Renda mensal normalizada |
| `tempo_no_site_min` | float | Tempo médio de navegação (min) |
| `paginas_visitadas` | float | Páginas visitadas por sessão |
| `itens_carrinho` | float | Itens adicionados ao carrinho |
| `ticket_medio` | float | Ticket médio de compra |
| `frequencia_visitas_mes` | float | Frequência de visitas mensal |
| `dias_desde_ultima_compra` | float | Dias desde a última compra |
| `score_engajamento` | float | Score de engajamento do cliente |
| `num_categorias_visitadas` | float | Categorias de produto exploradas |
| `target` | int | Classe binária: 1 = compra, 0 = não-compra |
| `periodo` | int | Período temporal (1 a 5) |

### Estatísticas

- **Instâncias:** 10.000
- **Features:** 10 numéricas
- **Target:** binário (2 classes)
- **Períodos:** 5 (2.000 amostras cada)
- **Drift injetado:** períodos 3–5 (deslocamento crescente na média de 5 features)

### Fonte

Dataset **sintético**, gerado pelo script `scripts/generate_dataset.py`.

### Reprodução

```bash
cd mlet_data-drift
uv run python 01_fundamentos_do_data_drift_em_ml/scripts/generate_dataset.py
```

O script usa `random_state=42` para reprodutibilidade total.
