# Datasets — Aula 6: Monitoramento Contínuo e Data SLOs em Produção

## Dataset Principal

### Descrição

Dataset sintético que simula o cenário de uma **fintech de crédito digital**, conforme
descrito no material da aula: *"Imagine que uma fintech acaba de lançar um modelo de
machine learning para aprovação instantânea de crédito"*.

O dataset contém dois conjuntos:

- **Referência (treino)**: 5.000 amostras com distribuição original.
- **Produção (com drift)**: 5.000 amostras com drift introduzido em `idade` (+5 anos),
  `renda_mensal` (−15%) e `score_credito` (−50 pontos), além de concept drift no target.

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `idade` | float | Idade do solicitante (18–70 anos) |
| `renda_mensal` | float | Renda mensal em R$ (1.000–50.000) |
| `score_credito` | float | Score de crédito (0–1.000) |
| `tempo_emprego` | float | Tempo de emprego em anos (0–40) |
| `valor_emprestimo` | float | Valor solicitado em R$ (500–200.000) |
| `taxa_utilizacao_credito` | float | Taxa de utilização do crédito (0–1) |
| `num_parcelas` | int | Número de parcelas (6, 12, 18, 24, 36, 48, 60) |
| `historico_atrasos` | int | Quantidade de atrasos anteriores (0–20) |
| `saldo_conta` | float | Saldo em conta corrente em R$ |
| `qtd_dependentes` | int | Número de dependentes (0–8) |
| `is_production` | int | Flag: 0 = referência, 1 = produção |
| `target` | int | Inadimplência: 0 = adimplente, 1 = inadimplente |

### Estatísticas

- **Instâncias**: 10.000 (5.000 referência + 5.000 produção)
- **Features**: 10 numéricas + 1 flag + 1 target
- **Target**: binário (taxa de inadimplência ~30% referência, ~40% produção)
- **Valores ausentes**: ~0.5% na referência, ~2% na produção (introduzidos em
  `idade`, `renda_mensal`, `score_credito`, `tempo_emprego`, `saldo_conta`)

### Drift Introduzido

| Feature | Referência | Produção | Tipo |
|---------|-----------|----------|------|
| `idade` | μ = 35 | μ = 40 (+5) | Data drift |
| `renda_mensal` | fator = 1.0 | fator = 0.85 (−15%) | Data drift |
| `score_credito` | μ = 600 | μ = 550 (−50) | Data drift |
| `target` | logit base | logit + 0.5 | Concept drift |
| `historico_atrasos` | λ = 0.8 | λ = 1.5 | Data drift |
| **Missing rate** | ~0.5% | ~2% | Quality drift |

### Fonte

Dataset **sintético**, gerado programaticamente pelo módulo `src/data_preprocessing.py`.

### Reprodução

```bash
python generate_dataset.py
```

Ou via Python:

```python
from src.data_preprocessing import DataPreprocessor
df = DataPreprocessor.generate_dataset(n_reference=5000, n_production=5000)
```
