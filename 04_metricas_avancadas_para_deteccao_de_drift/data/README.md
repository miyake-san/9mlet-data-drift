# Datasets – Aula 04: Métricas Avançadas para Detecção de Drift

## Dataset Principal

### Descrição

Dataset **sintético** que simula o cenário da fintech de crédito descrito no Documento 04.
Os dados representam clientes em dois períodos temporais (referência e atual), onde um
**drift multivariado sutil** é introduzido: as distribuições marginais de cada variável
permanecem similares entre os períodos, porém a **correlação conjunta entre variáveis muda**
(por exemplo, inversão da correlação entre renda e idade).

Este cenário ilustra exatamente o "ponto cego" dos testes univariados (KS, PSI) discutido
na Seção "Limites de Métricas Simples em Altas Dimensionalidades" do Documento 04.

### Estrutura

| Coluna | Tipo | Descrição |
|---|---|---|
| idade | float64 | Idade do cliente (anos), distribuição normal μ=40, σ=10 |
| renda | float64 | Renda mensal (R$), correlacionada com idade |
| divida | float64 | Valor total de dívidas (R$) |
| score_credito | float64 | Score de crédito (0–1000) |
| tempo_emprego | float64 | Tempo no emprego atual (anos) |
| num_parcelas | float64 | Número de parcelas ativas |
| periodo | str | Período: "referencia" ou "atual" |

### Estatísticas

- **Instâncias totais:** 10.000 (5.000 referência + 5.000 atual)
- **Features numéricas:** 6
- **Períodos:** 2 (referência sem drift, atual com drift multivariado)
- **Tipo de drift:** Inversão de correlação renda-idade no período atual

### Fonte

Sintético – gerado programaticamente com seeds fixas para reprodutibilidade.

### Reprodução

```bash
python -c "
from src.data_preprocessing import DataPreprocessor
dp = DataPreprocessor(seed=42)
dp.generate_and_save()
"
```

Ou dentro dos notebooks:

```python
from src.data_preprocessing import DataPreprocessor
dp = DataPreprocessor(seed=42)
df = dp.generate_dataset()
```
