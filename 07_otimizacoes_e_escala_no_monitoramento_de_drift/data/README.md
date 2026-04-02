# Datasets — Aula 7: Otimizações e Escala no Monitoramento de Drift

## Dataset Principal: FinBank Transactions

### Descrição

Dataset sintético que simula transações financeiras do caso FinBank, conforme descrito
no material da Aula 7. O dataset contém 10.000 transações distribuídas ao longo de
8 meses, com drift gradual injetado entre os meses 4–6 e drift abrupto nos meses 7–8.
Isso permite exercitar detecção de drift em cenários realistas de produção.

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| transaction_id | str | Identificador único da transação |
| timestamp | datetime | Data/hora da transação |
| month | int | Mês (1–8) para facilitar agrupamento temporal |
| amount | float | Valor da transação (R$) |
| num_items | int | Quantidade de itens na transação |
| hour_of_day | int | Hora do dia (0–23) |
| day_of_week | int | Dia da semana (0=seg, 6=dom) |
| customer_age | float | Idade do cliente |
| account_age_days | int | Idade da conta em dias |
| transaction_count_30d | int | Transações nos últimos 30 dias |
| avg_amount_30d | float | Valor médio nos últimos 30 dias |
| channel | str | Canal (app, web, physical, phone) |
| is_international | int | Transação internacional (0/1) |
| is_fraud | int | Rótulo de fraude (0/1) |

### Estatísticas

- **Instâncias:** 10.000
- **Features numéricas:** 8
- **Features categóricas:** 2 (channel, is_international)
- **Variável-alvo:** is_fraud (taxa ~3% estável meses 1-3, ~8% meses 7-8)
- **Drift injetado:**
  - Meses 1–3: distribuição estável (baseline)
  - Meses 4–6: drift gradual (aumento em amount, mudança em channel)
  - Meses 7–8: drift abrupto (mudança forte em múltiplas features)

### Fonte

Dataset 100% sintético gerado via NumPy com seeds fixas para reprodutibilidade.

### Reprodução

```bash
python scripts/generate_dataset.py
```
