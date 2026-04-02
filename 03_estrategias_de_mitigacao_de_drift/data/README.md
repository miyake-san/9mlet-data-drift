# Datasets — Aula 03: Estratégias de Mitigação de Drift

## Dataset Principal: Churn Telecom com Drift Simulado

### Descrição

Dataset sintético que simula o cenário discutido no documento acadêmico: uma operadora
de telecomunicações enfrenta drift quando um concorrente lança uma oferta disruptiva,
alterando hábitos de consumo e deslocando a taxa de cancelamento.

O dataset é dividido em **três períodos temporais** para capturar a dinâmica de drift:

- **Período 1 (T0–T1):** Regime estável pré-competição (3.000 amostras)
- **Período 2 (T1–T2):** Choque competitivo — drift abrupto (3.000 amostras)
- **Período 3 (T2–T3):** Adaptação gradual — drift incremental (3.000 amostras)

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `customer_id` | int | Identificador único do cliente |
| `tenure_months` | int | Tempo como cliente (meses) |
| `monthly_charges` | float | Valor mensal do plano (R$) |
| `total_charges` | float | Valor total pago (R$) |
| `data_usage_gb` | float | Consumo de dados (GB/mês) |
| `call_duration_min` | float | Duração de chamadas (min/mês) |
| `num_complaints` | int | Número de reclamações nos últimos 6 meses |
| `payment_delay_days` | int | Dias de atraso no pagamento |
| `contract_type` | int | Tipo de contrato (0=mensal, 1=anual, 2=bienal) |
| `has_premium_support` | int | Suporte premium (0=não, 1=sim) |
| `competitor_offer_exposure` | float | Nível de exposição à oferta do concorrente [0,1] |
| `customer_value_segment` | int | Segmento de valor (0=baixo, 1=médio, 2=alto) |
| `period` | int | Período temporal (1, 2 ou 3) |
| `timestamp_month` | int | Mês da observação (1–36) |
| `churn` | int | Cancelou o serviço (0=não, 1=sim) |

### Estatísticas

- **Instâncias:** 9.000 (3.000 por período)
- **Features:** 12 (+ id, period, timestamp, target)
- **Target:** `churn` (binário)
- **Taxa de churn por período:**
  - Período 1: ~12% (regime estável)
  - Período 2: ~28% (choque competitivo)
  - Período 3: ~20% (adaptação gradual)

### Drift Simulado

O drift é introduzido de forma controlada entre os períodos:

1. **Covariate shift:** `data_usage_gb` diminui significativamente no período 2
   (clientes migrando para o concorrente reduzem consumo).
2. **Prior probability shift:** Taxa de churn salta de ~12% para ~28%.
3. **Concept drift:** A relação entre `monthly_charges` e churn se inverte parcialmente
   (no período 2, clientes de planos mais caros também cancelam).

### Fonte

Dataset **sintético**, gerado programaticamente para fins pedagógicos.

### Reprodução

```bash
python -c "from src.data_preprocessing import DataPreprocessor; DataPreprocessor().generate_and_save()"
```
