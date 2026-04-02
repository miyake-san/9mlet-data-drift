# Datasets – Aula 2: Detecção Estatística de Drift de Dados

## Dataset Principal

### Descrição

Dataset sintético simulando dados de clientes de uma **fintech de crédito**, conforme o cenário
apresentado no documento da Aula 2. Os dados representam duas populações:

- **Referência (treinamento):** 5.000 clientes com perfil "antigo" (distribuições estáveis).
- **Produção (com drift):** 5.000 clientes com perfil "novo" (distribuições deslocadas).

O drift foi injetado propositalmente em variáveis-chave para simular o cenário descrito:
*"se antes a maioria dos tomadores de empréstimo tinha entre 30 e 50 anos, mas agora grande
parte dos novos clientes é de jovens de 20 e poucos anos"*.

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `idade` | float | Idade do cliente (anos) |
| `renda_mensal` | float | Renda mensal (R$) |
| `score_credito` | float | Score de crédito (0–1000) |
| `tempo_emprego` | float | Tempo no emprego atual (anos) |
| `valor_emprestimo` | float | Valor do empréstimo solicitado (R$) |
| `taxa_utilizacao_credito` | float | Proporção do crédito utilizado (0–1) |
| `num_parcelas_atraso` | int | Número de parcelas em atraso |
| `tipo_residencia` | str | Tipo: aluguel, proprio, financiado |
| `categoria_risco` | str | Baixo, Medio, Alto |
| `origem` | str | "referencia" ou "producao" |
| `inadimplente` | int | 0 = adimplente, 1 = inadimplente |

### Estatísticas

- **Instâncias totais:** 10.000 (5.000 referência + 5.000 produção)
- **Features numéricas:** 7
- **Features categóricas:** 2
- **Target:** `inadimplente` (binário)
- **Balanceamento:** ~80/20 (adimplente/inadimplente)

### Drift Injetado

| Feature | Referência (μ, σ) | Produção (μ, σ) | Tipo de Drift |
|---------|-------------------|-----------------|---------------|
| `idade` | (40, 10) | (30, 8) | Deslocamento de média |
| `renda_mensal` | (5000, 2000) | (3500, 1500) | Redução de média |
| `score_credito` | (650, 100) | (550, 120) | Deslocamento + aumento dispersão |
| `taxa_utilizacao_credito` | (0.3, 0.15) | (0.5, 0.2) | Aumento de média |
| `tipo_residencia` | 50/30/20% | 30/25/45% | Mudança de proporções |

### Fonte

Gerado sinteticamente com `numpy.random` e seeds fixas para reprodutibilidade.

### Reprodução

```bash
python -c "from src.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); dp.generate_and_save()"
```
