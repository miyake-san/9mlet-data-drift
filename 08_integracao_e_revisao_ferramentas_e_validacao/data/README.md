# Datasets - Aula 8: Integração e Revisão (Ferramentas e Validação)

## Dataset Principal

### Descrição
Dataset sintético simulando transações de uma **plataforma de pagamentos online**,
conforme o caso de negócio apresentado no Documento 04 da Aula 8.
O dataset contém dados de referência (baseline do modelo) e dados de produção
com drift simulado (mudanças em distribuições de features e surgimento de novos
padrões de fraude).

### Estrutura

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| transaction_id | str | Identificador único da transação |
| timestamp | datetime | Data/hora da transação |
| valor_transacao | float | Valor monetário da transação (R$) |
| tempo_conta_cliente | int | Dias desde a criação da conta do cliente |
| num_transacoes_24h | int | Número de transações do cliente nas últimas 24h |
| valor_medio_historico | float | Valor médio das transações históricas do cliente |
| distancia_localizacao | float | Distância (km) entre localização da transação e endereço cadastrado |
| hora_transacao | int | Hora do dia da transação (0-23) |
| tipo_cartao | str | Tipo de cartão (credito, debito, prepago) |
| canal_transacao | str | Canal utilizado (app, web, pos, telefone) |
| pais_origem | str | País de origem da transação |
| score_risco_dispositivo | float | Score de risco do dispositivo (0-1) |
| tentativas_senha | int | Número de tentativas de senha antes da transação |
| is_weekend | int | Se a transação ocorreu no fim de semana (0/1) |
| razao_valor_medio | float | Razão entre valor da transação e média histórica |
| fraude | int | Rótulo: 1 = fraude, 0 = legítima |
| periodo | str | Período dos dados: 'referencia' ou 'producao' |

### Estatísticas
- **Instâncias totais**: 10,000
  - Referência (baseline): 5,000
  - Produção (com drift): 5,000
- **Features numéricas**: 10
- **Features categóricas**: 4
- **Classes**: 2 (fraude/legítima)
  - Referência: ~5% fraude (balanceamento realista)
  - Produção: ~8% fraude (prior drift simulado)

### Tipos de Drift Simulados
1. **Covariate drift**: Aumento nos valores de `valor_transacao` e `distancia_localizacao`
2. **Prior drift**: Mudança na proporção de fraudes (5% → 8%)
3. **Concept drift**: Novos padrões de fraude via `canal_transacao` (telefone)

### Fonte
Sintético — gerado por `scripts/generate_dataset.py`

### Reprodução
```bash
python scripts/generate_dataset.py
```
