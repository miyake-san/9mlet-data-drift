# ADR-0002: Estrategia hibrida para deteccao de drift

## Status

Aceito

## Contexto

Metodos univariados sao interpretaveis e baratos, mas falham em detectar mudancas sutis nas relacoes entre variaveis. Metodos multivariados e baseados em modelos sao mais sensiveis, porem tendem a ser mais caros, menos transparentes e mais delicados de calibrar. A disciplina precisa mostrar tanto o basico quanto o estado da pratica.

## Decisao

Usar uma estrategia hibrida de deteccao: testes e metricas univariadas como KS, PSI e Wasserstein para monitoramento inicial e diagnostico rapido, complementadas por metricas avancadas como MMD, classifier drift e abordagens com embeddings em cenarios de alta dimensionalidade.

## Consequencias

### Positivas

- Combina interpretabilidade operacional com maior poder de deteccao para drifts complexos.
- Permite ensinar quando um metodo simples e suficiente e quando e necessario elevar o rigor da analise.

### Negativas

- Introduz mais parametros, thresholds e decisoes metodologicas para calibrar.
- Exige maior cuidado para evitar excesso de alarmes ou sobreposicao de metricas com pouco ganho marginal.

## Alternativas Consideradas

1. Usar apenas testes univariados - simples e barato, mas insuficiente para drifts multivariados e embeddings.
2. Usar apenas metricas avancadas - poderoso, mas com menor interpretabilidade e maior custo computacional para parte do curso.

## Referencias

- Rabanser, Gunnemann and Lipton. Failing loudly: dataset shift detection.
- Hinder, Vaquet and Hammer. Survey on monitoring evolving environments.