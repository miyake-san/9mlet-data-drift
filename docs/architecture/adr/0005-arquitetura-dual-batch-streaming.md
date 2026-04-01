# ADR-0005: Arquitetura dual batch e streaming

## Status

Aceito

## Contexto

A disciplina precisa servir tanto a casos de estudo reproduziveis em notebook quanto a cenarios de observabilidade em escala, proximos da realidade de producao. Um desenho apenas batch simplifica o ensino, mas limita a discussao sobre latencia e throughput. Um desenho apenas streaming aproxima o mundo real, mas aumenta significativamente a dificuldade de reproducao academica.

## Decisao

Adotar uma arquitetura dual. Os exemplos introdutorios e os experimentos fundamentais serao apresentados em batch, com comparacoes entre conjuntos de referencia e dados recentes. Os cenarios avancados passarao a incluir streaming, chunking temporal, brokers de eventos e processamento incremental.

## Consequencias

### Positivas

- Mantem a acessibilidade didatica sem abrir mao de discutir escala e operacao real.
- Permite reutilizar conceitos estatisticos em contextos operacionais distintos.

### Negativas

- Exige documentacao cuidadosa para que alunos entendam o que muda entre os dois modos de execucao.
- Pode gerar duplicacao parcial de exemplos e pipelines se a abstracao nao for bem planejada.

## Alternativas Consideradas

1. Batch apenas - ideal para ensino inicial, mas insuficiente para a aula de escala e MLOps avancado.
2. Streaming apenas - tecnicamente atraente, mas mais dificil de executar em ambientes educacionais heterogeneos.

## Referencias

- Kreps et al. Kafka and distributed event streaming.
- Baylor et al. TFX and continuous training in production ML.