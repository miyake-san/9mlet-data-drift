# Roteiro de Live - Aula 6 - Stack de Monitoramento de Drift

Formato: teleprompter. Frases curtas, leitura natural, pausas marcadas com [PAUSA].
Tempo estimado total: 12 a 15 minutos.
Pré-requisito: stack ja buildada localmente (docker compose build executado antes da live).

---

## Bloco 1 - Abertura e contexto (1 min)

Ola pessoal, bem vindos a este complemento pratico da Aula 6.

Na aula a gente viu, em teoria e no notebook, como definir SLOs de dados e como detectar drift usando teste KS, PSI e taxa de missings.

[PAUSA]

Hoje a proposta e diferente. A gente vai sair do notebook e colocar exatamente esses mesmos checks rodando como um servico, dentro de containers, sendo lidos por Prometheus e visualizados em Grafana.

O objetivo aqui e mostrar como o codigo que voce ja escreveu na aula vira monitoramento de producao, com dashboard atualizando ao vivo, sem reescrever nada.

[PAUSA]

Vou abrir o repositorio e mostrar a estrutura, depois subir os containers, e a gente acompanha junto a transicao entre o cenario sem drift e o cenario com drift.

---

## Bloco 2 - Visao geral da stack (2 min)

Dentro da pasta da Aula 6, eu criei uma subpasta chamada monitoring_stack. E ai dentro tem tres servicos.

O primeiro e o drift exporter. Esse e o app Python. Ele e o coracao da demo.

O segundo e o Prometheus, que coleta as metricas do exporter a cada cinco segundos.

O terceiro e o Grafana, que mostra tudo num dashboard ja pronto.

[PAUSA]

A parte mais importante e a seguinte: o Dockerfile do exporter aponta o build context para a pasta da Aula 6 inteira. Isso significa que ele copia o pacote src para dentro do container e usa exatamente os mesmos modulos que os notebooks usam.

A mesma classe SLOMonitor. O mesmo MonitoringPipeline. O mesmo DataPreprocessor.

Nao tem codigo duplicado. O que voce viu no notebook e o que esta rodando no container.

[PAUSA]

O que muda e so a forma de entrega. Em vez de imprimir um relatorio no Jupyter, o exporter publica as metricas no endpoint barra metrics, no formato que o Prometheus entende.

---

## Bloco 3 - Como o exporter simula drift (2 min)

Antes de subir, vale explicar o que o exporter faz em loop.

Primeiro, ele garante que o dataset sintetico de credito existe. Se nao existir, ele gera, usando a mesma funcao do notebook um.

[PAUSA]

Depois ele treina um modelo de regressao logistica, mas usando apenas a particao de referencia. Ou seja, o modelo so conhece o mundo antes do drift.

Em seguida, comeca o loop principal. E aqui esta o truque pedagogico da demo.

[PAUSA]

A cada iteracao, ele amostra um lote de quinhentos registros. Mas ele alterna entre duas fases.

Na fase pre drift, o lote vem da particao de referencia. O modelo se comporta bem. Acuracia alta, p-valor alto, PSI baixo.

Na fase pos drift, o lote vem da particao de producao. Que e a particao onde introduzimos shift na idade, na renda mensal e no score de credito.

[PAUSA]

A cada doze iteracoes, ele troca de fase. Cada iteracao leva dez segundos. Entao, em cerca de dois minutos, todo mundo no dashboard vai ver a estaca virar de verde para vermelho.

E isso e o que torna o conceito de drift visivel. Nao e mais uma celula de notebook. E uma linha caindo num grafico, ao vivo.

---

## Bloco 4 - Subindo os containers (2 min)

Vamos para o terminal.

Eu ja estou na pasta da Aula 6. Vou entrar na pasta monitoring_stack.

[ACAO: cd 06_monitoramento_continuo_e_data_slos_em_producao\monitoring_stack]

Pronto. Agora subo os tres servicos com um comando so.

[ACAO: docker compose up -d --build]

[PAUSA]

Como eu ja buildei antes, esse passo e rapido. Na primeira vez que voces rodarem em casa, leva de um a dois minutos para baixar a imagem base do Python e instalar numpy, pandas, scikit learn e o prometheus client.

[PAUSA]

Enquanto sobe, deixa eu mostrar os logs do exporter, porque ele faz o treino do modelo no startup.

[ACAO: docker compose logs -f drift-exporter]

[PAUSA]

Vejam que aparece a mensagem de geracao do dataset, depois o modelo treinado, e ai comeca a iteracao zero. Cada linha de log mostra a fase atual, quantos checks passaram e quantos falharam.

Na iteracao zero, fase pre drift, todos os checks passam. Status HEALTHY.

Vou deixar isso rodando aqui no canto e abrir o navegador.

---

## Bloco 5 - Validando o Prometheus (1 min)

Primeiro, sempre que voces subirem uma stack assim, vale a pena confirmar que o Prometheus esta vendo o exporter.

Abro localhost na porta nove zero nove zero.

[ACAO: navegar para http://localhost:9090/targets]

[PAUSA]

Aqui na pagina de targets, eu vejo o job drift exporter listado como UP, em verde. Isso significa que o Prometheus conseguiu fazer scrape na porta oito mil do exporter, na rede interna do Docker.

Se aparecesse DOWN, geralmente e o container ainda subindo. Espera trinta segundos e atualiza.

[PAUSA]

Para ja mostrar uma metrica, eu vou na aba Graph e digito ml_model_accuracy.

[ACAO: digitar query]

Pronto. Aparece a serie temporal da acuracia. Por enquanto so temos pontos da fase pre drift, entao a linha esta colada perto de oitenta e nove por cento.

---

## Bloco 6 - Dashboard do Grafana (3 min)

Agora a parte mais interessante. Vou no Grafana.

[ACAO: navegar para http://localhost:3000]

[PAUSA]

Login admin admin. E reparem: o dashboard ja abre direto. Eu deixei provisionado como dashboard home, entao nao precisa procurar.

O nome dele e Aula 6, ML Monitoring, Pre Pos Drift.

[PAUSA]

Vou explicar os paineis na ordem.

No topo esquerdo, o painel Fase Atual. Verde com a palavra PRE DRIFT. Esse painel le a metrica ml_phase e mostra em qual lado da simulacao a gente esta.

Do lado, Status Geral. Tambem em verde, dizendo HEALTHY. Esse e o agregado de todos os checks de SLO.

Mais a direita, contagem de checks aprovados e reprovados, e o contador acumulado de alertas. Esse counter de alertas e importante porque ele nao reseta. E o nosso historico de violacoes.

[PAUSA]

Na linha de baixo, o painel mais importante para o conceito da aula. Acuracia do Modelo versus SLO.

A linha cheia e a acuracia do batch corrente. A linha tracejada vermelha e o SLO, que eu configurei em oitenta e cinco por cento.

Enquanto a linha cheia estiver acima da tracejada, o modelo esta entregando o que prometeu.

[PAUSA]

Ao lado, o Brier Score. Esse e o indicador de calibracao. Quanto menor, melhor. E aqui a gente vai ver, na fase pos drift, que ele sobe, confirmando o que os papers de Ovadia e companhia ja mostram. Drift nao quebra so a acuracia, quebra a confianca da probabilidade.

[PAUSA]

Na proxima linha, os dois paineis de drift propriamente dito.

A esquerda, PSI por feature. Tem tres faixas de cor. Verde ate zero virgula um. Laranja entre zero virgula um e zero virgula dois e cinco. Vermelho acima.

A direita, p-valor do teste KS por feature. Linha cai abaixo de zero virgula zero cinco, drift detectado.

[PAUSA]

Mais abaixo, o heatmap de checks de SLO. Essa visualizacao e muito util em sala de guerra. Cada linha do heatmap e um check individual. Verde e passou. Vermelho e violou. Voce bate o olho e ja sabe quais features estao dando problema agora.

E por fim, taxa de missings por coluna, e o bar gauge de alertas acumulados por tipo de check.

---

## Bloco 7 - A transicao ao vivo (2 min)

Agora a gente espera.

A simulacao esta na iteracao oito. Em mais quatro iteracoes, ou seja, em quarenta segundos, a fase vira para pos drift.

[PAUSA, aguardar transicao]

Olha o que acontece. O painel de fase mudou de verde PRE DRIFT para vermelho POS DRIFT.

O status geral foi para CRITICAL.

A acuracia, que estava em torno de oitenta e nove por cento, caiu para a faixa dos setenta. Cruzou a linha tracejada do SLO.

[PAUSA]

O Brier Score subiu de forma visivel.

O PSI da feature idade, da renda mensal e do score de credito disparou. Saiu do verde e foi para o vermelho. Drift severo, exatamente como esperado.

Os p-valores do KS dessas mesmas features cairam para perto de zero.

[PAUSA]

E o heatmap de checks ficou com varias barras vermelhas. Olhem que voces conseguem ler exatamente quais SLOs falharam neste batch. Accuracy, drift KS idade, drift KS renda mensal, PSI idade, PSI score credito.

O contador de alertas no canto superior direito ja contabilizou mais de uma dezena de violacoes.

[PAUSA]

Essa e a foto que voces nunca conseguem ter so rodando o notebook. O notebook te da uma execucao pontual. O dashboard te da o filme.

E e esse filme que um time de SRE de machine learning olha quando o pager toca.

---

## Bloco 8 - O que isso ensina sobre producao (1 min)

Para fechar, tres observacoes praticas.

Primeira. Reparem que o codigo do exporter e o codigo do notebook sao o mesmo. Isso e o que a gente quer em producao. Mesma logica de check, mesma definicao de SLO, mesma classe Python.

A diferenca entre experimento e producao nao deveria ser o algoritmo. Deveria ser so o canal de entrega da metrica.

[PAUSA]

Segunda. Tudo aqui esta declarado em arquivo. Dockerfile, compose, prometheus.yml, datasource do Grafana, JSON do dashboard. Voce comita isso no Git e qualquer pessoa do time sobe um ambiente identico em um comando.

Essa e a base de qualquer pratica seria de monitoramento. Nao tem botao clicado em interface.

[PAUSA]

Terceira. O dashboard nao decide por voce. Ele te da SLI, SLO e error budget. A decisao de retreinar, de reverter ou de bloquear o deploy continua sendo do time.

O monitoramento existe para que essa decisao seja tomada com dado, e nao no escuro.

---

## Bloco 9 - Encerramento (30 seg)

E essa e a stack.

O passo a passo completo, com troubleshooting, esta no README dentro da pasta monitoring_stack. Tem inclusive sugestoes de como apertar o SLO e ver violacao mesmo na fase pre drift, caso voces queiram brincar mais.

Para parar tudo, basta um docker compose down. Para limpar volumes inclusive, docker compose down menos v.

[PAUSA]

Esse material complementa o notebook tres da Aula 6. Recomendo executar o notebook primeiro, para entender o calculo de cada metrica, e depois subir a stack para ver as mesmas metricas em movimento.

Era isso. Obrigado, e bons estudos.

[FIM]
