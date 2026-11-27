# Relatório — Explicabilidade local do classificador de crédito

**Reproduzir** (na raiz do repositório): `uv sync --frozen && uv run --frozen python 01-explicabilidade/analise_explicabilidade.py`
(~2 min em 4 CPUs). Não há dependências além das fixadas em `uv.lock`. Saídas em `01-explicabilidade/resultados/`
(`resultados.json`, `fidelidade_remocao.csv`, `casos.png`, `referencia_validacao.png`). Semente `20261117`, CatBoost com 2 threads.
Duas execuções seguidas produziram resultados idênticos (exceto tempos).

## 1. Método e escala

**SHAP intervencional para árvores** (Independent Tree SHAP), calculado pelo próprio CatBoost
(`get_feature_importance(type="ShapValues", reference_data=...)`). Motivos: é exato para o ensemble de árvores (não é
uma aproximação por amostragem como LIME/KernelSHAP), já vem na versão fixada do CatBoost e tem uma propriedade
verificável: **valor base + soma das contribuições = saída do modelo**. A explicação opera em **log-odds**
(saída bruta do CatBoost); contribuições não são pontos percentuais. O valor base é a média do log-odds do modelo
na referência. A versão intervencional foi preferida à "dependente de caminho" (padrão, sem referência) porque a
referência fica explícita, sai do treino e permite definir o teste de remoção da seção 3 com a mesma noção de
"ausência" de uma feature.

**Referência, escolhida na validação antes do teste.** Amostras aleatórias do treino com N ∈ {25, 50, 100, 200, 400},
5 sementes cada, explicando 200 linhas da validação e comparando com uma referência de 2.000 linhas.
Regra declarada: menor N com erro relativo L1 médio ≤ 10% e Jaccard top-3 ≥ 0,80.

| N | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| erro relativo vs. N=2000 | 22% | 16% | 11% | **9,7%** | 5,9% |
| Jaccard top-3 | 0,73 | 0,81 | 0,86 | **0,87** | 0,92 |
| ms por linha (lote de 200) | 0,9 | 1,7 | 2,3 | 2,4 | 2,7 |

**Escolhido: N = 200.** O "gabarito" de 2.000 linhas também é uma amostra, então o erro mede convergência, não
verdade absoluta.

## 2. Três casos do teste

Critério: a linha do teste cuja probabilidade prevista está mais perto dos percentis 10, 50 e 90 das probabilidades
previstas no teste. O rótulo observado não foi usado (os três têm `inadimplente_90d = 0`, o que não diz se a
probabilidade de cada um estava "certa": é uma única realização de um evento incerto). Risco base (média na
referência): **22,5%**. "Sem o fator" = probabilidade obtida retirando só essa contribuição do log-odds (leitura
aproximada, válida na escala aditiva).

| Caso | ID | Prob. | Principais fatores (contribuição em log-odds) |
|---|---|---|---|
| Baixo (p10) | S0007173 | **6,4%** | valor_solicitado 147 mil (+1,21); receita_mensal 154 mil (−0,66); comprometimento 1% (−0,55); saldo 84 mil (−0,48); variação 3m +18% (−0,36) |
| Intermediário (p50) | S0016371 | **18,0%** | comprometimento 5% (−0,37); 0 atrasos (−0,32); variação 3m −15% (+0,31); valor_solicitado 6,7 mil (−0,30); saldo 4,7 mil (+0,17) |
| Alto (p90) | S0005455 | **53,5%** | 1 atraso (+0,37); variação 3m −14% (+0,30); margem 11,6% (+0,21); receita 9,1 mil (+0,19); despesas fixas 1,8 mil (+0,16) |

Figura: `resultados/casos.png`.

**Leitura para leigos.**
*Baixo:* um negócio grande, com receita alta, pouca dívida comprometendo a receita, bom saldo e receita crescendo.
O valor pedido é alto em reais e, sozinho, puxaria o risco para cima (sem ele, o modelo daria ~2%), mas a receita
alta compensa: o pedido equivale a ~1 mês de receita, algo comum na base. Os dois fatores precisam ser lidos juntos.
*Intermediário:* nenhum atraso e pouca receita comprometida com dívidas reduzem o risco; a queda de 15% na receita
nos últimos 3 meses o aumenta. O saldo dá um empurrão contrário. O resultado fica um pouco abaixo da média.
*Alto:* teve um atraso no último ano, a receita caiu 14% e a margem é mais baixa que a típica; tudo empurra para
cima, e nenhum fator forte compensa. A probabilidade chega a 53%, mais que o dobro da média, mas o modelo não
afirma que o negócio vai atrasar.

## 3. Fidelidade

**(a) Aditividade:** |base + Σφ − log-odds do modelo| ≤ 6×10⁻¹⁵ em todas as 2.700 linhas do teste. Mede só
consistência aritmética: qualquer atribuição que distribua a diferença passaria nesse teste.

**(b) Remoção (critério principal).** Em 300 linhas aleatórias do teste, "removemos" as k features de maior |φ|
(k = 1…5), substituindo-as pelos valores de cada uma das 200 linhas da referência e tirando a média da saída;
comparamos com 5 sorteios de k features aleatórias. Se a explicação for fiel, remover as top-k deve mudar muito mais
o log-odds, e a soma das φ removidas deve prever essa mudança.

| k | variação média do log-odds: top-k | aleatórias | correlação previsto × real (top-k) | erro médio abs. (top-k) |
|---|---|---|---|---|
| 1 | 0,60 | 0,15 | 0,989 | 0,064 |
| 3 | 0,87 | 0,30 | 0,995 | 0,079 |
| 5 | 0,91 | 0,41 | 0,996 | 0,072 |

As features indicadas como mais importantes realmente movem a saída do modelo ~3–4× mais que features aleatórias,
e Σφ prevê a mudança com erro de ~0,07 log-odds (a diferença vem de interações entre features, que o SHAP reparte).
**O que não mede:** utilidade para uma pessoa, causalidade, nem comportamento em dados reais. **Hipóteses:** "remover"
= substituir por valores do treino independentemente das demais features. Isso quebra correlações (ex.: valor
solicitado de outra empresa com a receita desta) e avalia o modelo em pontos possivelmente implausíveis; além disso,
o teste usa a mesma noção de ausência que o próprio SHAP intervencional, então é favorável a ele por construção.

## 4. Estabilidade

Medidas: correlação de Spearman entre os vetores de contribuição, Jaccard das 3 features principais e mudança
relativa L1 (‖Δφ‖₁/‖φ‖₁).

* **Repetição idêntica:** diferença máxima 0. Esperado: o cálculo é determinístico. Isso não garante estabilidade
  frente a mudanças nas entradas ou na referência, por isso os dois testes abaixo.
* **Trocar a amostra de referência** (20 amostras de 200 linhas): desvio-padrão médio das top-3 ≈ 0,02–0,03
  log-odds. Jaccard top-3 médio: baixo 0,90, alto 0,98, **intermediário 0,55**. No caso intermediário as quatro
  primeiras contribuições são quase empatadas (0,37; 0,32; 0,31; 0,30), então trocas de posição são esperadas e não
  problemáticas; mas mostram que a ordem exata não deve ser comunicada como certa.
* **Perturbar ligeiramente as entradas** (50 por caso; mesma referência): valores em reais multiplicados pelo mesmo
  fator ∈ [0,98; 1,02] (preserva razões dívida/receita, valor/receita etc.); frações com ruído de ~1% (relativo) ou
  ±0,005 (variação e margem), cortadas aos domínios do dicionário; meses de atividade/relacionamento ±1 com
  relacionamento ≤ atividade; contagens, prazo e categorias inalterados (mudá-los é um salto discreto). Todas as
  perturbações passaram numa checagem automática de restrições. Resultado: Δprob máximo 0,5 / 1,3 / 2,1 p.p.;
  Spearman mínimo ≥ 0,98; top-3 idêntico em 100% / 84% / 100% das perturbações; mudança L1 mediana 1–4%.
  **Critério de problema:** top-3 mudar quando a previsão quase não muda. Isso só ocorreu no caso intermediário,
  pelo mesmo empate descrito acima.

## 5. Tempo

Ambiente: Linux x86_64, 4 CPUs, Python 3.12.3, CatBoost 1.2.8, 2 threads; referência de 200 linhas; 3 repetições.
Preparação (carregar modelo, amostrar referência, criar Pool, sem ler CSVs): ~3 ms. Explicação de um caso: **~12–17 ms**
(SHAP intervencional) contra ~3–5 ms do TreeSHAP sem referência (faixas de duas execuções). Em lote o custo cai para ~2,4 ms por linha.
Valores exatos em `resultados.json`. **Compromisso:** com N = 25 a explicação fica ~3× mais barata, mas o erro sobe
para 22% e o Jaccard top-3 cai para 0,73; acima de 200 o ganho é pequeno. Limitação: medições de milissegundos
em uma máquina compartilhada são ruidosas, e 3 repetições só dão uma ordem de grandeza.

## 6. Crítica

* **Dados sintéticos:** o gerador fixa relações simples; as conclusões dizem respeito a este modelo nesta população,
  não a risco de crédito real.
* **Correlação entre variáveis:** receita, valor, saldo, despesas e dívida andam juntos. O SHAP intervencional avalia
  combinações que quebram essas relações e reparte o efeito entre features correlacionadas: no caso baixo,
  valor_solicitado (+1,21) e receita (−0,66) quase se anulam, e lidos isoladamente enganam. No caso alto,
  despesas fixas recebe +0,16, embora o gerador não use essa variável diretamente: o modelo a usa como substituta
  (proxy) da margem. A explicação descreve o modelo, não o mundo.
* **Referência:** muda o valor base e as contribuições; com amostras pequenas o ranking oscila.
* **Fidelidade x utilidade:** fidelidade alta no teste de remoção não garante que uma pessoa entenda ou tome melhores
  decisões; log-odds é pouco intuitivo, e a tradução "sem este fator" é aproximada.
* **Estabilidade:** boa para os valores, frágil para a ordem quando há empates.
* **Sem causalidade:** "atrasos aumenta o risco" significa que o modelo usa a variável assim, não que mudá-la mudaria o
  resultado real.

**Melhoria com mais tempo:** agrupar features correlacionadas (ex.: um grupo "porte financeiro" com receita, valor,
saldo, despesas e dívida) e explicar por grupos, ou usar uma referência condicional (vizinhos no treino) para evitar
combinações implausíveis, e comparar a fidelidade e a estabilidade resultantes.
