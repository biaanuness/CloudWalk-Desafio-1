# Desafio 1 — Explicabilidade de risco de crédito

Uma equipe precisa compreender as previsões de um modelo tabular de risco. Você recebe dados financeiros **inteiramente sintéticos** e um classificador CatBoost já treinado, que estima a probabilidade de inadimplência nos 90 dias seguintes à concessão de um crédito fictício.

O objetivo é investigar se uma explicação local representa bem o comportamento do modelo, permanece estável e pode ser produzida com um custo computacional razoável. Não é necessário treinar um modelo melhor. Após preparar o ambiente, dedique aproximadamente **4 a 6 horas** à análise e à entrega; declare simplificações e priorize uma análise pequena e reproduzível. A instalação inicial das ferramentas e dependências fica fora desse período.

**Conhecimentos esperados:** Python básico, manipulação de tabelas e noções de classificação e probabilidade. Não é necessário conhecer SHAP ou outro método de explicabilidade previamente, nem derivar ou implementar um método do zero. O tempo sugerido inclui consultar a documentação da biblioteca escolhida; registre o que conseguiu investigar dentro desse período.

## Sua tarefa

1. Escolha e aplique um método de explicação local. Justifique a escolha e diga em qual escala a explicação opera: probabilidade, log-odds ou outra saída. Bibliotecas existentes são permitidas.
2. Selecione **três casos** de teste com riscos previstos baixo, intermediário e alto. Defina seu critério de seleção. Para cada caso, apresente a previsão, os fatores que mais influenciam a explicação e uma interpretação compreensível para alguém que não conhece o método. Não trate o rótulo observado como se fosse a probabilidade correta de um indivíduo.
3. Avalie a **fidelidade** das explicações ao modelo com um critério quantitativo, deixando claro o que ele mede e o que não mede. Discuta as hipóteses necessárias para esse critério.
4. Investigue a **estabilidade** com um procedimento simples: repita a explicação e/ou perturbe ligeiramente as entradas. Preserve tipos, categorias, domínios e relações explícitas do dicionário e discuta a plausibilidade das alterações; não é necessário reconstruir toda a distribuição conjunta. Defina uma medida de comparação e descreva quando uma diferença é esperada ou problemática. Se apenas repetir um método determinístico, explique por que obter resultados idênticos não garante estabilidade a mudanças das entradas.
5. Meça o **tempo de execução**. Separe preparação e explicação por caso, faça repetições — por exemplo, três — e registre ambiente, parâmetros e tamanho da amostra de referência, quando houver. Discuta o compromisso entre custo e qualidade observado ou uma limitação dessa avaliação.
6. Faça uma **crítica** breve da abordagem: limitações dos dados sintéticos, correlação entre variáveis, escolha da referência quando aplicável, fidelidade versus utilidade, estabilidade e ausência de interpretação causal. Indique uma melhoria que investigaria com mais tempo.

Quando o método precisar de uma referência, construa-a com o treino. Se houver parâmetros a ajustar, use a validação e declare as escolhas antes da análise final do teste. Preserve o teste para os três casos finais e a avaliação que você definir. O código de geração é aberto para reprodutibilidade; explicar a fórmula geradora não substitui explicar o CatBoost entregue.

**Escopo suficiente para 4–6 horas:** um método, três casos, uma medida quantitativa de fidelidade, um procedimento simples de estabilidade e medições repetidas de tempo, acompanhados de interpretação e crítica. Comparar métodos, pesquisar vários tamanhos de referência ou explorar muitas formas de perturbação são **extensões opcionais**; não são necessárias para uma boa entrega. Resultados negativos ou limitações bem identificadas também são informativos.

## Conceitos usados no enunciado

- **Explicação local:** descrição dos fatores associados à previsão de uma solicitação específica. Uma *feature* é uma variável de entrada, como receita ou prazo.
- **Probabilidade e log-odds:** a probabilidade `p` vai de 0 a 1. Para `0 < p < 1`, log-odds é o escore `log(p / (1 - p))`. Para converter um escore `s` em probabilidade, use `1 / (1 + exp(-s))`. Uma contribuição em log-odds não representa pontos percentuais.
- **Referência:** dados ou um valor usados como comparação pelo método. Essa escolha pode mudar a explicação.
- **Fidelidade:** quanto a explicação representa a saída ou o comportamento do modelo segundo o critério escolhido. Reconstruir uma previsão, por exemplo, não demonstra que a explicação seja útil para uma pessoa.
- **Estabilidade:** quanto a explicação muda quando repetimos o procedimento ou fazemos pequenas mudanças válidas. Uma mudança na própria previsão pode justificar mudanças na explicação.

## Entrega esperada

- Um notebook ou script executável com instruções e dependências adicionais fixadas.
- Um relatório curto, de até 2 páginas ou equivalente no notebook, incluindo as três explicações, fidelidade, estabilidade, tempos e crítica.
- As sementes, os IDs escolhidos, os parâmetros e os comandos necessários para reproduzir os resultados.

A avaliação considera correção técnica, desenho dos experimentos, interpretação honesta dos resultados, clareza e reprodutibilidade. Não há requisito de atingir uma pontuação específica nem de usar uma biblioteca específica. Não fornecemos explicações prontas ou uma solução do exercício.

## Começar com os artefatos entregues

Execute os comandos a partir da **raiz do pacote**, um nível acima desta pasta. O ambiente base usa Python 3.12 e dependências fixadas em `pyproject.toml` e `uv.lock`.

```bash
uv sync --frozen
uv run --frozen python 01-explicabilidade/predizer.py
uv run --frozen python 01-explicabilidade/predizer.py --limit 0 --output /tmp/predicoes-credito.csv
```

O primeiro comando de inferência imprime cinco probabilidades; `--limit 0` processa o arquivo completo. O CSV de entrada pode omitir `solicitacao_id` e `inadimplente_90d`; somente as 19 features do manifesto entram no modelo. Variáveis categóricas devem manter os textos definidos no dicionário. A classificação com limiar 0,5 é apenas uma convenção ilustrativa.

Exemplo Python, executado dentro desta pasta:

```python
import pandas as pd
from predizer import carregar_modelo, predizer, preparar_features

modelo, manifesto = carregar_modelo()
casos = pd.read_csv("dados/teste.csv").head(3)
X = preparar_features(casos, manifesto)
probabilidades = predizer(casos, modelo, manifesto)
print(probabilidades)
```

`X` contém as features em ordem, sem identificador ou alvo; utilize esse contrato ao integrar sua biblioteca de explicabilidade. Instale ferramentas adicionais em um ambiente próprio e registre as versões: o ambiente base contém apenas as dependências dos materiais fornecidos.

## Arquivos

| Arquivo | Conteúdo |
| --- | --- |
| `dados/treino.csv` | 12.600 solicitações para treinamento e referência |
| `dados/validacao.csv` | 2.700 solicitações para escolhas do método |
| `dados/teste.csv` | 2.700 solicitações reservadas para análise final |
| `dados/features.json` | Lista ordenada de features, categorias e tipos |
| `dados/metadados.json` | Sementes, proporções e hashes dos CSVs |
| `modelo/catboost.cbm` | Classificador pronto em formato nativo CatBoost |
| `modelo/metricas.json` | Métricas, configuração, versões e hashes do treinamento |
| `DICIONARIO_DADOS.md` | Definições e restrições das variáveis |
| `MODEL_CARD.md` | Objetivo, treinamento, desempenho e limitações |
| `gerar_dados.py`, `treinar_modelo.py`, `predizer.py` | Geração, treinamento e inferência |
| `test_explicabilidade.py` | Verificações de integridade e reprodutibilidade |

Consulte também [DATASETS_PUBLICOS.md](DATASETS_PUBLICOS.md) para possíveis extensões com dados públicos. Elas não são necessárias para concluir este desafio.

## Reproduzir os materiais

Os artefatos já estão incluídos. Estes comandos os recriam e sobrescrevem os arquivos locais correspondentes:

```bash
uv run --frozen python 01-explicabilidade/gerar_dados.py
uv run --frozen python 01-explicabilidade/treinar_modelo.py
uv run --frozen python -m unittest discover -s 01-explicabilidade -p 'test_*.py' -v
```

Para experimentar sem alterar os materiais entregues:

```bash
uv run --frozen python 01-explicabilidade/gerar_dados.py --rows 3000 --seed 42 --output-dir /tmp/credito-experimento/dados
uv run --frozen python 01-explicabilidade/treinar_modelo.py --data-dir /tmp/credito-experimento/dados --output-dir /tmp/credito-experimento/modelo --seed 42
uv run --frozen python 01-explicabilidade/predizer.py --input /tmp/credito-experimento/dados/teste.csv --manifest /tmp/credito-experimento/dados/features.json --model /tmp/credito-experimento/modelo/catboost.cbm
```

Com as versões fixadas, a geração reproduz os CSVs byte a byte. O treinamento usa CPU e duas threads; pequenas diferenças numéricas entre plataformas são possíveis, e metadados internos do CatBoost podem alterar o hash do arquivo a cada treinamento. Os testes usam os artefatos existentes e não treinam novamente nem recalculam métricas do teste.
