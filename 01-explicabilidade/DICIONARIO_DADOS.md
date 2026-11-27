# Dicionário dos dados

Cada linha representa uma solicitação fictícia de crédito por um pequeno estabelecimento. Todos os campos de entrada são observados no momento da solicitação. Os dados não correspondem a pessoas, empresas ou decisões reais. Valores monetários usam **reais (BRL)** e o CSV usa ponto decimal.

| Campo | Tipo / unidade | Definição e domínio |
| --- | --- | --- |
| `solicitacao_id` | texto | Identificador sintético único. Não é feature. |
| `receita_mensal` | real / BRL | Receita mensal de referência; entre 2.000 e 300.000. |
| `variacao_receita_3m` | real / fração | Variação relativa da receita em três meses; entre −0,65 e 0,65. `−0.10` significa queda de 10%. |
| `volatilidade_receita_6m` | real / fração | Coeficiente de variação ilustrativo da receita em seis meses; entre 0,025 e 0,65. |
| `tempo_atividade_meses` | inteiro / meses | Tempo de funcionamento do estabelecimento; entre 6 e 240. |
| `margem_liquida` | real / fração | Margem líquida estimada; entre −0,15 e 0,50, podendo indicar prejuízo. |
| `saldo_medio` | real / BRL | Saldo médio disponível, positivo e relacionado a receita e margem. |
| `despesas_fixas_mensais` | real / BRL | Parcela de despesas fixas; entre aproximadamente 10% e 85% da receita. Não são todas as despesas do negócio. |
| `divida_atual` | real / BRL | Saldo total de dívidas anteriores à nova solicitação; até nove vezes a receita mensal. |
| `comprometimento_receita` | real / fração | Parcela da receita comprometida com prestações de dívidas existentes; entre 0,001 e 0,90. Não inclui o novo crédito. |
| `percentual_vendas_cartao` | real / fração | Participação do cartão na receita; entre 0,03 e 0,99. Apesar do nome, é uma fração: `0.70` significa 70%. |
| `taxa_chargeback_90d` | real / fração | Taxa ilustrativa de contestações nos 90 dias anteriores; entre 0 e 0,20. |
| `atrasos_12m` | inteiro / ocorrências | Número de ocorrências de atraso nos 12 meses anteriores; entre 0 e 12. |
| `consultas_credito_6m` | inteiro / consultas | Consultas de crédito anteriores à solicitação; entre 0 e 20. |
| `valor_solicitado` | real / BRL | Crédito solicitado; entre aproximadamente 0,1 e quatro receitas mensais. |
| `prazo_meses` | inteiro / meses | Prazo proposto: 6, 12, 18 ou 24 meses. |
| `ticket_medio` | real / BRL | Valor médio ilustrativo de uma venda; entre 10 e 2.500. |
| `relacionamento_meses` | inteiro / meses | Tempo de relacionamento com a instituição fictícia; não supera o tempo de atividade. |
| `setor` | categoria | `varejo`, `servicos`, `alimentacao`, `comercio_online`, `outros`. |
| `canal_origem` | categoria | `aplicativo`, `indicacao`, `parceiro`. |
| `inadimplente_90d` | inteiro binário | Alvo: 1 representa atraso superior a 30 dias observado nos 90 dias após o crédito; 0 representa ausência desse evento. Não é feature. |

São 19 features: 17 numéricas e duas categóricas. Não existem valores ausentes. As categorias são processadas nativamente pelo CatBoost. Não há identificadores pessoais, atributos protegidos, texto livre ou datas reais.

## Processo de geração e separação

`gerar_dados.py` usa uma semente fixa, distribuições lognormais, beta, normal e Poisson, limites explícitos e relações entre setor, receita, margem, saldo, dívida, atraso e solicitação. O alvo é uma amostra Bernoulli com probabilidade logística que combina variáveis, interações e ruído não observado. A probabilidade geradora e o ruído latente **não são exportados**. Algumas features têm pouco ou nenhum efeito direto na fórmula; todas são observáveis antes do alvo.

As separações são aleatórias e estratificadas pelo alvo, nas proporções 70%/15%/15%, com IDs exclusivos. Não há múltiplas solicitações da mesma entidade. Cada CSV inclui ID, features e alvo, sem uma coluna de split. O contrato de entrada é a lista explícita de `dados/features.json`; selecionar todas as colunas numéricas introduziria o alvo indevidamente.

Receita, saldo, dívida, valor solicitado e comprometimento estão relacionados. Alterar cada campo independentemente pode criar situações incompatíveis com a população sintética. Como não existe série histórica subjacente, as janelas de três, seis e doze meses são descritores ilustrativos. O conjunto é útil para estudar explicações e limitações de métodos, não para estimar risco real ou inferir causalidade.
