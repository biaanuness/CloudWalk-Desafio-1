# Datasets públicos para explicabilidade de crédito

Fontes consultadas em **17 de novembro de 2026**. Os arquivos deste desafio são
sintéticos; os conjuntos abaixo são alternativas para uma edição com dados reais.
Metadados e resultados das consultas estão em `../docs/fontes-verificadas.json`.

| Dataset | Dimensão | Adequação ao desafio | Limitações |
|---|---|---|---|
| [Default of Credit Card Clients — UCI 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | 30.000 observações, 23 preditores | Inadimplência no mês seguinte; histórico de pagamentos, faturas e limite de crédito. É a opção recomendada para uma versão real do desafio. | Dados de Taiwan de 2005; unidades monetárias em dólares taiwaneses. Códigos inteiros incluem categorias, não apenas quantidades. |
| [Statlog German Credit — UCI 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1.000 observações, 20 preditores | Benchmark clássico de classificação de risco, pequeno e rápido para comparar explicadores. | Poucos exemplos, dados antigos e problemas de descrição/codificação registrados pela versão corrigida abaixo. A classe original usa `1=good`, `2=bad`. |
| [South German Credit — UCI 573](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) | 1.000 observações; 20 preditores e um alvo | Correção documentada do German Credit. Preferível ao original quando o objetivo é interpretar o significado das variáveis. | Créditos de 1973–1975, com casos de crédito ruim sobreamostrados: 700 bons e 300 ruins. A proporção de classes não estima a prevalência de uma população atual. |

## Escolha sugerida

Use **Default of Credit Card Clients** para oferecer mais exemplos e diversidade
de perfis. Use **South German Credit** quando a prioridade for baixo custo de
execução e discussão detalhada de variáveis categóricas.

O cartão do UCI de South German informa 21 features, mas a própria descrição
especifica **20 preditores mais a variável de resposta**. O alvo nunca deve ser
incluído nas entradas do modelo. South German e German Credit representam os
mesmos créditos, não devem ser combinados como fontes independentes nem usados
como treino e teste um do outro.

## Como adaptar o desafio

1. Baixe os dados pela página oficial e registre a versão e o DOI.
2. Identifique alvo, identificadores e categorias pelo dicionário original.
   Para o UCI 350, `Y` é inadimplência e `ID` é identificador; os preditores são
   `X1` a `X23`. A transformação de categorias deve ser explícita.
3. Faça as partições antes de ajustar qualquer transformação. Use treino para
   aprender, validação para escolher a configuração e teste para a avaliação.
4. Treine um novo CatBoost com o esquema do dataset escolhido. O `.cbm` deste
   pacote foi treinado no esquema sintético e não aceita os dados públicos
   diretamente.
5. Adapte as regras de perturbação às unidades e relações do dataset. Variáveis
   demográficas presentes nas bases reais permitem discutir limitações das
   explicações e possíveis associações indevidas.

Essas bases não fornecem uma explicação causal verdadeira por observação.
Fidelidade e estabilidade precisam ser avaliadas em relação ao comportamento
do modelo, com as hipóteses do experimento documentadas.

## Acesso, atribuição e referências

Os arquivos abaixo responderam com HTTP 200 **sem autenticação** na verificação
de 17/11/2026. O download é simples; interpretar os códigos e treinar o modelo
adequado à base continuam sendo etapas de preparação da equipe.

| Dataset | Arquivo oficial pronto para baixar | Formato |
|---|---|---|
| Default of Credit Card Clients | [Download CSV](https://archive.ics.uci.edu/static/public/350/data.csv) | CSV com cabeçalho; usar `Y` como alvo e excluir `ID`. |
| German Credit | [Download CSV](https://archive.ics.uci.edu/static/public/144/data.csv) | CSV com códigos categóricos; consultar o dicionário. |
| South German Credit | [Download ZIP](https://archive.ics.uci.edu/static/public/573/south+german+credit+update.zip) | 13.130 bytes; contém `SouthGermanCredit.asc`, tabela de códigos e exemplo em R. O arquivo ASC é uma tabela de texto. |

O ZIP de South German foi baixado integralmente e teve sua integridade conferida.
Os dois CSVs tiveram o acesso e o início do conteúdo verificados. A API de South
German não oferece `data_url`; use o ZIP, disponível na página oficial.
O registro está em [fontes-acesso.json](../docs/fontes-acesso.json).

As três páginas do UCI exibiam **CC BY 4.0** na consulta. Ao redistribuir uma
versão real, mantenha a atribuição aos autores, o DOI, a licença e a descrição
das transformações realizadas.

- Taiwan: I-Cheng Yeh, DOI [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
  [CSV disponibilizado pelo UCI](https://archive.ics.uci.edu/static/public/350/data.csv).
- German Credit: Hans Hofmann, DOI [10.24432/C5NC77](https://doi.org/10.24432/C5NC77).
- South German Credit: DOI [10.24432/C5QG88](https://doi.org/10.24432/C5QG88).
  A página oficial descreve os erros de codificação da distribuição antiga.

Foram verificadas as páginas e as APIs oficiais. As bases reais não foram usadas
no treinamento nem redistribuídas neste pacote.
