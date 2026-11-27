# Model card — Classificador de crédito sintético

## Finalidade e entrada

Classificador binário CatBoost para um exercício de explicabilidade. Estima `P(inadimplente_90d = 1 | features)` no processo sintético descrito no dicionário. O evento representa atraso superior a 30 dias observado nos 90 dias após a concessão fictícia.

Recebe 19 features, incluindo duas categóricas nativas, na ordem de `dados/features.json`. ID, alvo, split e probabilidade latente não entram no modelo. Não há imputação, normalização externa ou codificação one-hot a reconstruir. `predizer.py` seleciona e valida o contrato de entrada; rejeita ausências, números não finitos e categorias fora do domínio fornecido.

## Treinamento e avaliação

- Dados: 18.000 solicitações sintéticas; 12.600 de treino, 2.700 de validação e 2.700 de teste, estratificadas e disjuntas.
- Semente de geração e treinamento: `20261117`.
- Treinador: CatBoost 1.2.8, CPU, duas threads, até 450 árvores, profundidade 6, taxa de aprendizado 0,06 e regularização L2 igual a 5.
- Objetivo: log-loss. Seleção da quantidade de árvores: AUC na validação, com early stopping após 45 iterações sem melhora.
- O teste é avaliado uma vez ao final de cada execução de treinamento. Ele não seleciona hiperparâmetros, quantidade de árvores ou limiar. Reexecutar o treinamento reproduz a avaliação; não constitui um novo holdout.
- Baseline: probabilidade constante igual à prevalência no treino.

O artefato entregue tem **189 árvores**. O treinamento levou aproximadamente **1,24 s** em macOS arm64, usando Python 3.12.13; esse tempo não é uma promessa de desempenho em outro equipamento. As métricas do teste reservado são:

| Métrica | CatBoost | Baseline constante |
| --- | ---: | ---: |
| ROC AUC | 0,7919 | — |
| Average precision | 0,5853 | — |
| Log-loss | 0,4479 | 0,5566 |
| Brier score | 0,1441 | 0,1849 |
| Precisão com limiar 0,5 | 0,6656 | — |
| Recall com limiar 0,5 | 0,3192 | — |
| Acurácia com limiar 0,5 | 0,7941 | — |

A prevalência de inadimplência no teste é **24,48%**. Valores completos, matriz de confusão, parâmetros, versões e hashes estão em [`modelo/metricas.json`](modelo/metricas.json). Estas métricas descrevem classificação; a avaliação da qualidade das explicações faz parte do desafio.

O limiar 0,5 aparece apenas nas métricas ilustrativas e na CLI. Não foi otimizado para custos financeiros, aprovação de crédito ou capacidade operacional. AUC e average precision medem discriminação; log-loss e Brier avaliam a qualidade das probabilidades, mas não demonstram calibração em uma população real.

## Artefato e reprodução

O modelo está em `modelo/catboost.cbm`, formato nativo carregado por `CatBoostClassifier.load_model(..., format="cbm")`, sem pickle ou joblib. Use artefatos de origem conhecida e confira os hashes registrados em `modelo/metricas.json`. Metadados internos podem fazer o hash de um novo treinamento diferir mesmo quando as previsões são equivalentes.

O script de treinamento verifica os hashes dos CSVs, a exclusão de ID/alvo das features e a ausência de sobreposição dos IDs. Os testes conferem geração determinística, restrições financeiras, integridade dos artefatos, carregamento nativo e previsões finitas. Os comandos estão no README.

## Limitações e uso excluído

O conjunto foi construído para fins educacionais. A amostragem não é representativa de clientes reais, o mecanismo do alvo é artificial e as relações entre variáveis refletem escolhas do gerador. Não existem evidências de validade externa, equidade, robustez temporal ou conformidade para concessão de crédito.

Explicações descrevem o comportamento deste modelo sob hipóteses e referências escolhidas; não demonstram causas da inadimplência. Variáveis correlacionadas podem compartilhar ou deslocar atribuições. Perturbações podem sair da distribuição válida. A ausência de atributos protegidos não elimina proxies nem prova equidade.

O modelo não deve fundamentar decisões financeiras reais. Seu uso pretendido é comparar e criticar métodos de explicabilidade em um problema tabular controlado.
