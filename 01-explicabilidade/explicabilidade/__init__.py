"""Pacote da solução: explicações locais (SHAP intervencional) do classificador CatBoost.

Módulos, na ordem em que a análise os usa:
  configuracao  -> parâmetros e sementes (modo completo e modo rápido)
  nucleo        -> classe Explicador (modelo + referência + SHAP) e funções auxiliares
  referencia    -> escolha do tamanho da referência na validação
  casos         -> seleção e descrição dos três casos do teste
  fidelidade    -> aditividade e teste de remoção top-k vs. aleatório
  estabilidade  -> repetição, troca de referência e perturbação das entradas
  tempo         -> medição de preparação e explicação por caso
  figuras       -> gráficos salvos em resultados/
"""
