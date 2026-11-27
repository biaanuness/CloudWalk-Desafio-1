"""Etapa 5: tempo de execução (preparação separada da explicação por caso)."""

from __future__ import annotations

import time

import pandas as pd

from .configuracao import Configuracao
from .nucleo import Explicador, amostrar_referencia


def medir_tempo(carregar_modelo, treino_X: pd.DataFrame, n_ref: int,
                casos_X: dict[str, pd.DataFrame], cfg: Configuracao) -> dict:
    """Preparação = carregar o modelo + amostrar a referência + criar o Pool (sem ler CSVs).
    Explicação = uma chamada de SHAP para um caso. Cada medição é repetida."""
    preparo = []
    por_caso = {k: [] for k in casos_X}
    sem_ref = {k: [] for k in casos_X}
    for _ in range(cfg.repeticoes_tempo):
        t0 = time.perf_counter()
        modelo, manifesto = carregar_modelo()
        exp = Explicador(modelo, manifesto, amostrar_referencia(treino_X, n_ref, cfg.semente),
                         cfg.threads)
        preparo.append(time.perf_counter() - t0)
        for nome, x in casos_X.items():
            t0 = time.perf_counter()
            exp.shap(x)
            por_caso[nome].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            exp.shap_sem_referencia(x)
            sem_ref[nome].append(time.perf_counter() - t0)
    return {"preparacao_s": preparo, "explicacao_por_caso_s": por_caso,
            "treeshap_sem_referencia_por_caso_s": sem_ref}
