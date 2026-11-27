"""Etapa 1: escolher o tamanho da referência usando apenas a VALIDAÇÃO."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .configuracao import Configuracao
from .nucleo import Explicador, amostrar_referencia, jaccard_topk

log = logging.getLogger(__name__)


def escolher_referencia(modelo, manifesto, treino_X: pd.DataFrame, validacao_X: pd.DataFrame,
                        cfg: Configuracao) -> tuple[int, list[dict]]:
    """Compara referências pequenas com uma grande ("gabarito") e aplica a regra declarada:
    menor N com erro relativo médio <= limite e Jaccard top-3 médio >= limite."""
    rng = np.random.default_rng(cfg.semente)
    linhas = validacao_X.iloc[rng.choice(len(validacao_X), cfg.linhas_validacao, replace=False)]
    ouro = Explicador(modelo, manifesto, amostrar_referencia(treino_X, cfg.n_ouro, cfg.semente + 999),
                      cfg.threads)
    phi_ouro, _ = ouro.shap(linhas)
    escala = np.abs(phi_ouro).sum(axis=1)  # tamanho total de cada explicação "gabarito"

    tabela = []
    for n in cfg.n_candidatos:
        erros, jaccards, tempos = [], [], []
        for s in range(cfg.sementes_validacao):
            exp = Explicador(modelo, manifesto, amostrar_referencia(treino_X, n, cfg.semente + s),
                             cfg.threads)
            t0 = time.perf_counter()
            phi, _ = exp.shap(linhas)
            tempos.append(time.perf_counter() - t0)
            erros.append(np.mean(np.abs(phi - phi_ouro).sum(axis=1) / escala))
            jaccards.append(np.mean([jaccard_topk(a, b) for a, b in zip(phi, phi_ouro)]))
        linha = {
            "n_referencia": n,
            "erro_relativo_medio": float(np.mean(erros)),
            "jaccard_top3_medio": float(np.mean(jaccards)),
            "segundos_por_linha": float(np.mean(tempos) / cfg.linhas_validacao),
        }
        log.info("   N=%4d  erro relativo=%.3f  Jaccard top-3=%.3f  %.2f ms/linha", n,
                 linha["erro_relativo_medio"], linha["jaccard_top3_medio"],
                 1000 * linha["segundos_por_linha"])
        tabela.append(linha)
    aprovados = [r for r in tabela if r["erro_relativo_medio"] <= cfg.limite_erro_rel
                 and r["jaccard_top3_medio"] >= cfg.limite_jaccard]
    escolhido = aprovados[0]["n_referencia"] if aprovados else cfg.n_candidatos[-1]
    if not aprovados:
        log.warning("   nenhum N atendeu à regra; usando o maior candidato")
    return escolhido, tabela
