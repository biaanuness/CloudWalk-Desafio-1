"""Etapa 3: fidelidade das explicações ao modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .configuracao import Configuracao
from .nucleo import Explicador


def valor_removendo(exp: Explicador, x: pd.DataFrame, remover: list[int]) -> float:
    """Média da saída do modelo quando as features `remover` recebem os valores de cada linha
    da referência e as demais ficam como em x (a mesma ideia de "ausência" do SHAP
    intervencional)."""
    lote = pd.concat([x] * len(exp.referencia), ignore_index=True)
    for j in remover:
        nome = exp.features[j]
        lote[nome] = exp.referencia[nome].to_numpy()
    return float(exp.logodds(lote).mean())


def avaliar_fidelidade(exp: Explicador, X: pd.DataFrame, rng, cfg: Configuracao) -> dict:
    """(a) Aditividade: base + soma das contribuições = log-odds do modelo.
    (b) Remoção: tirar as top-k features deve mudar a saída mais que tirar k aleatórias,
        e a soma das contribuições removidas deve prever essa mudança."""
    phi, base = exp.shap(X)
    s = exp.logodds(X)
    aditividade = np.abs(base + phi.sum(axis=1) - s)

    registros = []
    for i in range(len(X)):
        x = X.iloc[[i]]
        ordem = np.argsort(-np.abs(phi[i]))
        for k in range(1, cfg.k_max + 1):
            top = list(ordem[:k])
            registros.append({"linha": i, "k": k, "tipo": "top", "previsto": phi[i, top].sum(),
                              "real": s[i] - valor_removendo(exp, x, top)})
            for _ in range(cfg.sorteios_aleatorios):
                aleat = list(rng.choice(len(exp.features), k, replace=False))
                registros.append({"linha": i, "k": k, "tipo": "aleatorio",
                                  "previsto": phi[i, aleat].sum(),
                                  "real": s[i] - valor_removendo(exp, x, aleat)})
    df = pd.DataFrame(registros)
    resumo = []
    for (k, tipo), g in df.groupby(["k", "tipo"]):
        resumo.append({
            "k": int(k), "tipo": tipo,
            "variacao_media_abs_logodds": float(g["real"].abs().mean()),
            "correlacao_previsto_real": float(np.corrcoef(g["previsto"], g["real"])[0, 1]),
            "erro_medio_abs": float((g["previsto"] - g["real"]).abs().mean()),
        })
    return {"linhas": len(X), "aditividade_erro_max": float(aditividade.max()),
            "remocao": resumo, "_df": df}
