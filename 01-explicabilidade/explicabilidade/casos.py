"""Etapa 2: selecionar e descrever os três casos do teste."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .nucleo import Explicador, jaccard_topk, sigmoide


def selecionar_casos(prob: np.ndarray, quantis: dict[str, float]) -> dict[str, int]:
    """Para cada quantil, a posição da linha cuja probabilidade prevista está mais perto dele.
    O rótulo observado NÃO é usado na seleção."""
    return {nome: int(np.argmin(np.abs(prob - np.quantile(prob, q)))) for nome, q in quantis.items()}


def descrever_caso(exp: Explicador, x: pd.DataFrame, top: int = 5) -> dict:
    """Previsão, valor base, as `top` maiores contribuições e o vetor completo (phi)."""
    phi, base = exp.shap(x)
    phi, base = phi[0], float(base[0])
    s = float(exp.logodds(x)[0])
    fatores = []
    for j in np.argsort(-np.abs(phi))[:top]:
        nome = exp.features[j]
        valor = x.iloc[0][nome]
        fatores.append({
            "feature": nome,
            "valor": valor.item() if hasattr(valor, "item") else valor,
            "contribuicao_logodds": float(phi[j]),
            # Leitura aproximada: probabilidade se retirássemos só esta contribuição.
            "probabilidade_sem_este_fator": float(sigmoide(s - phi[j])),
        })
    phi_cam = exp.shap_sem_referencia(x)[0]
    return {
        "logodds_modelo": s,
        "probabilidade": float(sigmoide(s)),
        "valor_base_logodds": base,
        "probabilidade_base": float(sigmoide(base)),
        "erro_aditividade": float(abs(base + phi.sum() - s)),
        "fatores": fatores,
        "spearman_vs_treeshap_sem_referencia": float(spearmanr(phi, phi_cam).statistic),
        "jaccard_top3_vs_treeshap_sem_referencia": jaccard_topk(phi, phi_cam),
        "phi": phi,
    }
