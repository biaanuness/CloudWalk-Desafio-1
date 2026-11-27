"""Etapa 4: estabilidade das explicações."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .configuracao import Configuracao
from .nucleo import Explicador, amostrar_referencia, jaccard_topk, sigmoide

MONETARIAS = ["receita_mensal", "saldo_medio", "despesas_fixas_mensais", "divida_atual",
              "valor_solicitado"]
LIMITES = {
    "variacao_receita_3m": (-0.65, 0.65), "volatilidade_receita_6m": (0.025, 0.65),
    "margem_liquida": (-0.15, 0.50), "comprometimento_receita": (0.001, 0.90),
    "percentual_vendas_cartao": (0.03, 0.99), "taxa_chargeback_90d": (0.0, 0.20),
    "ticket_medio": (10, 2500),
}


def perturbar(x: pd.Series, rng) -> pd.Series:
    """Pequena mudança plausível que respeita o dicionário de dados.

    - Valores em reais (receita, saldo, despesas, dívida, valor solicitado) são multiplicados
      pelo MESMO fator entre 0,98 e 1,02: as razões entre eles ficam preservadas.
    - Frações positivas e ticket mudam ~1% do próprio valor; frações com sinal (variação,
      margem) mudam +-0,005 em média. Tudo é cortado aos limites do dicionário.
    - Meses de atividade e de relacionamento mudam -1, 0 ou +1, com relacionamento <= atividade.
    - Contagens (atrasos, consultas), prazo e categorias NÃO mudam: alterá-los é um salto
      discreto, não uma perturbação "ligeira".
    """
    y = x.copy()
    fator = rng.uniform(0.98, 1.02)
    if not (2000 <= y["receita_mensal"] * fator <= 300000):
        fator = 1.0
    for nome in MONETARIAS:
        y[nome] = round(float(y[nome]) * fator, 2)
    for nome in ["volatilidade_receita_6m", "comprometimento_receita", "percentual_vendas_cartao",
                 "taxa_chargeback_90d", "ticket_medio"]:
        y[nome] = float(y[nome]) * (1 + rng.normal(0, 0.01))
    for nome in ["variacao_receita_3m", "margem_liquida"]:
        y[nome] = float(y[nome]) + rng.normal(0, 0.005)
    for nome, (lo, hi) in LIMITES.items():
        y[nome] = float(np.clip(y[nome], lo, hi))
    atividade = int(np.clip(y["tempo_atividade_meses"] + rng.integers(-1, 2), 6, 240))
    relacionamento = int(np.clip(y["relacionamento_meses"] + rng.integers(-1, 2), 0, atividade))
    y["tempo_atividade_meses"], y["relacionamento_meses"] = atividade, relacionamento
    return y


def checar_restricoes(x: pd.Series) -> bool:
    """Confere os domínios e relações explícitas do dicionário de dados."""
    r = x["receita_mensal"]
    return bool(
        2000 <= r <= 300000
        and x["saldo_medio"] > 0
        and x["divida_atual"] <= 9 * r + 1e-6
        and 0.099 * r <= x["valor_solicitado"] <= 4.01 * r
        and x["relacionamento_meses"] <= x["tempo_atividade_meses"]
        and all(lo <= x[n] <= hi for n, (lo, hi) in LIMITES.items())
    )


def avaliar_estabilidade(modelo, manifesto, treino_X: pd.DataFrame, exp: Explicador,
                         casos_X: dict[str, pd.DataFrame], rng, cfg: Configuracao) -> dict:
    resultado = {}
    for nome, x in casos_X.items():
        phi0 = exp.shap(x)[0][0]
        s0 = exp.logodds(x)[0]

        # (a) Repetição idêntica: mesmo caso, mesma referência.
        dif_repeticao = float(np.abs(exp.shap(x)[0][0] - phi0).max())

        # (b) Trocar a amostra de referência (mesmo tamanho, sementes diferentes).
        phis_ref = np.array([
            Explicador(modelo, manifesto,
                       amostrar_referencia(treino_X, len(exp.referencia), cfg.semente + 100 + s),
                       cfg.threads).shap(x)[0][0]
            for s in range(1, cfg.sementes_referencia + 1)
        ])
        top3 = np.argsort(-np.abs(phi0))[:3]
        jac_ref = [jaccard_topk(phi0, p) for p in phis_ref]

        # (c) Perturbar ligeiramente as entradas (mesma referência).
        linhas = []
        for _ in range(cfg.perturbacoes):
            y = perturbar(x.iloc[0], rng)
            if not checar_restricoes(y):
                raise AssertionError("perturbação violou o dicionário de dados")
            linhas.append(y)
        Xp = pd.DataFrame(linhas)[exp.features]
        for c in manifesto["categorical_features"]:
            Xp[c] = Xp[c].astype(str)
        phip, _ = exp.shap(Xp)
        sp = exp.logodds(Xp)
        rho = [spearmanr(phi0, p).statistic for p in phip]
        jac = [jaccard_topk(phi0, p) for p in phip]
        mud_rel = np.abs(phip - phi0).sum(axis=1) / np.abs(phi0).sum()
        resultado[nome] = {
            "repeticao_dif_max": dif_repeticao,
            "referencia": {
                "desvio_padrao_medio_top3": float(phis_ref[:, top3].std(axis=0).mean()),
                "contribuicao_media_abs_top3": float(np.abs(phi0[top3]).mean()),
                "jaccard_top3_medio": float(np.mean(jac_ref)),
                "jaccard_top3_min": float(np.min(jac_ref)),
            },
            "perturbacao": {
                "delta_prob_mediana_abs": float(np.median(np.abs(sigmoide(sp) - sigmoide(s0)))),
                "delta_prob_max_abs": float(np.max(np.abs(sigmoide(sp) - sigmoide(s0)))),
                "delta_logodds_max_abs": float(np.max(np.abs(sp - s0))),
                "spearman_mediano": float(np.median(rho)),
                "spearman_min": float(np.min(rho)),
                "jaccard_top3_medio": float(np.mean(jac)),
                "fracao_top3_identico": float(np.mean(np.array(jac) == 1.0)),
                "mudanca_relativa_l1_mediana": float(np.median(mud_rel)),
                "mudanca_relativa_l1_max": float(np.max(mud_rel)),
            },
        }
    return resultado
