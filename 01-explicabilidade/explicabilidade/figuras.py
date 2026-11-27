"""Gráficos salvos na pasta de resultados."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera imagens sem precisar de tela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def figura_casos(casos_desc: dict, caminho: Path) -> None:
    fig, eixos = plt.subplots(1, len(casos_desc), figsize=(15, 4.5))
    for ax, (nome, d) in zip(eixos, casos_desc.items()):
        fatores = d["fatores"][::-1]
        cores = ["#c0392b" if f["contribuicao_logodds"] > 0 else "#2471a3" for f in fatores]
        ax.barh([f["feature"] for f in fatores], [f["contribuicao_logodds"] for f in fatores],
                color=cores)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Risco {nome}: p = {d['probabilidade']:.1%}\n(base {d['probabilidade_base']:.1%})")
        ax.set_xlabel("contribuição (log-odds)\nvermelho aumenta o risco, azul reduz")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def figura_referencia(tabela: list[dict], limite_erro: float, n_ouro: int, caminho: Path) -> None:
    df = pd.DataFrame(tabela)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["n_referencia"], df["erro_relativo_medio"], "o-", color="#c0392b")
    ax1.axhline(limite_erro, ls="--", color="#c0392b", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("tamanho da referência (linhas do treino)")
    ax1.set_ylabel(f"erro relativo vs. referência de {n_ouro}", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(df["n_referencia"], df["segundos_por_linha"] * 1000, "s-", color="#2471a3")
    ax2.set_ylabel("ms por linha explicada", color="#2471a3")
    ax1.set_title("Validação: qualidade x custo da referência")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)
