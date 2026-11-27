"""Ponto de entrada: roda a análise de explicabilidade de ponta a ponta.

Execute a partir da raiz do repositório:
    uv run --frozen python 01-explicabilidade/analise_explicabilidade.py            # completo (~2 min)
    uv run --frozen python 01-explicabilidade/analise_explicabilidade.py --rapido   # demonstração (~15 s)

Etapas (cada uma é um módulo do pacote `explicabilidade/`):
  1. referencia    escolhe o tamanho da referência na VALIDAÇÃO (regra declarada antes do teste)
  2. casos         seleciona e explica 3 casos do TESTE (percentis 10, 50 e 90)
  3. fidelidade    aditividade + remoção top-k vs. k aleatórias
  4. estabilidade  repetição, troca da referência e perturbação das entradas
  5. tempo         preparação e explicação por caso, com repetições

Saídas na pasta escolhida (padrão `01-explicabilidade/resultados/`):
  execucao.log, resultados.json, explicacoes_casos.csv, fidelidade_remocao.csv,
  casos.png, referencia_validacao.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from explicabilidade.casos import descrever_caso, selecionar_casos  # noqa: E402
from explicabilidade.configuracao import Configuracao  # noqa: E402
from explicabilidade.estabilidade import avaliar_estabilidade  # noqa: E402
from explicabilidade.fidelidade import avaliar_fidelidade  # noqa: E402
from explicabilidade.figuras import figura_casos, figura_referencia  # noqa: E402
from explicabilidade.nucleo import Explicador, amostrar_referencia, sigmoide  # noqa: E402
from explicabilidade.referencia import escolher_referencia  # noqa: E402
from explicabilidade.tempo import medir_tempo  # noqa: E402
from predizer import carregar_modelo, preparar_features  # noqa: E402

log = logging.getLogger("explicabilidade")


def configurar_log(pasta: Path) -> None:
    """Mostra o progresso no terminal e grava o mesmo conteúdo em execucao.log."""
    formato = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    raiz = logging.getLogger("explicabilidade")
    raiz.setLevel(logging.INFO)
    raiz.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(pasta / "execucao.log", mode="w", encoding="utf-8")):
        handler.setFormatter(formato)
        raiz.addHandler(handler)


def para_json(obj):
    """Converte tipos do numpy e remove campos internos (começados por _ ou 'phi')."""
    if isinstance(obj, dict):
        return {k: para_json(v) for k, v in obj.items() if not k.startswith("_") and k != "phi"}
    if isinstance(obj, (list, tuple)):
        return [para_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapido", action="store_true",
                        help="versão reduzida para demonstração; não substitui os números do relatório")
    parser.add_argument("--saida", type=Path, default=None,
                        help="pasta de saída (padrão: resultados/ ou resultados_rapido/)")
    args = parser.parse_args()

    cfg = Configuracao().rapida() if args.rapido else Configuracao()
    saida = args.saida or BASE_DIR / ("resultados_rapido" if args.rapido else "resultados")
    saida.mkdir(parents=True, exist_ok=True)
    configurar_log(saida)
    inicio = time.perf_counter()
    log.info("Modo: %s | saída: %s", "rápido" if args.rapido else "completo", saida)

    rng = np.random.default_rng(cfg.semente)
    modelo, manifesto = carregar_modelo()
    dados = {n: pd.read_csv(BASE_DIR / "dados" / f"{n}.csv") for n in ["treino", "validacao", "teste"]}
    treino_X, validacao_X, teste_X = (preparar_features(dados[n], manifesto)
                                      for n in ["treino", "validacao", "teste"])
    log.info("Dados: treino=%d, validação=%d, teste=%d linhas; modelo com %d árvores",
             len(treino_X), len(validacao_X), len(teste_X), modelo.tree_count_)

    # 1. Referência --------------------------------------------------------------------------
    log.info("1) Escolhendo o tamanho da referência na VALIDAÇÃO (gabarito: %d linhas)", cfg.n_ouro)
    n_ref, tabela_ref = escolher_referencia(modelo, manifesto, treino_X, validacao_X, cfg)
    log.info("   escolhido: N=%d linhas do treino", n_ref)
    exp = Explicador(modelo, manifesto, amostrar_referencia(treino_X, n_ref, cfg.semente), cfg.threads)

    # 2. Casos -------------------------------------------------------------------------------
    log.info("2) Selecionando os três casos do TESTE pelos percentis da probabilidade prevista")
    prob_teste = sigmoide(exp.logodds(teste_X))
    idx = selecionar_casos(prob_teste, cfg.quantis_casos)
    casos_X = {nome: teste_X.iloc[[i]] for nome, i in idx.items()}
    casos_desc, linhas_csv = {}, []
    for nome, i in idx.items():
        d = descrever_caso(exp, casos_X[nome])
        d["solicitacao_id"] = dados["teste"].loc[i, "solicitacao_id"]
        d["quantil_alvo"] = cfg.quantis_casos[nome]
        casos_desc[nome] = d
        log.info("   %-13s %s  p=%.3f (base %.3f)  top: %s", nome, d["solicitacao_id"],
                 d["probabilidade"], d["probabilidade_base"],
                 ", ".join(f"{f['feature']} {f['contribuicao_logodds']:+.2f}" for f in d["fatores"][:3]))
        for j, feat in enumerate(exp.features):
            linhas_csv.append({"caso": nome, "solicitacao_id": d["solicitacao_id"], "feature": feat,
                               "valor": casos_X[nome].iloc[0][feat],
                               "contribuicao_logodds": d["phi"][j]})
    pd.DataFrame(linhas_csv).to_csv(saida / "explicacoes_casos.csv", index=False)
    figura_casos(casos_desc, saida / "casos.png")

    # 3. Fidelidade --------------------------------------------------------------------------
    log.info("3) Fidelidade em %d linhas aleatórias do teste (k = 1..%d)",
             cfg.linhas_fidelidade, cfg.k_max)
    amostra = rng.choice(len(teste_X), cfg.linhas_fidelidade, replace=False)
    fid = avaliar_fidelidade(exp, teste_X.iloc[amostra], rng, cfg)
    phi_t, base_t = exp.shap(teste_X)
    fid["aditividade_erro_max_teste_inteiro"] = float(
        np.abs(base_t + phi_t.sum(axis=1) - exp.logodds(teste_X)).max())
    log.info("   aditividade: erro máximo no teste inteiro = %.1e", fid["aditividade_erro_max_teste_inteiro"])
    for r in fid["remocao"]:
        log.info("   k=%d %-9s variação média |Δlog-odds|=%.3f  correlação previsto×real=%.3f",
                 r["k"], r["tipo"], r["variacao_media_abs_logodds"], r["correlacao_previsto_real"])

    # 4. Estabilidade ------------------------------------------------------------------------
    log.info("4) Estabilidade: %d referências alternativas e %d perturbações por caso",
             cfg.sementes_referencia, cfg.perturbacoes)
    est = avaliar_estabilidade(modelo, manifesto, treino_X, exp, casos_X, rng, cfg)
    for nome, e in est.items():
        log.info("   %-13s repetição Δ=%.1g | referência Jaccard=%.2f | perturbação Spearman min=%.3f, "
                 "top-3 idêntico=%.0f%%, Δprob máx=%.4f", nome, e["repeticao_dif_max"],
                 e["referencia"]["jaccard_top3_medio"], e["perturbacao"]["spearman_min"],
                 100 * e["perturbacao"]["fracao_top3_identico"], e["perturbacao"]["delta_prob_max_abs"])

    # 5. Tempo -------------------------------------------------------------------------------
    log.info("5) Tempo: %d repetições", cfg.repeticoes_tempo)
    tempo = medir_tempo(carregar_modelo, treino_X, n_ref, casos_X, cfg)
    log.info("   preparação: %s ms", ", ".join(f"{1000 * t:.1f}" for t in tempo["preparacao_s"]))
    for nome, ts in tempo["explicacao_por_caso_s"].items():
        log.info("   %-13s SHAP intervencional: %s ms | sem referência: %s ms", nome,
                 ", ".join(f"{1000 * t:.1f}" for t in ts),
                 ", ".join(f"{1000 * t:.1f}" for t in tempo["treeshap_sem_referencia_por_caso_s"][nome]))
    figura_referencia(tabela_ref, cfg.limite_erro_rel, cfg.n_ouro, saida / "referencia_validacao.png")

    resultados = {
        "modo": "rapido" if args.rapido else "completo",
        "ambiente": {
            "python": platform.python_version(), "plataforma": platform.platform(),
            "processador": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
            "catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "threads": cfg.threads,
        },
        "parametros": {**para_json(cfg.__dict__), "n_referencia_escolhido": n_ref,
                       "regra": f"menor N com erro relativo <= {cfg.limite_erro_rel} "
                                f"e Jaccard top-3 >= {cfg.limite_jaccard}"},
        "escolha_referencia_validacao": tabela_ref,
        "casos": casos_desc,
        "fidelidade": fid,
        "estabilidade": est,
        "tempo": tempo,
    }
    (saida / "resultados.json").write_text(
        json.dumps(para_json(resultados), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fid["_df"].to_csv(saida / "fidelidade_remocao.csv", index=False)
    log.info("Pronto em %.0f s. Resultados em %s", time.perf_counter() - inicio, saida)


if __name__ == "__main__":
    main()
