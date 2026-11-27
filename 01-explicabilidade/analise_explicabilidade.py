"""Análise de explicabilidade local do classificador CatBoost de risco de crédito.

Método: SHAP intervencional para árvores (Independent Tree SHAP), calculado pelo
próprio CatBoost (`get_feature_importance(type="ShapValues", reference_data=...)`).
A explicação opera na escala de **log-odds** (saída bruta do modelo).

Etapas (cada uma corresponde a um item do enunciado):
  1. Escolha do tamanho da referência na VALIDAÇÃO (regra declarada antes do teste).
  2. Seleção dos três casos do TESTE (percentis 10, 50 e 90 da probabilidade prevista).
  3. Fidelidade: aditividade + teste de remoção (top-k vs. k aleatórias).
  4. Estabilidade: repetição, troca da amostra de referência e perturbação das entradas.
  5. Tempo: preparação e explicação por caso, com 3 repetições.

Execute a partir da raiz do repositório:
    uv run --frozen python 01-explicabilidade/analise_explicabilidade.py

Usa apenas as dependências fixadas em pyproject.toml/uv.lock. Resultados vão para
01-explicabilidade/resultados/.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import catboost
import matplotlib

matplotlib.use("Agg")  # gera imagens sem precisar de tela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
from predizer import carregar_modelo, preparar_features  # noqa: E402

SAIDA = BASE_DIR / "resultados"

# ---------------------------------------------------------------------------
# Parâmetros declarados ANTES de olhar o teste
# ---------------------------------------------------------------------------
SEED = 20261117               # mesma semente dos materiais
THREADS = 2                   # mesmo número de threads do treinamento
N_CANDIDATOS = [25, 50, 100, 200, 400]   # tamanhos de referência testados
N_OURO = 2000                 # referência grande usada como "gabarito" na validação
SEMENTES_VALIDACAO = 5        # amostras de referência por tamanho candidato
LINHAS_VALIDACAO = 200        # linhas da validação usadas na escolha
# Regra de escolha: menor N com erro relativo médio <= 10% e Jaccard top-3 >= 0,80
LIMITE_ERRO_REL = 0.10
LIMITE_JACCARD = 0.80
QUANTIS_CASOS = {"baixo": 0.10, "intermediario": 0.50, "alto": 0.90}
LINHAS_FIDELIDADE = 300       # amostra aleatória do teste para a fidelidade
K_MAX = 5                     # remove de 1 até 5 features
SORTEIOS_ALEATORIOS = 5       # conjuntos aleatórios de k features por linha
SEMENTES_REFERENCIA = 20      # estabilidade: trocas da amostra de referência
PERTURBACOES = 50             # estabilidade: perturbações por caso
REPETICOES_TEMPO = 3


def sigmoide(s):
    """Converte log-odds em probabilidade."""
    return 1.0 / (1.0 + np.exp(-s))


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    """Sobreposição entre as k features de maior |contribuição| de duas explicações."""
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / len(ta | tb)


class Explicador:
    """Agrupa o modelo, a referência e as funções que usamos o tempo todo."""

    def __init__(self, modelo, manifesto, referencia: pd.DataFrame):
        self.modelo = modelo
        self.manifesto = manifesto
        self.cat = manifesto["categorical_features"]
        self.features = manifesto["features"]
        self.referencia = referencia.reset_index(drop=True)
        self.pool_ref = self._pool(self.referencia)

    def _pool(self, X: pd.DataFrame) -> Pool:
        # O CatBoost exige um rótulo nos Pools para calcular SHAP intervencional,
        # mas não o usa no cálculo (conferimos: rótulos diferentes dão valores iguais).
        # Por isso passamos um rótulo fictício e nunca o alvo do teste.
        rotulo_ficticio = np.arange(len(X)) % 2
        return Pool(X, label=rotulo_ficticio, cat_features=self.cat)

    def logodds(self, X: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict(X, prediction_type="RawFormulaVal", thread_count=THREADS)

    def shap(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (contribuições [linhas x features], valor base [linhas])."""
        valores = self.modelo.get_feature_importance(
            self._pool(X), type="ShapValues", reference_data=self.pool_ref,
            thread_count=THREADS,
        )
        return valores[:, :-1], valores[:, -1]


def amostrar_referencia(treino_X: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    return treino_X.sample(n=n, random_state=semente)


# ---------------------------------------------------------------------------
# 1. Escolha do tamanho da referência (somente validação)
# ---------------------------------------------------------------------------
def escolher_referencia(modelo, manifesto, treino_X, validacao_X):
    rng = np.random.default_rng(SEED)
    linhas = validacao_X.iloc[rng.choice(len(validacao_X), LINHAS_VALIDACAO, replace=False)]
    ouro = Explicador(modelo, manifesto, amostrar_referencia(treino_X, N_OURO, SEED + 999))
    phi_ouro, _ = ouro.shap(linhas)
    escala = np.abs(phi_ouro).sum(axis=1)  # tamanho total de cada explicação "gabarito"

    tabela = []
    for n in N_CANDIDATOS:
        erros, jaccards, tempos = [], [], []
        for s in range(SEMENTES_VALIDACAO):
            exp = Explicador(modelo, manifesto, amostrar_referencia(treino_X, n, SEED + s))
            t0 = time.perf_counter()
            phi, _ = exp.shap(linhas)
            tempos.append(time.perf_counter() - t0)
            erros.append(np.mean(np.abs(phi - phi_ouro).sum(axis=1) / escala))
            jaccards.append(np.mean([jaccard_topk(a, b) for a, b in zip(phi, phi_ouro)]))
        tabela.append({
            "n_referencia": n,
            "erro_relativo_medio": float(np.mean(erros)),
            "jaccard_top3_medio": float(np.mean(jaccards)),
            "segundos_por_linha": float(np.mean(tempos) / LINHAS_VALIDACAO),
        })
    aprovados = [r for r in tabela
                 if r["erro_relativo_medio"] <= LIMITE_ERRO_REL and r["jaccard_top3_medio"] >= LIMITE_JACCARD]
    escolhido = aprovados[0]["n_referencia"] if aprovados else N_CANDIDATOS[-1]
    return escolhido, tabela


# ---------------------------------------------------------------------------
# 2. Seleção dos três casos
# ---------------------------------------------------------------------------
def selecionar_casos(teste: pd.DataFrame, prob: np.ndarray) -> dict[str, int]:
    """Para cada quantil, a linha cuja probabilidade prevista está mais próxima dele."""
    casos = {}
    for nome, q in QUANTIS_CASOS.items():
        alvo = np.quantile(prob, q)
        casos[nome] = int(np.argmin(np.abs(prob - alvo)))
    return casos


def descrever_caso(exp: Explicador, x: pd.DataFrame, top: int = 5) -> dict:
    phi, base = exp.shap(x)
    phi, base = phi[0], float(base[0])
    s = float(exp.logodds(x)[0])
    ordem = np.argsort(-np.abs(phi))[:top]
    fatores = []
    for j in ordem:
        nome = exp.features[j]
        fatores.append({
            "feature": nome,
            "valor": x.iloc[0][nome].item() if hasattr(x.iloc[0][nome], "item") else x.iloc[0][nome],
            "contribuicao_logodds": float(phi[j]),
            # Leitura aproximada: probabilidade se tirássemos só esta contribuição.
            "probabilidade_sem_este_fator": float(sigmoide(s - phi[j])),
        })
    return {
        "logodds_modelo": s,
        "probabilidade": float(sigmoide(s)),
        "valor_base_logodds": base,
        "probabilidade_base": float(sigmoide(base)),
        "erro_aditividade": float(abs(base + phi.sum() - s)),
        "fatores": fatores,
        "phi": phi,
    }


# ---------------------------------------------------------------------------
# 3. Fidelidade
# ---------------------------------------------------------------------------
def valor_removendo(exp: Explicador, x: pd.DataFrame, remover: list[int]) -> float:
    """Média da saída do modelo quando as features `remover` recebem os valores de
    cada linha da referência e as demais ficam como em x (mesma ideia de "ausência"
    usada pelo SHAP intervencional)."""
    lote = pd.concat([x] * len(exp.referencia), ignore_index=True)
    for j in remover:
        nome = exp.features[j]
        lote[nome] = exp.referencia[nome].to_numpy()
    return float(exp.logodds(lote).mean())


def avaliar_fidelidade(exp: Explicador, X: pd.DataFrame, rng) -> dict:
    phi, base = exp.shap(X)
    s = exp.logodds(X)
    aditividade = np.abs(base + phi.sum(axis=1) - s)

    registros = []
    for i in range(len(X)):
        x = X.iloc[[i]]
        ordem = np.argsort(-np.abs(phi[i]))
        for k in range(1, K_MAX + 1):
            top = list(ordem[:k])
            real_top = s[i] - valor_removendo(exp, x, top)
            registros.append({"linha": i, "k": k, "tipo": "top", "previsto": phi[i, top].sum(),
                              "real": real_top})
            for _ in range(SORTEIOS_ALEATORIOS):
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
    return {
        "linhas": len(X),
        "aditividade_erro_max": float(aditividade.max()),
        "remocao": resumo,
        "_df": df,
    }


# ---------------------------------------------------------------------------
# 4. Estabilidade
# ---------------------------------------------------------------------------
MONETARIAS = ["receita_mensal", "saldo_medio", "despesas_fixas_mensais", "divida_atual", "valor_solicitado"]
LIMITES = {
    "variacao_receita_3m": (-0.65, 0.65), "volatilidade_receita_6m": (0.025, 0.65),
    "margem_liquida": (-0.15, 0.50), "comprometimento_receita": (0.001, 0.90),
    "percentual_vendas_cartao": (0.03, 0.99), "taxa_chargeback_90d": (0.0, 0.20),
    "ticket_medio": (10, 2500),
}


def perturbar(x: pd.Series, rng) -> pd.Series:
    """Pequena mudança plausível que respeita o dicionário de dados.

    - Valores em reais (receita, saldo, despesas, dívida, valor solicitado) são
      multiplicados pelo MESMO fator entre 0,98 e 1,02: as razões entre eles
      (ex.: dívida/receita, valor/receita) ficam preservadas.
    - Frações positivas mudam até ~2% do próprio valor; frações com sinal
      (variação, margem) mudam +-0,005 em média. Tudo é cortado aos limites.
    - Meses de atividade e de relacionamento mudam -1, 0 ou +1, mantendo
      relacionamento <= atividade.
    - Contagens (atrasos, consultas), prazo e categorias NÃO mudam: qualquer
      alteração nelas é um salto discreto, não uma perturbação "ligeira".
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
    r = x["receita_mensal"]
    return bool(
        2000 <= r <= 300000
        and x["saldo_medio"] > 0
        and x["divida_atual"] <= 9 * r + 1e-6
        and 0.099 * r <= x["valor_solicitado"] <= 4.01 * r
        and x["relacionamento_meses"] <= x["tempo_atividade_meses"]
        and all(lo <= x[n] <= hi for n, (lo, hi) in LIMITES.items())
    )


def avaliar_estabilidade(modelo, manifesto, treino_X, exp: Explicador, casos_X: dict, rng) -> dict:
    resultado = {}
    for nome, x in casos_X.items():
        phi0, _ = exp.shap(x)
        phi0 = phi0[0]
        s0 = exp.logodds(x)[0]

        # (a) Repetição idêntica: mesmo caso, mesma referência.
        phi_rep, _ = exp.shap(x)
        dif_repeticao = float(np.abs(phi_rep[0] - phi0).max())

        # (b) Trocar a amostra de referência (mesmo tamanho, sementes diferentes).
        phis_ref = []
        for s in range(1, SEMENTES_REFERENCIA + 1):
            e = Explicador(modelo, manifesto, amostrar_referencia(treino_X, len(exp.referencia), SEED + 100 + s))
            phis_ref.append(e.shap(x)[0][0])
        phis_ref = np.array(phis_ref)
        top3 = np.argsort(-np.abs(phi0))[:3]

        # (c) Perturbar ligeiramente as entradas (mesma referência).
        linhas = []
        for _ in range(PERTURBACOES):
            y = perturbar(x.iloc[0], rng)
            assert checar_restricoes(y), "perturbação violou o dicionário de dados"
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
                "jaccard_top3_medio": float(np.mean([jaccard_topk(phi0, p) for p in phis_ref])),
                "jaccard_top3_min": float(np.min([jaccard_topk(phi0, p) for p in phis_ref])),
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


# ---------------------------------------------------------------------------
# 5. Tempo
# ---------------------------------------------------------------------------
def medir_tempo(treino_X, n_ref, casos_X) -> dict:
    preparo, por_caso, caminho = [], {k: [] for k in casos_X}, {k: [] for k in casos_X}
    for r in range(REPETICOES_TEMPO):
        t0 = time.perf_counter()
        modelo, manifesto = carregar_modelo()
        exp = Explicador(modelo, manifesto, amostrar_referencia(treino_X, n_ref, SEED))
        preparo.append(time.perf_counter() - t0)
        for nome, x in casos_X.items():
            t0 = time.perf_counter()
            exp.shap(x)
            por_caso[nome].append(time.perf_counter() - t0)
            # Comparação: TreeSHAP "dependente de caminho" (sem referência explícita)
            t0 = time.perf_counter()
            modelo.get_feature_importance(Pool(x, cat_features=exp.cat), type="ShapValues",
                                          thread_count=THREADS)
            caminho[nome].append(time.perf_counter() - t0)
    return {
        "preparacao_s": preparo,
        "explicacao_por_caso_s": por_caso,
        "treeshap_sem_referencia_por_caso_s": caminho,
    }


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_casos(casos_desc: dict, caminho: Path):
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (nome, d) in zip(eixos, casos_desc.items()):
        fatores = d["fatores"][::-1]
        cores = ["#c0392b" if f["contribuicao_logodds"] > 0 else "#2471a3" for f in fatores]
        ax.barh([f["feature"] for f in fatores], [f["contribuicao_logodds"] for f in fatores], color=cores)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"Risco {nome}: p = {d['probabilidade']:.1%}\n(base {d['probabilidade_base']:.1%})")
        ax.set_xlabel("contribuição (log-odds)\nvermelho aumenta o risco, azul reduz")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def figura_referencia(tabela: list[dict], caminho: Path):
    df = pd.DataFrame(tabela)
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["n_referencia"], df["erro_relativo_medio"], "o-", color="#c0392b")
    ax1.axhline(LIMITE_ERRO_REL, ls="--", color="#c0392b", lw=0.8)
    ax1.set_xscale("log")
    ax1.set_xlabel("tamanho da referência (linhas do treino)")
    ax1.set_ylabel("erro relativo vs. referência de 2000", color="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(df["n_referencia"], df["segundos_por_linha"] * 1000, "s-", color="#2471a3")
    ax2.set_ylabel("ms por linha explicada", color="#2471a3")
    ax1.set_title("Validação: qualidade x custo da referência")
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def para_json(obj):
    if isinstance(obj, dict):
        return {k: para_json(v) for k, v in obj.items() if not k.startswith("_") and k != "phi"}
    if isinstance(obj, list):
        return [para_json(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main() -> None:
    SAIDA.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    modelo, manifesto = carregar_modelo()
    treino = pd.read_csv(BASE_DIR / "dados" / "treino.csv")
    validacao = pd.read_csv(BASE_DIR / "dados" / "validacao.csv")
    teste = pd.read_csv(BASE_DIR / "dados" / "teste.csv")
    treino_X = preparar_features(treino, manifesto)
    validacao_X = preparar_features(validacao, manifesto)
    teste_X = preparar_features(teste, manifesto)

    print("1) Escolhendo o tamanho da referência na validação...")
    n_ref, tabela_ref = escolher_referencia(modelo, manifesto, treino_X, validacao_X)
    print(f"   escolhido: {n_ref} linhas do treino")
    exp = Explicador(modelo, manifesto, amostrar_referencia(treino_X, n_ref, SEED))

    print("2) Selecionando e explicando os três casos do teste...")
    prob_teste = sigmoide(exp.logodds(teste_X))
    idx = selecionar_casos(teste, prob_teste)
    casos_X = {nome: teste_X.iloc[[i]] for nome, i in idx.items()}
    casos_desc = {}
    for nome, i in idx.items():
        casos_desc[nome] = descrever_caso(exp, casos_X[nome])
        casos_desc[nome]["solicitacao_id"] = teste.loc[i, "solicitacao_id"]
        casos_desc[nome]["quantil_alvo"] = QUANTIS_CASOS[nome]
        # Comparação com TreeSHAP sem referência (dependente de caminho)
        phi_cam = modelo.get_feature_importance(Pool(casos_X[nome], cat_features=exp.cat),
                                                type="ShapValues", thread_count=THREADS)[0, :-1]
        casos_desc[nome]["spearman_vs_treeshap_sem_referencia"] = float(
            spearmanr(casos_desc[nome]["phi"], phi_cam).statistic)
        casos_desc[nome]["jaccard_top3_vs_treeshap_sem_referencia"] = jaccard_topk(
            casos_desc[nome]["phi"], phi_cam)
    figura_casos(casos_desc, SAIDA / "casos.png")

    print("3) Avaliando a fidelidade...")
    amostra = rng.choice(len(teste_X), LINHAS_FIDELIDADE, replace=False)
    fid = avaliar_fidelidade(exp, teste_X.iloc[amostra], rng)
    fid["aditividade_erro_max_teste_inteiro"] = float(np.abs(
        exp.shap(teste_X)[1] + exp.shap(teste_X)[0].sum(axis=1) - exp.logodds(teste_X)).max())

    print("4) Avaliando a estabilidade...")
    est = avaliar_estabilidade(modelo, manifesto, treino_X, exp, casos_X, rng)

    print("5) Medindo o tempo...")
    tempo = medir_tempo(treino_X, n_ref, casos_X)
    figura_referencia(tabela_ref, SAIDA / "referencia_validacao.png")

    resultados = {
        "ambiente": {
            "python": platform.python_version(), "plataforma": platform.platform(),
            "processador": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
            "catboost": catboost.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "threads": THREADS,
        },
        "parametros": {
            "semente": SEED, "n_candidatos": N_CANDIDATOS, "n_ouro": N_OURO,
            "sementes_validacao": SEMENTES_VALIDACAO, "linhas_validacao": LINHAS_VALIDACAO,
            "regra": f"menor N com erro relativo <= {LIMITE_ERRO_REL} e Jaccard top-3 >= {LIMITE_JACCARD}",
            "n_referencia_escolhido": n_ref, "linhas_fidelidade": LINHAS_FIDELIDADE,
            "k_max": K_MAX, "sorteios_aleatorios": SORTEIOS_ALEATORIOS,
            "sementes_referencia": SEMENTES_REFERENCIA, "perturbacoes": PERTURBACOES,
            "repeticoes_tempo": REPETICOES_TEMPO,
        },
        "escolha_referencia_validacao": tabela_ref,
        "casos": casos_desc,
        "fidelidade": fid,
        "estabilidade": est,
        "tempo": tempo,
    }
    (SAIDA / "resultados.json").write_text(
        json.dumps(para_json(resultados), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fid["_df"].to_csv(SAIDA / "fidelidade_remocao.csv", index=False)
    print(f"Pronto. Resultados em {SAIDA}")


if __name__ == "__main__":
    main()
