"""Gera um conjunto financeiro fictício, reproduzível e sem dados pessoais."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEED = 20261117
TARGET = "inadimplente_90d"
ID = "solicitacao_id"

FEATURES = [
    "receita_mensal", "variacao_receita_3m", "volatilidade_receita_6m",
    "tempo_atividade_meses", "margem_liquida", "saldo_medio",
    "despesas_fixas_mensais", "divida_atual", "comprometimento_receita",
    "percentual_vendas_cartao", "taxa_chargeback_90d", "atrasos_12m",
    "consultas_credito_6m", "valor_solicitado", "prazo_meses",
    "ticket_medio", "relacionamento_meses", "setor", "canal_origem",
]
CATEGORICAL_FEATURES = ["setor", "canal_origem"]
CATEGORIES = {
    "setor": ["varejo", "servicos", "alimentacao", "comercio_online", "outros"],
    "canal_origem": ["aplicativo", "indicacao", "parceiro"],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gerar_dados(rows: int = 18000, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Os fatores latentes e a probabilidade de geração nunca são exportados."""
    if rows < 1000 or rows > 1_000_000:
        raise ValueError("--rows deve estar entre 1000 e 1000000")
    if seed < 0 or seed > 2**32 - 2:
        raise ValueError("--seed deve estar entre 0 e 4294967294")
    rng = np.random.default_rng(seed)
    setor_idx = rng.choice(5, rows, p=[0.30, 0.27, 0.22, 0.15, 0.06])
    setor = np.array(CATEGORIES["setor"])[setor_idx]
    canal = rng.choice(CATEGORIES["canal_origem"], rows, p=[0.61, 0.22, 0.17])
    receita = np.clip(
        rng.lognormal(np.log(16000), 0.85, rows)
        * np.array([1.10, 0.90, 0.85, 1.30, 1.0])[setor_idx],
        2000, 300000,
    )
    atividade = np.clip(rng.gamma(2.3, 16, rows).astype(int) + 6, 6, 240)
    relacionamento = np.floor(atividade * rng.beta(1.5, 2.0, rows)).astype(int)
    variacao = np.clip(rng.normal(0.025, 0.17, rows), -0.65, 0.65)
    volatilidade = np.clip(
        0.025 + 0.5 * rng.beta(2, 5, rows) + 0.20 * np.abs(variacao), 0.025, 0.65
    )
    margem = np.clip(
        np.array([0.15, 0.28, 0.12, 0.19, 0.20])[setor_idx]
        + 0.15 * variacao + rng.normal(0, 0.075, rows), -0.15, 0.50
    )
    despesas = receita * np.clip(
        0.38 - 0.45 * margem + rng.normal(0, 0.09, rows), 0.10, 0.85
    )
    saldo = receita * np.maximum(0.01, 0.16 + 2.2 * margem + rng.normal(0, 0.22, rows))
    divida = receita * np.clip(rng.gamma(1.6, 0.8, rows), 0.01, 9)
    comprometimento = np.clip(
        divida / receita / rng.uniform(6, 24, rows), 0.001, 0.90
    )
    cartao = np.clip(
        np.array([0.70, 0.48, 0.68, 0.88, 0.55])[setor_idx]
        + rng.normal(0, 0.13, rows), 0.03, 0.99
    )
    chargeback = np.clip(
        rng.beta(1.2, 85, rows) * (1 + 0.8 * (setor_idx == 3)), 0, 0.20
    )
    atrasos = np.minimum(
        rng.poisson(0.18 + 1.5 * comprometimento + 2 * np.maximum(-variacao, 0)), 12
    )
    consultas = np.minimum(rng.poisson(1 + 7 * comprometimento + 0.3 * atrasos), 20)
    solicitado = receita * np.clip(rng.lognormal(np.log(0.9), 0.55, rows), 0.10, 4)
    prazo = rng.choice([6, 12, 18, 24], rows, p=[0.15, 0.40, 0.25, 0.20])
    ticket = np.clip(
        rng.lognormal(np.log(np.array([100, 190, 45, 145, 120])[setor_idx]), 0.5),
        10, 2500,
    )

    # Processo ilustrativo: ruído não observado impede a classificação perfeita.
    # Estas fórmulas não representam política de crédito ou relações causais reais.
    liquidez = saldo / receita
    logit = (
        -2.9 + 0.55 * divida / receita + 0.63 * atrasos
        + 3.5 * comprometimento - 2.0 * variacao + 1.8 * volatilidade
        - 1.8 * margem - 0.65 * liquidez + 0.48 * solicitado / receita
        + 6.0 * chargeback + 0.18 * (atividade < 18)
        - 0.20 * np.log1p(relacionamento / 12)
        + 0.35 * ((solicitado / receita > 1.5) & (liquidez < 0.35))
        + 0.25 * ((variacao < -0.15) & (atrasos > 0))
        + rng.normal(0, 0.60, rows)
    )
    probabilidade = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probabilidade)
    frame = pd.DataFrame({
        ID: [f"S{i:07d}" for i in range(1, rows + 1)],
        "receita_mensal": receita.round(2),
        "variacao_receita_3m": variacao.round(5),
        "volatilidade_receita_6m": volatilidade.round(5),
        "tempo_atividade_meses": atividade,
        "margem_liquida": margem.round(5),
        "saldo_medio": saldo.round(2),
        "despesas_fixas_mensais": despesas.round(2),
        "divida_atual": divida.round(2),
        "comprometimento_receita": comprometimento.round(5),
        "percentual_vendas_cartao": cartao.round(5),
        "taxa_chargeback_90d": chargeback.round(5),
        "atrasos_12m": atrasos,
        "consultas_credito_6m": consultas,
        "valor_solicitado": solicitado.round(2),
        "prazo_meses": prazo,
        "ticket_medio": ticket.round(2),
        "relacionamento_meses": relacionamento,
        "setor": setor,
        "canal_origem": canal,
        TARGET: target,
    })
    return frame


def separar_dados(frame: pd.DataFrame, seed: int = DEFAULT_SEED) -> dict[str, pd.DataFrame]:
    treino, restante = train_test_split(
        frame, test_size=0.30, stratify=frame[TARGET], random_state=seed
    )
    validacao, teste = train_test_split(
        restante, test_size=0.50, stratify=restante[TARGET], random_state=seed + 1
    )
    return {
        nome: dados.sort_values(ID).reset_index(drop=True)
        for nome, dados in [("treino", treino), ("validacao", validacao), ("teste", teste)]
    }


def salvar_dados(output_dir: Path, rows: int, seed: int) -> dict:
    frame = gerar_dados(rows, seed)
    splits = separar_dados(frame, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "id": ID,
        "target": TARGET,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_values": CATEGORIES,
        "numeric_features": [name for name in FEATURES if name not in CATEGORICAL_FEATURES],
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
    }
    (output_dir / "features.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": 1,
        "synthetic": True,
        "seed": seed,
        "rows": rows,
        "split_strategy": "estratificada aleatoria: treino 70%, validacao 15%, teste 15%",
        "split_seeds": [seed, seed + 1],
        "target_definition": "atraso superior a 30 dias observado nos 90 dias apos o credito",
        "splits": {},
    }
    for name, split in splits.items():
        path = output_dir / f"{name}.csv"
        split.to_csv(path, index=False, lineterminator="\n")
        metadata["splits"][name] = {
            "rows": len(split),
            "target_rate": float(split[TARGET].mean()),
            "sha256": sha256_file(path),
        }
    (output_dir / "metadados.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "dados")
    args = parser.parse_args()
    try:
        metadata = salvar_dados(args.output_dir, args.rows, args.seed)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
