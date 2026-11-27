"""Carrega o modelo CatBoost nativo e calcula probabilidades de inadimplência."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).resolve().parent


def carregar_modelo(
    model_path: Path = BASE_DIR / "modelo" / "catboost.cbm",
    manifest_path: Path = BASE_DIR / "dados" / "features.json",
) -> tuple[CatBoostClassifier, dict]:
    """Utilize somente artefatos de origem conhecida; não usa pickle/joblib."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["id"] in manifest["features"] or manifest["target"] in manifest["features"]:
        raise ValueError("Identificador e alvo não podem fazer parte das features")
    model = CatBoostClassifier()
    model.load_model(str(model_path), format="cbm")
    if model.feature_names_ != manifest["features"]:
        raise ValueError("O modelo e o manifesto possuem features incompatíveis")
    return model, manifest


def preparar_features(frame: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """Seleciona explicitamente as features e preserva a ordem de treinamento."""
    missing = sorted(set(manifest["features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Features ausentes: {', '.join(missing)}")
    features = frame.loc[:, manifest["features"]].copy()
    if features.isna().any().any():
        raise ValueError("As features não podem conter valores ausentes")
    for name in manifest["numeric_features"]:
        features[name] = pd.to_numeric(features[name], errors="raise")
    numeric = features.loc[:, manifest["numeric_features"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("As features numéricas devem ser finitas")
    for name in manifest["categorical_features"]:
        if not features[name].isin(manifest["categorical_values"][name]).all():
            raise ValueError(f"Categoria desconhecida na feature {name}")
        features[name] = features[name].astype(str)
    return features


def predizer(frame: pd.DataFrame, model: CatBoostClassifier, manifest: dict) -> np.ndarray:
    features = preparar_features(frame, manifest)
    return model.predict_proba(features, thread_count=2)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BASE_DIR / "dados" / "teste.csv")
    parser.add_argument("--model", type=Path, default=BASE_DIR / "modelo" / "catboost.cbm")
    parser.add_argument("--manifest", type=Path, default=BASE_DIR / "dados" / "features.json")
    parser.add_argument("--output", type=Path, help="CSV de saída; sem esta opção imprime no terminal")
    parser.add_argument("--limit", type=int, default=5, help="Quantidade de linhas; 0 lê todas")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit deve ser maior ou igual a zero")
    frame = pd.read_csv(args.input, nrows=args.limit or None)
    if frame.empty:
        parser.error("O CSV de entrada não pode estar vazio")
    model, manifest = carregar_modelo(args.model, args.manifest)
    probabilities = predizer(frame, model, manifest)
    output = pd.DataFrame({
        "probabilidade_inadimplencia_90d": probabilities,
        "classe_limiar_0_5": (probabilities >= 0.5).astype(int),
    })
    if manifest["id"] in frame:
        output.insert(0, manifest["id"], frame[manifest["id"]].to_numpy())
    if args.output:
        if args.output.resolve() == args.input.resolve():
            parser.error("O arquivo de saída precisa ser diferente do arquivo de entrada")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Predições gravadas em {args.output}")
    else:
        print(output.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
