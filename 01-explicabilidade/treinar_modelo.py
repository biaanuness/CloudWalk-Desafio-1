"""Treina com validação para early stopping e avalia o teste reservado uma vez."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)

from gerar_dados import DEFAULT_SEED, sha256_file
from predizer import preparar_features

BASE_DIR = Path(__file__).resolve().parent


def calcular_metricas(target: pd.Series, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "average_precision": float(average_precision_score(target, probabilities)),
        "log_loss": float(log_loss(target, probabilities)),
        "brier_score": float(brier_score_loss(target, probabilities)),
        "accuracy_threshold_0_5": float(accuracy_score(target, predicted)),
        "precision_threshold_0_5": float(precision_score(target, predicted, zero_division=0)),
        "recall_threshold_0_5": float(recall_score(target, predicted, zero_division=0)),
        "f1_threshold_0_5": float(f1_score(target, predicted, zero_division=0)),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(
            target, predicted, labels=[0, 1]
        ).tolist(),
        "target_rate": float(target.mean()),
        "rows": len(target),
    }


def treinar(data_dir: Path, output_dir: Path, seed: int = DEFAULT_SEED) -> dict:
    manifest_path = data_dir / "features.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads((data_dir / "metadados.json").read_text(encoding="utf-8"))
    if manifest["id"] in manifest["features"] or manifest["target"] in manifest["features"]:
        raise ValueError("O manifesto inclui identificador ou alvo entre as features")
    frames = {}
    seen_ids: set[str] = set()
    for name in ["treino", "validacao", "teste"]:
        path = data_dir / f"{name}.csv"
        if sha256_file(path) != metadata["splits"][name]["sha256"]:
            raise ValueError(f"Hash divergente para {path}; gere novamente os dados e metadados")
        frame = pd.read_csv(path)
        ids = set(frame[manifest["id"]])
        if len(ids) != len(frame) or seen_ids.intersection(ids):
            raise ValueError("Identificadores duplicados ou presentes em múltiplos splits")
        seen_ids.update(ids)
        if set(frame[manifest["target"]].unique()) != {0, 1}:
            raise ValueError("Cada split precisa conter as duas classes binárias")
        frames[name] = frame

    pools = {
        name: Pool(
            preparar_features(frames[name], manifest),
            label=frames[name][manifest["target"]],
            cat_features=manifest["categorical_features"],
        )
        for name in ["treino", "validacao"]
    }
    parameters = {
        "iterations": 450,
        "depth": 6,
        "learning_rate": 0.06,
        "l2_leaf_reg": 5,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": seed,
        "thread_count": 2,
        "task_type": "CPU",
        "allow_writing_files": False,
    }
    model = CatBoostClassifier(**parameters)
    start = perf_counter()
    model.fit(
        pools["treino"], eval_set=pools["validacao"],
        early_stopping_rounds=45, use_best_model=True, verbose=False,
    )
    training_seconds = perf_counter() - start
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "catboost.cbm"
    model.save_model(str(model_path), format="cbm")

    # Holdout: sem seleção de hiperparâmetros ou limiar a partir deste resultado.
    test_features = preparar_features(frames["teste"], manifest)
    probabilities = model.predict_proba(test_features, thread_count=2)[:, 1]
    metrics = calcular_metricas(frames["teste"][manifest["target"]], probabilities)
    baseline_probability = float(frames["treino"][manifest["target"]].mean())
    baseline = np.full(len(frames["teste"]), baseline_probability)
    report = {
        "model_format": "catboost-cbm",
        "model_sha256": sha256_file(model_path),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "dataset_seed": metadata["seed"],
        "training_seed": seed,
        "parameters": parameters,
        "early_stopping_rounds": 45,
        "best_iteration_zero_based": model.get_best_iteration(),
        "tree_count": model.tree_count_,
        "feature_count": len(manifest["features"]),
        "training_rows": len(frames["treino"]),
        "validation_rows": len(frames["validacao"]),
        "training_seconds": training_seconds,
        "test": metrics,
        "constant_baseline_from_training_rate": {
            "probability": baseline_probability,
            "log_loss": float(log_loss(frames["teste"][manifest["target"]], baseline)),
            "brier_score": float(brier_score_loss(frames["teste"][manifest["target"]], baseline)),
        },
        "versions": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "catboost"]},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "data_sha256": {name: metadata["splits"][name]["sha256"] for name in frames},
    }
    (output_dir / "metricas.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "dados")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "modelo")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed deve estar entre 0 e 4294967295")
    report = treinar(args.data_dir, args.output_dir, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
