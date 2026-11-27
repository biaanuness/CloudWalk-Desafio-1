"""Peças usadas por todas as etapas: o Explicador e funções auxiliares."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def sigmoide(s):
    """Converte log-odds em probabilidade: 1 / (1 + exp(-s))."""
    return 1.0 / (1.0 + np.exp(-s))


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    """Sobreposição (0 a 1) entre as k features de maior |contribuição| de duas explicações."""
    ta = set(np.argsort(-np.abs(a))[:k])
    tb = set(np.argsort(-np.abs(b))[:k])
    return len(ta & tb) / len(ta | tb)


def amostrar_referencia(treino_X: pd.DataFrame, n: int, semente: int) -> pd.DataFrame:
    """Amostra aleatória simples de n linhas do TREINO, usada como referência do SHAP."""
    return treino_X.sample(n=n, random_state=semente)


class Explicador:
    """Junta o modelo, a referência e as duas operações que usamos o tempo todo:
    a saída do modelo em log-odds e as contribuições SHAP intervencionais."""

    def __init__(self, modelo: CatBoostClassifier, manifesto: dict, referencia: pd.DataFrame,
                 threads: int = 2):
        self.modelo = modelo
        self.manifesto = manifesto
        self.cat = manifesto["categorical_features"]
        self.features = manifesto["features"]
        self.threads = threads
        self.referencia = referencia.reset_index(drop=True)
        self.pool_ref = self._pool(self.referencia)

    def _pool(self, X: pd.DataFrame) -> Pool:
        # O CatBoost exige um rótulo nos Pools para o SHAP intervencional, mas não o usa no
        # cálculo (conferido: rótulos diferentes dão valores idênticos). Usamos um rótulo
        # fictício e nunca o alvo do teste.
        return Pool(X, label=np.arange(len(X)) % 2, cat_features=self.cat)

    def logodds(self, X: pd.DataFrame) -> np.ndarray:
        """Saída bruta do modelo (log-odds)."""
        return self.modelo.predict(X, prediction_type="RawFormulaVal", thread_count=self.threads)

    def shap(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (contribuições [linhas x features], valor base [linhas]), em log-odds."""
        valores = self.modelo.get_feature_importance(
            self._pool(X), type="ShapValues", reference_data=self.pool_ref,
            thread_count=self.threads,
        )
        return valores[:, :-1], valores[:, -1]

    def shap_sem_referencia(self, X: pd.DataFrame) -> np.ndarray:
        """TreeSHAP 'dependente de caminho' (padrão do CatBoost), só para comparação."""
        valores = self.modelo.get_feature_importance(
            Pool(X, cat_features=self.cat), type="ShapValues", thread_count=self.threads)
        return valores[:, :-1]
