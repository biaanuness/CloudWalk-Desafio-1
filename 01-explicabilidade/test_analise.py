"""Testes rápidos da solução (pacote explicabilidade/). Rodam em poucos segundos."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from explicabilidade.casos import selecionar_casos
from explicabilidade.estabilidade import checar_restricoes, perturbar
from explicabilidade.nucleo import Explicador, amostrar_referencia, jaccard_topk
from gerar_dados import BASE_DIR
from predizer import carregar_modelo, preparar_features


class AnaliseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.modelo, cls.manifesto = carregar_modelo()
        treino = pd.read_csv(BASE_DIR / "dados" / "treino.csv")
        cls.treino_X = preparar_features(treino, cls.manifesto)
        cls.teste_X = preparar_features(pd.read_csv(BASE_DIR / "dados" / "teste.csv").head(50),
                                        cls.manifesto)
        cls.exp = Explicador(cls.modelo, cls.manifesto, amostrar_referencia(cls.treino_X, 50, 1))

    def test_aditividade(self) -> None:
        phi, base = self.exp.shap(self.teste_X)
        np.testing.assert_allclose(base + phi.sum(axis=1), self.exp.logodds(self.teste_X), atol=1e-9)

    def test_valor_base_e_media_da_referencia(self) -> None:
        _, base = self.exp.shap(self.teste_X.head(1))
        self.assertAlmostEqual(base[0], self.exp.logodds(self.exp.referencia).mean(), places=9)

    def test_deterministico(self) -> None:
        a, _ = self.exp.shap(self.teste_X.head(3))
        b, _ = self.exp.shap(self.teste_X.head(3))
        np.testing.assert_array_equal(a, b)

    def test_perturbacoes_respeitam_dicionario(self) -> None:
        rng = np.random.default_rng(0)
        for i in range(len(self.teste_X)):
            x = self.teste_X.iloc[i]
            self.assertTrue(checar_restricoes(x))
            y = perturbar(x, rng)
            self.assertTrue(checar_restricoes(y))
            for coluna in ["atrasos_12m", "consultas_credito_6m", "prazo_meses", "setor", "canal_origem"]:
                self.assertEqual(x[coluna], y[coluna])

    def test_selecao_de_casos_por_quantil(self) -> None:
        prob = np.linspace(0, 1, 101)
        self.assertEqual(selecionar_casos(prob, {"a": 0.1, "b": 0.9}), {"a": 10, "b": 90})

    def test_jaccard(self) -> None:
        self.assertEqual(jaccard_topk(np.array([3, 2, 1, 0]), np.array([-3, 2, 1, 0])), 1.0)
        self.assertEqual(jaccard_topk(np.array([3, 2, 1, 0, 0, 0]), np.array([0, 0, 0, 3, 2, 1])), 0.0)


if __name__ == "__main__":
    unittest.main()
