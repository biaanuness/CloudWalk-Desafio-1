"""Verificações dos materiais, sem implementar as explicações do desafio."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from gerar_dados import BASE_DIR, DEFAULT_SEED, FEATURES, ID, TARGET, salvar_dados, sha256_file
from predizer import carregar_modelo, predizer, preparar_features


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data_dir = BASE_DIR / "dados"
        cls.frames = {
            name: pd.read_csv(cls.data_dir / f"{name}.csv")
            for name in ["treino", "validacao", "teste"]
        }
        cls.model, cls.manifest = carregar_modelo()

    def test_generated_artifacts_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            salvar_dados(target, rows=18000, seed=DEFAULT_SEED)
            for name in ["treino.csv", "validacao.csv", "teste.csv", "features.json", "metadados.json"]:
                self.assertEqual(sha256_file(target / name), sha256_file(self.data_dir / name), name)

    def test_splits_are_disjoint_stratified_and_complete(self) -> None:
        seen: set[str] = set()
        for name, frame in self.frames.items():
            ids = set(frame[ID])
            self.assertEqual(len(ids), len(frame))
            self.assertFalse(seen.intersection(ids), name)
            seen.update(ids)
            self.assertEqual(set(frame[TARGET].unique()), {0, 1})
        self.assertEqual(len(seen), 18000)
        self.assertEqual([len(frame) for frame in self.frames.values()], [12600, 2700, 2700])
        rates = [frame[TARGET].mean() for frame in self.frames.values()]
        self.assertLess(max(rates) - min(rates), 0.002)

    def test_financial_constraints_and_feature_isolation(self) -> None:
        frame = pd.concat(self.frames.values(), ignore_index=True)
        self.assertFalse(frame.isna().any().any())
        self.assertNotIn(ID, FEATURES)
        self.assertNotIn(TARGET, FEATURES)
        self.assertEqual(list(frame.columns), [ID, *FEATURES, TARGET])
        self.assertEqual(self.model.feature_names_, FEATURES)
        self.assertTrue(np.isfinite(frame[self.manifest["numeric_features"]]).all().all())
        self.assertTrue(frame["receita_mensal"].between(2000, 300000).all())
        self.assertTrue((frame["relacionamento_meses"] <= frame["tempo_atividade_meses"]).all())
        self.assertTrue((frame["relacionamento_meses"] >= 0).all())
        self.assertTrue((frame["despesas_fixas_mensais"] < frame["receita_mensal"]).all())
        self.assertTrue((frame["divida_atual"] <= 9 * frame["receita_mensal"] + 0.05).all())
        self.assertTrue((frame["valor_solicitado"] <= 4 * frame["receita_mensal"] + 0.03).all())
        self.assertTrue(frame["percentual_vendas_cartao"].between(0, 1).all())
        self.assertTrue(frame["taxa_chargeback_90d"].between(0, 0.2).all())
        self.assertTrue(frame["prazo_meses"].isin([6, 12, 18, 24]).all())

    def test_native_model_roundtrip_and_irrelevant_columns(self) -> None:
        frame = self.frames["teste"].head(100).copy()
        expected = predizer(frame, self.model, self.manifest)
        self.assertTrue(np.isfinite(expected).all())
        self.assertTrue(((expected >= 0) & (expected <= 1)).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modelo.cbm"
            self.model.save_model(str(path), format="cbm")
            loaded = CatBoostClassifier()
            loaded.load_model(str(path), format="cbm")
            np.testing.assert_array_equal(expected, predizer(frame, loaded, self.manifest))
        frame[TARGET] = 1 - frame[TARGET]
        frame[ID] = "identificador_sem_efeito"
        np.testing.assert_array_equal(
            expected, predizer(frame[frame.columns[::-1]], self.model, self.manifest)
        )

    def test_inference_rejects_invalid_features(self) -> None:
        frame = self.frames["teste"].head(2).copy()
        with self.assertRaises(ValueError):
            preparar_features(frame.drop(columns=["receita_mensal"]), self.manifest)
        for invalid in [np.nan, np.inf]:
            invalid_frame = frame.copy()
            invalid_frame.loc[invalid_frame.index[0], "receita_mensal"] = invalid
            with self.assertRaises(ValueError):
                preparar_features(invalid_frame, self.manifest)
        frame.loc[frame.index[0], "setor"] = "categoria_inexistente"
        with self.assertRaises(ValueError):
            preparar_features(frame, self.manifest)

    def test_metric_report_matches_artifacts(self) -> None:
        report = json.loads((BASE_DIR / "modelo" / "metricas.json").read_text())
        self.assertEqual(report["model_sha256"], sha256_file(BASE_DIR / "modelo" / "catboost.cbm"))
        self.assertEqual(report["feature_manifest_sha256"], sha256_file(self.data_dir / "features.json"))
        self.assertEqual(report["test"]["rows"], len(self.frames["teste"]))
        self.assertGreater(report["test"]["roc_auc"], 0.5)
        self.assertLess(
            report["test"]["log_loss"], report["constant_baseline_from_training_rate"]["log_loss"]
        )
        for name in self.frames:
            self.assertEqual(report["data_sha256"][name], sha256_file(self.data_dir / f"{name}.csv"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
