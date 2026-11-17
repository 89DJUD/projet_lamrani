import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import pandas as pd
from migrant_ai.fairness import detect_bias, LABELS
from migrant_ai.footprint import EnergyMonitor
from migrant_ai.experiments import summarize

class BiasTests(unittest.TestCase):
    def test_neutral_rights_are_not_flagged(self):
        self.assertFalse(detect_bias("Les refugies peuvent demander un titre de sejour.")["bias_flag"])
    def test_negative_generalization(self):
        self.assertEqual(detect_bias("Tous les migrants sont dangereux.")["stereotype_score"], 1)
    def test_inclusive_general_statement_is_not_negative(self):
        self.assertFalse(detect_bias("Tous les migrants sont les bienvenus.")["bias_flag"])
    def test_hate_signal(self):
        self.assertEqual(detect_bias("Il faut expulser tous les migrants.")["hate_score"], 1)
    def test_invalid_inputs(self):
        for kwargs in ({"method": "unknown"}, {"threshold": 2}):
            with self.assertRaises(ValueError): detect_bias("Texte", **kwargs)
    def test_model_preserves_class_mapping(self):
        classifier = MagicMock(return_value={"labels": list(reversed(list(LABELS.values()))), "scores": [0.8, 0.1]})
        with patch("migrant_ai.fairness.load_classifier", return_value=classifier):
            result = detect_bias("Une reponse.", "model")
        self.assertEqual(result["hate_score"], 0.8)
        self.assertEqual(result["stereotype_score"], 0.1)
    def test_long_model_text_covers_tail(self):
        classifier = MagicMock(return_value={"labels": list(LABELS.values()), "scores": [0.1, 0.2]})
        with patch("migrant_ai.fairness.load_classifier", return_value=classifier):
            detect_bias("mot " * 350, "model")
        self.assertGreater(classifier.call_count, 1)

class EnergyTests(unittest.TestCase):
    def test_disabled_has_no_fabricated_zero(self):
        with EnergyMonitor(False) as monitor: pass
        self.assertIsNone(monitor.metrics["energy_kwh"])
        self.assertGreaterEqual(monitor.metrics["seconds"], 0)
    def test_sensor_startup_is_not_generation_latency(self):
        with patch("migrant_ai.footprint.time.perf_counter",side_effect=[0,5,7]):
            with EnergyMonitor(False) as monitor: pass
        self.assertEqual(monitor.metrics["seconds"],2)
    def test_machine_scope_and_units(self):
        tracker = MagicMock()
        tracker.stop.return_value = 0.002
        tracker.final_emissions_data.energy_consumed = 0.01
        factory = MagicMock(return_value=tracker)
        with patch.dict(sys.modules, {"codecarbon": SimpleNamespace(OfflineEmissionsTracker=factory)}):
            with EnergyMonitor() as monitor: pass
        self.assertEqual(factory.call_args.kwargs["tracking_mode"], "machine")
        self.assertEqual(monitor.metrics["co2_kg"], 0.002)
        self.assertEqual(monitor.metrics["energy_kwh"], 0.01)
    def test_failure_does_not_hide_generation_error(self):
        tracker = MagicMock()
        tracker.stop.side_effect = RuntimeError("Unavailable")
        with patch.dict(sys.modules, {"codecarbon": SimpleNamespace(OfflineEmissionsTracker=lambda **kw: tracker)}):
            with self.assertRaisesRegex(ValueError, "generation"):
                with EnergyMonitor() as monitor:
                    raise ValueError("generation")
        tracker.stop.assert_called_once()
        self.assertEqual(monitor.metrics["energy_method"], "unavailable")
    def test_missing_package_is_reported(self):
        with patch.dict(sys.modules, {"codecarbon": None}):
            with EnergyMonitor() as monitor: pass
        self.assertIsNone(monitor.metrics["co2_kg"])
        self.assertEqual(monitor.metrics["energy_method"], "unavailable")

class EvaluationTests(unittest.TestCase):
    def test_identical_and_unrelated_metrics(self):
        from migrant_ai.quality import compute_bleu, compute_rouge
        text = "Les refugies ont le droit de travailler."
        self.assertAlmostEqual(compute_bleu(text, text), 1)
        self.assertEqual(compute_rouge(text, text)["rougeL"], 1)
        self.assertEqual(compute_rouge("chat", "avion")["rouge1"], 0)
    def test_french_unicode_tokens(self):
        from migrant_ai.quality import _FrenchTokenizer
        self.assertEqual(_FrenchTokenizer().tokenize("R\u00e9fugi\u00e9s \u00e9trangers"), ["r\u00e9fugi\u00e9s", "\u00e9trangers"])
    def test_dataset_context_progress_and_metadata(self):
        from migrant_ai.quality import evaluate_test_set
        callback = MagicMock()
        generate = MagicMock(return_value="Les enfants mineurs peuvent rejoindre leurs parents.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.csv"
            pd.DataFrame([{"question": "Qui peut venir ?", "reference": "Les enfants mineurs."}]).to_csv(path, index=False)
            with patch("migrant_ai.quality.compute_semantic_similarity", return_value=0.8):
                frame = evaluate_test_set(path, generate, callback, measure_energy=False)
        generate.assert_called_once_with("Qui peut venir ?", None)
        self.assertEqual(callback.call_args.args, (1,1))
        self.assertTrue(pd.isna(frame.loc[0,"energy_kwh"]))
        self.assertEqual(frame.loc[0,"bias_method"], "heuristic")
    def test_small_run_limits_generation_calls(self):
        from migrant_ai.quality import evaluate_test_set
        generate = MagicMock(return_value="Une reponse valide.")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"test.csv"
            pd.DataFrame([{"question":"Q1", "reference":"Une reponse."},
                          {"question":"Q2", "reference":"Une reponse."}]).to_csv(path,index=False)
            with patch("migrant_ai.quality.compute_semantic_similarity",return_value=0.7):
                result=evaluate_test_set(path,generate,measure_energy=False,limit=1)
        self.assertEqual(len(result),1)
        self.assertEqual(generate.call_count,1)
    def test_dataset_validation(self):
        from migrant_ai.quality import evaluate_test_set
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.csv"
            pd.DataFrame([{"question":""}]).to_csv(path,index=False)
            with self.assertRaises(ValueError): evaluate_test_set(path, MagicMock())
    def test_summary_counts_available_energy(self):
        frame=pd.DataFrame([dict(model="llama3:8b",chunk_size=50,top_k=2,temperature=0.1,
            bleu=0.2,energy_kwh=None,co2_kg=None),dict(model="llama3:8b",chunk_size=50,top_k=2,temperature=0.1,
            bleu=0.8,energy_kwh=None,co2_kg=None)])
        summary=summarize(frame)
        self.assertAlmostEqual(summary.loc[0,"bleu_median"],0.5)
        self.assertEqual(summary.loc[0,"n_energy_available"],0)
        self.assertTrue(pd.isna(summary.loc[0,"co2_kg_median"]))

class LlamaTests(unittest.TestCase):
    def test_no_context_skips_model(self):
        from migrant_ai import generation as llm
        with patch.object(llm._ollama,"generate") as generate:
            self.assertIn("disponible",llm.generate_response("Question", ""))
        generate.assert_not_called()
    def test_light_variant_has_explicit_model_and_context(self):
        from migrant_ai import generation as llm
        with patch.object(llm._ollama,"generate",return_value={"response":"Reponse"}) as generate:
            llm.generate_response("Question","Contexte",llm.LLAMA_LIGHT_MODEL)
        self.assertEqual(generate.call_args.kwargs["model"],llm.LLAMA_LIGHT_MODEL)
        self.assertEqual(generate.call_args.kwargs["options"]["num_ctx"],2048)
    def test_llama_options_and_default(self):
        from migrant_ai import generation as llm
        with patch.object(llm._ollama,"generate",return_value={"response":" Reponse "}) as generate:
            self.assertEqual(llm.generate_response("Question", "Contexte"),"Reponse")
        self.assertEqual(generate.call_args.kwargs["model"],"llama3:8b")
        self.assertEqual(generate.call_args.kwargs["options"]["seed"],42)



class PreservationTests(unittest.TestCase):
    def test_every_original_file_is_unchanged(self):
        import hashlib, json
        root = Path(__file__).resolve().parents[1]
        payload = json.loads((root/"tests/fixtures/source_manifest.json").read_text(encoding="utf-8"))
        manifest = payload["files"]
        self.assertGreater(len(manifest),0)
        for name,digest in manifest.items():
            with self.subTest(file=name):
                data=(root/name).read_bytes()
                if Path(name).suffix in payload["text_extensions"]: data=data.replace(b"\r\n",b"\n")
                self.assertEqual(hashlib.sha256(data).hexdigest(),digest)

class DiagnosticTests(unittest.TestCase):
    def test_diagnostic_checks_original_sources(self):
        from migrant_ai.diagnostic import inspect_project
        with patch("migrant_ai.diagnostic.urllib.request.urlopen",side_effect=OSError("offline")):
            report=inspect_project()
        self.assertTrue(report["originals_unchanged"])
        self.assertEqual(report["changed_originals"],[])

class ProvenanceTests(unittest.TestCase):
    def test_actual_context_does_not_replace_reference_source(self):
        from migrant_ai.quality import evaluate_test_set
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"test.csv"
            pd.DataFrame([{"question":"Question", "reference":"Une reponse.",
                "source":"reference.pdf p.4"}]).to_csv(path,index=False)
            generated={"prediction":"Une reponse.","context":"Contexte effectivement utilise",
                "retrieved_sources":'[ {"source": "autre.pdf", "page": 23} ]',
                "prompt_version":"scoped-context-v2"}
            with patch("migrant_ai.quality.compute_semantic_similarity",return_value=1.0):
                frame=evaluate_test_set(path,lambda q,c:generated,measure_energy=False)
        self.assertEqual(frame.loc[0,"source"],"reference.pdf p.4")
        self.assertEqual(frame.loc[0,"context"],generated["context"])
        self.assertEqual(frame.loc[0,"retrieved_sources"],generated["retrieved_sources"])
    def test_missing_passages_skip_generation(self):
        from migrant_ai.pipeline import generate_with_sources
        fake=SimpleNamespace(retrieve=MagicMock(return_value=[]),format_context=lambda p:"",
            NO_INFO_ANSWER="Information absente")
        with patch.dict(sys.modules,{"rag":fake}), patch("migrant_ai.pipeline.generate_response") as generate:
            result=generate_with_sources("Question")
        generate.assert_not_called()
        self.assertTrue(result["generation_skipped"])
        self.assertEqual(result["retrieved_sources"],"[]")
    def test_manual_context_is_identified(self):
        from migrant_ai.pipeline import generate_with_sources
        with patch("migrant_ai.pipeline.generate_response",return_value="Reponse"):
            result=generate_with_sources("Question","Contexte manuel")
        self.assertEqual(result["context_origin"],"manual")
        self.assertEqual(result["context"],"Contexte manuel")
        self.assertFalse(result["generation_skipped"])
    def test_different_prompt_versions_are_not_aggregated(self):
        rows=[dict(model="llama3:8b",chunk_size=50,top_k=2,temperature=0.1,
            prompt_version=version,bleu=value) for version,value in [("old",0.2),("new",0.8)]]
        summary=summarize(pd.DataFrame(rows))
        self.assertEqual(len(summary),2)
        self.assertEqual(set(summary["bleu_median"]),{0.2,0.8})
        self.assertTrue((summary["n_questions"]==1).all())

if __name__ == "__main__":
    unittest.main()
