import tempfile
import unittest
from pathlib import Path
from migrant_ai.campaign import append_record, read_ledger, CONFIGURATIONS

class CheckpointTests(unittest.TestCase):
    def test_reopen_preserves_records_and_unicode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            self.assertEqual(read_ledger(path), {})
            first = {"record_id": "1", "prediction": "réfugiés"}
            append_record(path, first)
            append_record(path, {"record_id": "2", "prediction": "test"})
            self.assertEqual(read_ledger(path)["1"], first)
            self.assertEqual(len(read_ledger(path)), 2)
    def test_duplicate_or_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.jsonl"
            append_record(path, {"record_id": "1"})
            append_record(path, {"record_id": "1"})
            with self.assertRaises(ValueError):
                read_ledger(path)
            path.write_text('{"record_id":', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_ledger(path)
    def test_protocol_has_six_unique_configurations(self):
        self.assertEqual(len(set(CONFIGURATIONS)), 6)
        self.assertEqual({(c, k) for c, k, _ in CONFIGURATIONS}, {(50, 2), (100, 5)})
