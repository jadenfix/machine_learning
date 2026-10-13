import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "Ensemble_Methods_for_Breast_Cancer_Dataset.ipynb"


class ResearchRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name) / "repo"
        self.work.mkdir()
        shutil.copy2(ROOT / "run_research.py", self.work / "run_research.py")
        shutil.copy2(ROOT / NOTEBOOK, self.work / NOTEBOOK)

    def tearDown(self):
        self.temp.cleanup()

    def run_workflow(self):
        return subprocess.run(
            [sys.executable, "run_research.py", NOTEBOOK, "--output", "evidence"],
            cwd=self.work,
            text=True,
            capture_output=True,
            timeout=60,
        )

    def test_failed_rerun_preserves_last_complete_evidence(self):
        first = self.run_workflow()
        self.assertEqual(first.returncode, 0, first.stderr)
        report_path = self.work / "evidence" / "execution.json"
        original = report_path.read_bytes()

        notebook = json.loads((self.work / NOTEBOOK).read_text())
        notebook["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["raise RuntimeError('forced failure')\n"],
            }
        )
        (self.work / NOTEBOOK).write_text(json.dumps(notebook))

        failed = self.run_workflow()
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("forced failure", failed.stderr)
        self.assertEqual(report_path.read_bytes(), original)
        self.assertEqual(list(self.work.glob(".evidence.staging-*")), [])

    def test_successful_rerun_replaces_stale_directory_contents(self):
        first = self.run_workflow()
        self.assertEqual(first.returncode, 0, first.stderr)
        stale = self.work / "evidence" / "stale.txt"
        stale.write_text("must disappear")

        second = self.run_workflow()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertFalse(stale.exists())
        report = json.loads((self.work / "evidence" / "execution.json").read_text())
        self.assertEqual(report["publication"], "transactional")
        self.assertEqual(report["executed_cells"], [1])
        self.assertEqual(len(report["notebook_sha256"]), 64)

    def test_repository_root_cannot_be_used_as_output(self):
        result = subprocess.run(
            [sys.executable, "run_research.py", NOTEBOOK, "--output", "."],
            cwd=self.work,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not be the repository root", result.stderr)

    def test_repository_ancestor_cannot_be_used_as_output(self):
        parent_sentinel = self.work.parent / "parent-sentinel.txt"
        sibling = self.work.parent / "sibling"
        sibling.mkdir()
        sibling_sentinel = sibling / "sibling-sentinel.txt"
        parent_sentinel.write_text("preserve parent")
        sibling_sentinel.write_text("preserve sibling")

        result = subprocess.run(
            [sys.executable, "run_research.py", NOTEBOOK, "--output", ".."],
            cwd=self.work,
            text=True,
            capture_output=True,
            timeout=10,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("or one of its ancestors", result.stderr)
        self.assertEqual(parent_sentinel.read_text(), "preserve parent")
        self.assertEqual(sibling_sentinel.read_text(), "preserve sibling")
        self.assertEqual(list(self.work.parent.glob(".*.staging-*")), [])

    def test_existing_file_cannot_be_used_as_output(self):
        (self.work / "evidence").write_text("source bytes")
        result = self.run_workflow()
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a directory", result.stderr)
        self.assertEqual((self.work / "evidence").read_text(), "source bytes")


if __name__ == "__main__":
    unittest.main()
