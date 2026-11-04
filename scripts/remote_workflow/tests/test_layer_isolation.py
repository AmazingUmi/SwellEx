"""Enforce strict 3-module boundaries across data_processing, network_training, and remote_workflow."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[3]
SCRIPTS = PROJECT / "scripts"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class LayerIsolationTests(unittest.TestCase):
    def test_top_level_scripts_layout_contains_only_three_layers_and_launcher(self):
        allowed = {"__init__.py", "run.sh", "data_processing", "network_training", "remote_workflow"}
        actual = {
            p.name
            for p in SCRIPTS.iterdir()
            if p.name != "__pycache__" and not p.name.startswith(".")
        }
        # Only assert exact top-level set once legacy pack directories are removed.
        if not (SCRIPTS / "base_pack").exists():
            self.assertEqual(actual, allowed)

    def test_data_processing_has_no_network_training_remote_workflow_torch_or_matplotlib_imports(self):
        forbidden_roots = {"torch", "torchvision", "matplotlib"}
        forbidden_prefixes = ("scripts.network_training", "scripts.remote_workflow")
        for path in sorted((SCRIPTS / "data_processing").rglob("*.py")):
            if path.name.startswith("._"):
                continue
            for mod in _imported_modules(path):
                root = mod.split(".")[0]
                self.assertNotIn(root, forbidden_roots, f"{path} imports forbidden {mod}")
                self.assertFalse(
                    mod.startswith(forbidden_prefixes),
                    f"{path} imports forbidden cross-layer module {mod}",
                )
                self.assertNotIn("localization_networks", mod, f"{path} imports legacy {mod}")

    def test_network_training_has_no_remote_workflow_scipy_or_raw_generator_imports(self):
        forbidden_roots = {"scipy"}
        forbidden_prefixes = ("scripts.remote_workflow",)
        for path in sorted((SCRIPTS / "network_training").rglob("*.py")):
            if path.name.startswith("._"):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Origindata", text, f"{path} references Origindata")
            for mod in _imported_modules(path):
                root = mod.split(".")[0]
                self.assertNotIn(root, forbidden_roots, f"{path} imports forbidden {mod}")
                self.assertFalse(
                    mod.startswith(forbidden_prefixes),
                    f"{path} imports forbidden cross-layer module {mod}",
                )
                self.assertNotIn("generation_core", mod, f"{path} imports raw signal generator {mod}")
                self.assertNotIn("localization_networks", mod, f"{path} imports legacy {mod}")

    def test_remote_workflow_has_no_legacy_localization_networks_imports(self):
        for path in sorted((SCRIPTS / "remote_workflow").rglob("*.py")):
            if path.name.startswith("._") or path.name == "test_layer_isolation.py":
                continue
            for mod in _imported_modules(path):
                self.assertNotIn("localization_networks", mod, f"{path} imports legacy {mod}")

    def test_public_dispatcher_help_uses_standard_library_only(self):
        code = """import importlib.abc, sys
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'scipy', 'h5py', 'matplotlib'}:
            raise AssertionError(f'Forbidden numerical import in dispatcher: {fullname}')
sys.meta_path.insert(0, BlockHeavy())
from scripts.remote_workflow.packs import PACKS
from scripts.remote_workflow.run import main
for name in PACKS:
    try:
        main([name, '--help'])
    except SystemExit as exc:
        assert exc.code == 0
"""
        proc = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)


if __name__ == "__main__":
    unittest.main()
