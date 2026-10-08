#!/usr/bin/env python3
"""Executable contract checks; run numerical validation on the compute host.

quick validates current Python implementations on an explicitly selected device.
datasets validates every generated HDF5 file and compares the three feature routes.
No original data or persistent network output is changed.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts" / "localization_networks"
PACKAGE = "scripts.localization_networks"
sys.path.insert(0, str(ROOT))
METHODS = ("RBD", "ELM", "SCM")
DATA_CLASSES = {"RBD": "RbdRangeH5Dataset", "ELM": "ElmRangeH5Dataset", "SCM": "ScmRangeH5Dataset"}

def number(value, name, *, minimum=0, integer=False, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{name} must be a finite number")
    require(value >= minimum and (not positive or value > 0), f"{name} is outside its permitted range")
    require(not integer or value == int(value), f"{name} must be an integer")


def normalize_expected(raw):
    """Use the generation schema as the single authority for configured validation."""
    require(isinstance(raw, dict), "expected config must be an object")
    flat = {"num_elements", "freq_hz", "min_samples"}
    if set(raw) <= flat:
        for key in ("num_elements", "min_samples"):
            if key in raw:
                number(raw[key], key, positive=True, integer=True)
        if "freq_hz" in raw:
            require(isinstance(raw["freq_hz"], list) and raw["freq_hz"], "freq_hz must be a nonempty array")
            for value in raw["freq_hz"]:
                number(value, "freq_hz")
            require(all(a < b for a, b in zip(raw["freq_hz"], raw["freq_hz"][1:])), "freq_hz must be strictly increasing")
        return raw
    from scripts.data_processing.generation_config import normalize_config
    try:
        return normalize_config(raw, root=ROOT)
    except ValueError as exc:
        raise AssertionError(f"invalid canonical generation config: {exc}") from exc


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Report:
    def __init__(self, command, output):
        self.output = output
        self.data = {"schema_version": 1, "command": command, "status": "FAIL",
                     "started_at": datetime.now(timezone.utc).isoformat(),
                     "python": sys.version, "checks": [], "dataset_directories": {m: [] for m in METHODS}}

    def check(self, name, function):
        start = time.monotonic()
        try:
            details = function()
            result = {"name": name, "status": "PASS", "details": details or {}}
        except Exception as exc:
            result = {"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc()}
        result["duration_s"] = round(time.monotonic() - start, 3)
        self.data["checks"].append(result)
        print(f"{result['status']} {name}", flush=True)
        self.save(final=False)
        return result["status"] == "PASS"

    def save(self, final=True):
        if final:
            self.data["status"] = "PASS" if self.data["checks"] and all(c["status"] == "PASS" for c in self.data["checks"]) else "FAIL"
            self.data["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_name(self.output.name + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.output)


def data_module(method):
    return importlib.import_module(f"{PACKAGE}.{method}_method.network.data")


def model_module(method):
    return importlib.import_module(f"{PACKAGE}.{method}_method.network.model")


def grnn_module(name):
    return importlib.import_module(f"{PACKAGE}.0_GRNN_related.{name}")


def syntax_check():
    paths = sorted(p for p in SCRIPTS.rglob("*.py") if not p.name.startswith("._"))
    failures = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            failures.append(f"{path}: {exc}")
    require(not failures, "\n".join(failures))
    return {"files": len(paths)}


def imports_check():
    names = [f"{PACKAGE}.common.{name}" for name in ("h5_utils", "paths", "training_utils", "prediction_utils")]
    names += [f"{PACKAGE}.0_GRNN_related.{name}" for name in ("grnn_model", "grnn_data", "grnn_paths", "grnn_cli")]
    names.append("scripts.data_processing.generation_config")
    for method in METHODS:
        for path in sorted((SCRIPTS / f"{method}_method" / "network").rglob("*.py")):
            if not path.name.startswith("._"):
                parts = path.relative_to(ROOT).with_suffix("").parts
                names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    failures = []
    for name in sorted(set(names)):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    require(not failures, "\n".join(failures))
    return {"modules": len(set(names))}


def cli_check(method):
    entry = SCRIPTS / ("0_GRNN_related/GRNN_main.py" if method == "GRNN" else f"{method}_method/Network_main.py")
    commands = ([], ["build" if method == "GRNN" else "train"], ["predict"])
    results = []
    for command in commands:
        proc = subprocess.run([sys.executable, str(entry), *command, "--help"], cwd=ROOT,
                              text=True, capture_output=True, timeout=90)
        require(proc.returncode == 0 and "usage:" in proc.stdout.lower(),
                f"{entry} {' '.join(command)} --help: exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        results.append({"arguments": [*command, "--help"], "exit_code": proc.returncode})
    parser_module = grnn_module("grnn_cli") if method == "GRNN" else importlib.import_module(f"{PACKAGE}.{method}_method.network.cli")
    parser = parser_module.build_parser()
    if method != "GRNN":
        require(set(parser_module.MODEL_CHOICES) == set(model_module(method).MODEL_REGISTRY), "CLI model choices differ from registry")
    return {"commands": results}


def device_check(device):
    import torch
    chosen = torch.device(device)
    if chosen.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no CPU fallback")
    tensor = torch.ones(1, device=chosen)
    require(tensor.device.type == chosen.type, f"device mismatch: {tensor.device}")
    return {"requested": device, "actual": str(tensor.device), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(chosen) if chosen.type == "cuda" else None,
            "cpu_threads": torch.get_num_threads()}


def threads_check():
    import torch
    threads = max(1, min(4, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.set_num_threads(threads)
    return {"cpu_threads": torch.get_num_threads()}


def workflow_tests_check():
    directory = SCRIPTS / "workflow" / "tests"
    require(directory.is_dir() and list(directory.glob("test*.py")), "workflow unittest files are missing")
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", "test*.py", "-v"],
                          cwd=ROOT, text=True, capture_output=True, timeout=180)
    require(proc.returncode == 0, f"workflow unittest exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def visualization_tests_check():
    directory = ROOT / "scripts" / "visualization" / "tests"
    require(directory.is_dir() and list(directory.glob("test*.py")), "visualization unittest files are missing")
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", "test*.py", "-v"],
                          cwd=ROOT, text=True, capture_output=True, timeout=180)
    require(proc.returncode == 0, f"visualization unittest exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def model_check(method, name, device):
    import torch
    module = model_module(method)
    # Real registered classes and defaults, including full ResNet-50.
    height = {"RBD": 21, "ELM": 210, "SCM": 231}[method]
    shape = (2, height, 23)
    torch.manual_seed(2026)
    config = module.build_model_config(name, shape, dropout=0.0)
    model = module.build_model(name, config).to(device)
    require(isinstance(model, module.MODEL_REGISTRY[name][1]), "registry did not construct its registered class")
    model.train()
    x = torch.randn(2, *shape, device=device, requires_grad=True)
    prediction = model(x)
    require(tuple(prediction.shape) == (2, 1), f"prediction shape {tuple(prediction.shape)}")
    require(bool(torch.isfinite(prediction).all()), "nonfinite predictions")
    loss = torch.nn.functional.mse_loss(prediction, torch.tensor([[0.4], [-0.6]], device=device))
    loss.backward()
    require(x.grad is not None and bool(torch.isfinite(x.grad).all()), "input gradients missing/nonfinite")
    gradients = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    require(gradients and all(g is not None and bool(torch.isfinite(g).all()) for _, g in gradients),
            "parameter gradients missing/nonfinite: " + ", ".join(n for n, g in gradients if g is None or not bool(torch.isfinite(g).all())))
    require(any(bool((g != 0).any()) for _, g in gradients), "all parameter gradients zero")
    model.eval()
    with torch.no_grad():
        evaluated = model(x.detach())
    require(tuple(evaluated.shape) == (2, 1) and bool(torch.isfinite(evaluated).all()), "invalid eval predictions")
    details = {"class": f"{type(model).__module__}.{type(model).__name__}", "input_shape": list(x.shape),
               "output_shape": list(prediction.shape), "parameters": sum(p.numel() for p in model.parameters()),
               "loss": float(loss.detach()), "device": str(prediction.device)}
    del model, x, prediction, loss, evaluated, gradients
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return details


def shared_check():
    import torch
    from scripts.localization_networks.common import h5_utils, paths
    from scripts.localization_networks.common.training_utils import target_stats
    require(paths.PROJECT_ROOT == ROOT, "common project root points outside the checkout root")
    require(grnn_module("grnn_paths").PROJECT_ROOT == ROOT, "GRNN project root points outside the checkout root")
    for count in (1, 2, 7, 41):
        train, val = h5_utils.split_indices(count, 0.25, 2026)
        require(not set(train) & set(val), f"split overlaps for n={count}")
        require(sorted(train + val) == list(range(count)), f"split loses samples for n={count}")
        require(train and (val if count > 1 else not val), f"empty split for n={count}")
        require((train, val) == h5_utils.split_indices(count, 0.25, 2026), "seed is not repeatable")
    require(h5_utils.split_indices(41, 0.25, 2026) != h5_utils.split_indices(41, 0.25, 2027), "seed does not affect split")
    labels = type("Labels", (), {"labels": [2.0, 4.0, 7.0]})()
    require(torch.equal(h5_utils.subset_labels(labels, [2, 0]), torch.tensor([[7.0], [2.0]])), "subset labels/order mismatch")
    require(target_stats(torch.full((3, 1), 5.0)) == (5.0, 1.0), "constant target scale must remain usable")
    with tempfile.TemporaryDirectory(prefix="swellex-paths-") as tmp:
        directory = Path(tmp)
        a, b = directory / "a_train.h5", directory / "b_train.h5"
        a.touch(); b.touch()
        resolved = h5_utils.resolve_h5_paths([str(b), str(directory / "*_train.h5"), str(a)])
        require(resolved == [a.resolve(), b.resolve()], "HDF5 resolution not sorted/deduplicated")
        require(h5_utils.resolve_h5_paths([str(directory / "missing*.h5")], required=False) == [], "optional resolution differs")
        try:
            h5_utils.resolve_h5_paths([str(directory / "missing*.h5")])
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing required data accepted")
        fixed = directory / "data_best.pt"
        require(paths.latest_time_suffixed_path(fixed) is None, "missing checkpoint should resolve to None")
        old = paths.with_time_suffix(fixed, "0101_010101"); old.touch()
        newer = paths.with_time_suffix(fixed, "0101_010100"); newer.touch()
        os.utime(old, (10, 10)); os.utime(newer, (20, 20))
        require(paths.latest_time_suffixed_path(fixed) == newer, "latest checkpoint must use mtime rather than filename")
        require(paths.dataset_dir(str(directory)) == directory, "explicit dataset path not honored")
        require(paths.train_output_dir(directory, "model") == directory / "model" / "train_outputs", "training path mismatch")
        require(paths.train_output_dir(directory / "model", "model") == directory / "model" / "train_outputs", "model path duplicated")
    return {"split_sizes": [1, 2, 7, 41], "path_resolution": "sorted unique, explicit paths, mtime checkpoints"}


def expected_pairs(method, elements):
    # Enumerate the documented MATLAB column-major upper triangle independently.
    return [(i, j) for j in range(1, elements + 1) for i in range(1, j + (method == "SCM"))]


def write_fixture(path, method, reverse, *, constant=False):
    import h5py
    import numpy as np
    n, elements, frequencies = 7, 4, 11
    pairs = expected_pairs(method, elements) if method != "RBD" else []
    height = len(pairs) if pairs else elements
    # Decimal place markers make sample, row, frequency, and RI axes observable.
    sample, row, freq, ri = np.indices((n, height, frequencies, 2))
    features = (sample * 10000 + row * 100 + freq * 3 + ri).astype(np.float32)
    if constant:
        features.fill(8.0)
    with h5py.File(path, "w") as h5:
        x = h5.create_dataset("X", data=features.transpose(3, 2, 1, 0) if reverse else features)
        logical = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
        x.attrs["axis_order"] = logical
        x.attrs["storage_axis_order"] = ",".join(logical.split(",")[::-1]) if reverse else logical
        h5["y_range_km"] = np.array([1, 2, 3, 4, np.nan, 6, 7], dtype=np.float32)
        h5["valid_sample"] = np.array([1, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
        h5["array/depth_m"] = np.arange(elements) + 100.0
        h5["array/element_idx"] = np.arange(1, elements + 1)
        h5["frequency/freq_hz"] = np.arange(frequencies) * 5.0
        h5["split/source_segment_idx"] = np.arange(n) + 51
        h5["time/window_center_s"] = np.arange(n) + 70.5
        if pairs:
            h5["pair/numerator_element_idx"] = [p[0] for p in pairs]
            h5["pair/denominator_element_idx"] = [p[1] for p in pairs]
    return features, [0, 2, 3, 5, 6]


def loader_check(method):
    import numpy as np
    import torch
    module = data_module(method)
    cls = getattr(module, DATA_CLASSES[method])
    variants = []
    with tempfile.TemporaryDirectory(prefix="swellex-h5-") as tmp:
        directory = Path(tmp)
        for reverse in (False, True):
            fixture = directory / f"{method}_{reverse}.h5"
            features, keep = write_fixture(fixture, method, reverse)
            dataset = cls([fixture], normalize_input=False)
            try:
                require(len(dataset) == len(keep), "valid mask/nonfinite label filtering incorrect")
                require(dataset.source_segment_idx == [51 + s for s in keep], "source indices not filtered in sample order")
                require(dataset.window_center_s == [70.5 + s for s in keep], "window times not filtered in sample order")
                for item, source in enumerate(keep):
                    x, y = dataset[item]
                    expected = torch.from_numpy(features[source].transpose(2, 0, 1).copy())
                    require(torch.equal(x, expected), f"{method} reversed={reverse}: feature axes/order changed at sample {source}")
                    require(tuple(y.shape) == (1,) and float(y[0]) == source + 1, "label shape/value incorrect")
            finally:
                dataset.close()
            normalized = cls([fixture], normalize_input=True)
            try:
                x, _ = normalized[0]
                require(bool(torch.isfinite(x).all()), "normalized features nonfinite")
                require(torch.allclose(x.mean((1, 2)), torch.zeros(2), atol=2e-5), "channel means not centered")
                require(torch.allclose(x.std((1, 2)), torch.ones(2), atol=2e-5), "channel scales not normalized")
            finally:
                normalized.close()
            if method == "SCM":
                load_scm_grnn_features = grnn_module("grnn_data").load_scm_grnn_features
                bundle = load_scm_grnn_features([fixture])
                expected = np.stack([features[s].transpose(2, 0, 1).reshape(-1) for s in keep])
                require(np.array_equal(bundle.x.numpy(), expected), "GRNN SCM flatten order differs from CHW contract")
                require(bundle.source_segment_idx == [51 + s for s in keep], "GRNN source filtering incorrect")
            variants.append("MATLAB-reversed" if reverse else "native")
            # Historical files may omit declarations; retain that input contract.
            import h5py
            with h5py.File(fixture, "r+") as h5:
                del h5["X"].attrs["axis_order"]
                del h5["X"].attrs["storage_axis_order"]
            legacy = cls([fixture], normalize_input=False)
            try:
                require(torch.equal(legacy[0][0], torch.from_numpy(features[0].transpose(2, 0, 1).copy())), "legacy undeclared HDF5 axes changed")
            finally:
                legacy.close()
            small = directory / f"small_{reverse}.h5"
            small_features, _ = write_fixture(small, method, reverse)
            with h5py.File(small, "r+") as h5:
                stored = np.asarray(h5["X"])
                attrs = dict(h5["X"].attrs)
                stored = stored[..., :2] if reverse else stored[:2]
                del h5["X"]
                x = h5.create_dataset("X", data=stored)
                for key, value in attrs.items():
                    x.attrs[key] = value
                for key, values in (("y_range_km", [1.0, 2.0]), ("valid_sample", [1, 1]),
                                    ("split/source_segment_idx", [51, 52]), ("time/window_center_s", [70.5, 71.5])):
                    del h5[key]
                    h5[key] = values
            tiny = cls([small], normalize_input=False)
            try:
                require(len(tiny) == 2, "declared two-sample fixture lost samples")
                for index in range(2):
                    expected = torch.from_numpy(small_features[index].transpose(2, 0, 1).copy())
                    require(torch.equal(tiny[index][0], expected), "sample axis of length two confused with RI axis")
                if method == "SCM":
                    small_grnn = grnn_module("grnn_data").load_scm_grnn_features([small])
                    expected_flat = np.stack([small_features[s].transpose(2, 0, 1).reshape(-1) for s in range(2)])
                    require(np.array_equal(small_grnn.x.numpy(), expected_flat), "GRNN declared two-sample axes incorrect")
            finally:
                tiny.close()
        constant = directory / "constant.h5"
        write_fixture(constant, method, True, constant=True)
        dataset = cls([constant])
        try:
            x, _ = dataset[0]
            require(torch.equal(x, torch.zeros_like(x)), "constant input normalization should produce finite zeros")
        finally:
            dataset.close()
        bundle = module.random_split_bundle([constant], normalize_input=False, val_fraction=0.25, seed=2026, split_mode="validation")
        try:
            require(not set(bundle.train_idx) & set(bundle.val_idx), "bundle split overlaps")
            require(sorted(bundle.train_idx + bundle.val_idx) == list(range(len(bundle.dataset))), "bundle split loses samples")
            require(tuple(bundle.train_labels.shape) == (len(bundle.train_idx), 1), "bundle labels shape incorrect")
            require(bundle.source_paths == [constant] and bundle.split_mode == "validation", "bundle metadata changed")
        finally:
            bundle.dataset.close()
        fixed = module.fixed_split_bundle([constant], [constant], normalize_input=False, split_mode="fixed_validation")
        try:
            require(len(fixed.train_dataset) == 5 and len(fixed.val_dataset) == 5, "fixed bundle ignores valid filtering")
            require(fixed.train_idx == list(range(5)) and fixed.val_idx == list(range(5)), "fixed bundle local indices changed")
            require(torch.equal(fixed.train_labels, torch.tensor([[1.0], [3.0], [4.0], [6.0], [7.0]])), "fixed bundle label order incorrect")
        finally:
            fixed.train_dataset.close()
            fixed.val_dataset.close()
        invalid = directory / "invalid.h5"
        write_fixture(invalid, method, True)
        import h5py
        with h5py.File(invalid, "r+") as h5:
            h5["valid_sample"][...] = np.zeros(7, dtype=np.uint8)
        try:
            cls([invalid])
        except ValueError as exc:
            require("No valid samples" in str(exc), f"unexpected all-invalid rejection: {exc}")
        else:
            raise AssertionError("all-invalid file accepted")
    return {"layouts": variants, "pair_semantics": "i<j" if method == "ELM" else "i<=j" if method == "SCM" else "elements",
            "checks": ["coordinate values", "valid mask", "nonfinite label", "source/time alignment", "normalization", "bundle"]}


def grnn_check(device):
    import torch
    ScmGrnnRegressor = grnn_module("grnn_model").ScmGrnnRegressor
    reference = torch.tensor([[-1.0, 2.0], [1.0, 2.0]])
    targets = torch.tensor([[2.0], [8.0]])
    queries = torch.tensor([[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0], [100.0, 2.0]])
    model = ScmGrnnRegressor(spread=0.01, device=device)
    try:
        model.predict(queries)
    except RuntimeError:
        pass
    else:
        raise AssertionError("unfitted GRNN accepted prediction")
    model.fit_reference(reference, targets)
    require(model.x_reference.device.type == torch.device(device).type, "GRNN reference did not use requested device")
    values = model.predict(queries, batch_size=1)
    require(bool(torch.isfinite(values).all()) and tuple(values.shape) == (4, 1), "GRNN nonfinite/wrong-shape outputs")
    require(torch.allclose(values[:3], torch.tensor([[2.0], [5.0], [8.0]]), atol=1e-5), "reference point/symmetry limit incorrect")
    require(bool(((values >= 2.0) & (values <= 8.0)).all()), "positive-kernel regression left reference target convex hull")
    require(torch.equal(values, model.predict(queries, batch_size=4)), "batch size affects predictions")
    restored = ScmGrnnRegressor.from_artifact(model.to_artifact(), device=device)
    require(torch.equal(values, restored.predict(queries)), "artifact round trip changes predictions")
    permuted = ScmGrnnRegressor(spread=0.01, device=device)
    permuted.fit_reference(reference.flip(0), targets.flip(0))
    require(torch.allclose(values, permuted.predict(queries), atol=1e-6), "reference order affects predictions")
    standardized = ScmGrnnRegressor(spread=0.3, standardize_input=True, device=device)
    standardized.fit_reference(reference, targets)
    rescaled = ScmGrnnRegressor(spread=0.3, standardize_input=True, device=device)
    scale = torch.tensor([3.0, 4.0]); shift = torch.tensor([9.0, -7.0])
    rescaled.fit_reference(reference * scale + shift, targets)
    require(torch.allclose(standardized.predict(queries), rescaled.predict(queries * scale + shift), atol=2e-5), "standardized GRNN not invariant under feature units/offset")
    constant = ScmGrnnRegressor(spread=0.1, standardize_input=True, device=device)
    constant.fit_reference(reference, torch.full((2, 1), 3.25))
    require(torch.allclose(constant.predict(queries), torch.full((4, 1), 3.25)), "GRNN does not preserve constant targets")
    for spread in (0.0, -0.1):
        try:
            ScmGrnnRegressor(spread=spread, device=device)
        except ValueError:
            pass
        else:
            raise AssertionError("nonpositive spread accepted")
    return {"device": device, "invariants": ["unfitted rejection", "constant preservation", "symmetry", "reference-point limit",
            "convex hull", "far-query finite", "batch invariance", "reference-order invariance", "standardized affine invariance", "artifact round trip"]}


def quick(report, device):
    report.check("python.syntax", syntax_check)
    report.check("runtime.cpu_threads", threads_check)
    report.check("python.imports", imports_check)
    for method in (*METHODS, "GRNN"):
        report.check(f"cli.{method}", lambda m=method: cli_check(m))
    usable = report.check("runtime.device", lambda: device_check(device))
    report.check("shared.paths_and_splits", shared_check)
    report.check("workflow.unit_tests", workflow_tests_check)
    report.check("visualization.unit_tests", visualization_tests_check)
    for method in METHODS:
        report.check(f"loader.{method}", lambda m=method: loader_check(m))
        try:
            names = list(model_module(method).MODEL_REGISTRY)
            require(len(names) == 4, f"{method} registry must contain four current models")
        except Exception as exc:
            report.check(f"models.{method}.registry", lambda: (_ for _ in ()).throw(exc))
            continue
        for name in names:
            def run(m=method, n=name):
                require(usable, "requested runtime device check failed")
                return model_check(m, n, device)
            report.check(f"model.{method}.{name}", run)
    def run_grnn():
        require(usable, "requested runtime device check failed")
        return grnn_check(device)
    report.check("grnn.numerical_invariants", run_grnn)
    report.check("dataset_validator.negative_controls", validator_controls_check)


def vector(h5, key, *, count=None, finite=True):
    import numpy as np
    require(key in h5, f"missing {key}")
    values = np.asarray(h5[key]).reshape(-1)
    require(count is None or len(values) == count, f"{key} has {len(values)} entries; expected {count}")
    require(not finite or bool(np.isfinite(values).all()), f"{key} contains nonfinite values")
    return values


def attribute_text(value):
    import numpy as np
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        return attribute_text(value.reshape(-1)[0])
    return str(value)


def classify(path):
    import h5py
    # Names take priority so malformed SCM diagonal metadata cannot be reclassified as ELM.
    for method in METHODS:
        if path.name.upper().startswith(method + "_"):
            return method
    with h5py.File(path, "r") as h5:
        if "/rbd" in h5:
            return "RBD"
        if "/pair/numerator_element_idx" in h5 and "/pair/denominator_element_idx" in h5:
            i = vector(h5, "/pair/numerator_element_idx")
            j = vector(h5, "/pair/denominator_element_idx")
            return "SCM" if (i == j).any() else "ELM"
    raise AssertionError(f"cannot identify feature method for {path}")


def read_dataset_metadata(path):
    candidates = list(path.parent.glob("*_metadata.json"))
    require(len(candidates) <= 1, f"ambiguous dataset metadata in {path.parent}")
    if not candidates:
        return {}
    value = json.loads(candidates[0].read_text(encoding="utf-8"))
    require(isinstance(value, dict), "dataset metadata must be an object")
    return value


def configured_window_check(record, expected, metadata):
    import numpy as np
    if "dataset" not in expected:
        return {}
    d, method = expected["dataset"], record["method"]
    snapshots = 1 if method == "RBD" else d["num_snapshots_per_segment"]
    hop = 1 if method == "RBD" else snapshots - d["snapshot_overlap_count"]
    base_start = d["segment_start_s"] or 0.0
    step = d["segment_step_s"] * hop
    duration = d["segment_duration_s"] + (snapshots - 1) * d["segment_step_s"]
    wanted_starts = base_start + (record["sources"].astype(np.float64) - 1) * step
    require(np.allclose(record["starts"], wanted_starts, rtol=0, atol=1e-8), "source indices/window starts disagree with configured start/step/Ns")
    require(np.allclose(record["stops"] - record["starts"], duration, rtol=0, atol=1e-8), "window duration differs from configured duration/step/Ns")
    if d["segment_end_s"] is not None:
        require(bool((record["starts"] + (snapshots - 1) * d["segment_step_s"] <= d["segment_end_s"] + 1e-8).all()), "windows exceed configured final base-window start")
    wanted_first = np.floor(record["starts"] * d["fs"] + 0.5).astype(np.uint64) + 1
    samples = round(d["fs"] * d["segment_duration_s"]) + (snapshots - 1) * round(d["fs"] * d["segment_step_s"])
    require(np.array_equal(record["sample_starts"], wanted_first), "sample_start_idx disagrees with configured fs and window time")
    require(np.array_equal(record["sample_stops"], wanted_first + samples - 1), "sample_stop_idx disagrees with configured fs/duration/Ns")
    fft_samples = round(d["fs"] * d["segment_duration_s"])
    full = np.arange(fft_samples // 2 + 1) * d["fs"] / fft_samples
    bins = record["freq"] * fft_samples / d["fs"]
    require(bool(((bins >= 0) & (bins <= fft_samples // 2)).all()) and np.allclose(bins, np.round(bins), rtol=0, atol=1e-8), "frequency axis is not on configured one-sided snapshot FFT grid")
    modes = ["full"] if method == "RBD" and expected["rbd"]["frequency_estimation"] == "full" else d["frequency_selection_modes"]
    f = d["frequency_selection_config"]
    fixed_bins = set()
    for mode in modes:
        if mode == "full":
            fixed_bins.update(range(len(full)))
        elif mode in ("deep", "shallow"):
            defaults = [49,64,79,94,112,130,148,166,201,235,283,338,388] if mode == "deep" else [109,127,145,163,198,232,280,335,385]
            targets = np.asarray(f.get(mode + "_target_freq_hz", defaults)).reshape(-1)
            fixed_bins.update(int(np.abs(full - target).argmin()) for target in targets)
        elif mode == "mel":
            lo, hi = f.get("mel_min_freq_hz", max(1, full[0])), f.get("mel_max_freq_hz", full[-1])
            mel = np.linspace(2595 * np.log10(1 + lo / 700), 2595 * np.log10(1 + hi / 700), f.get("mel_num_bins", 64))
            targets = 700 * (10 ** (mel / 2595) - 1)
            fixed_bins.update(int(np.abs(full - target).argmin()) for target in targets)
    actual_bins = set(map(int, np.round(bins)))
    if "adapt" not in modes:
        require(actual_bins == fixed_bins, "frequencies disagree with configured frequency selection")
    else:
        require(fixed_bins <= actual_bins, "adaptive union omitted configured fixed frequency bins")
        extras = full[list(actual_bins - fixed_bins)]
        require(len(extras) <= f.get("adapt_num_bins", 16) and bool(((extras >= f.get("adapt_min_freq_hz", 1)) & (extras <= f.get("adapt_max_freq_hz", full[-1])) & (extras > 0)).all()), "adaptive bins violate configured count/range")
    if metadata:
        for key, value in (("fs_hz", d["fs"]), ("segment_duration_s", duration), ("segment_step_s", step)):
            require(key in metadata and math.isclose(metadata[key], value, rel_tol=0, abs_tol=1e-8), f"metadata {key} differs from expected config")
        require(metadata.get("split_strategy") == d["split_strategy"], "metadata split strategy differs from expected config")
        if method == "RBD":
            actual = metadata.get("rbd_config", {})
            for key in ("normalize_spectrum", "use_plane_wave"):
                require(actual.get(key) == expected["rbd"][key], f"metadata RBD {key} differs from expected config")
            require(actual.get("rbd_beam_selection") == expected["rbd"]["beam_selection"], "metadata RBD beam mode differs")
            require(actual.get("rbd_frequency_estimation") == expected["rbd"]["frequency_estimation"], "metadata RBD frequency mode differs")
            if expected["rbd"]["beam_selection"] == "multipath":
                require(actual.get("rbd_multipath_options") == expected["rbd"]["multipath_options"], "metadata multipath parameters differ")
    prefix = {"RBD": "RBD_green_freq_nn_", "ELM": "ELM_lsr_freq_nn_", "SCM": "SCM_freq_nn_"}[method]
    name = Path(record["path"]).name
    if name.startswith(prefix):
        require(name.startswith(prefix + d["event_name"] + "_"), "HDF5 event name differs from expected config")
    return {"fs": d["fs"], "window_duration_s": duration, "window_step_s": step, "snapshots": snapshots, "frequency_selection": modes}


def configured_split_check(records, expected):
    import numpy as np
    if "dataset" not in expected:
        return {}
    d = expected["dataset"]
    first = records[0]
    metadata = first.get("metadata", {})
    start = d["segment_start_s"] or 0.0
    end = metadata.get("candidate_segment_end_s", d["segment_end_s"])
    require(end is not None, "null segment_end_s requires generated candidate_segment_end_s metadata")
    require(d["segment_end_s"] is None or end <= d["segment_end_s"] + 1e-8, "metadata candidate end exceeds expected config")
    base_count = int(math.floor((end - start) / d["segment_step_s"] + 1e-8)) + 1
    snapshots = 1 if first["method"] == "RBD" else d["num_snapshots_per_segment"]
    hop = 1 if first["method"] == "RBD" else snapshots - d["snapshot_overlap_count"]
    count = (base_count - snapshots) // hop + 1
    require(count > 1, "configured candidate grid has too few windows")
    if metadata:
        require(metadata.get("global_num_segments") == count, "metadata global_num_segments differs from configured grid")
    grid = np.arange(1, count + 1)
    step = hop * d["segment_step_s"]
    duration = d["segment_duration_s"] + (snapshots - 1) * d["segment_step_s"]
    if d["split_strategy"] == "periodic":
        ratio = d["train_test_ratio"]
        train = grid[((grid - 1) % sum(ratio)) < ratio[0]]
        test = grid[((grid - 1) % sum(ratio)) >= ratio[0]]
        if not len(test):
            train, test = grid[:-1], grid[-1:]
    else:
        centers = start + (grid - 1) * step + duration / 2
        raw_t, raw_r = first["raw_times"], first["raw_ranges"]
        candidate_labels = np.interp(centers, raw_t, raw_r, left=np.nan, right=np.nan)
        valid = np.isfinite(candidate_labels)
        require(valid.any(), "no finite range labels on configured grid")
        pivot = int(grid[np.nanargmin(candidate_labels)])
        gap = math.floor(d["gap_s"] / step)
        width = min(max(1, math.floor(d["half_duration_s"] / step)), pivot - gap - 1, count - pivot - gap)
        require(width > 0, "Range_nearby has no symmetric configured windows")
        before = np.arange(pivot - gap - width, pivot - gap)
        after = np.arange(pivot + gap + 1, pivot + gap + width + 1)
        pair_valid = valid[before - 1] & valid[after - 1]
        before, after = before[pair_valid], after[pair_valid]
        train, test = (before, after) if d["train_side"] == "before" else (after, before)
        stride = max(1, math.floor((d["test_step_s"] or step) / step + 0.5))
        test = test[::stride]
    for split, wanted in (("train", train), ("test", test)):
        actual = sorted(int(source) for r in records if r["split"] == split for source in r["sources"])
        require(actual == wanted.tolist(), f"{split} source membership/coverage differs from configured {d['split_strategy']} split; expected={wanted[:20].tolist()}, actual={actual[:20]}")
    return {"strategy": d["split_strategy"], "candidate_windows": count, "expected_train": len(train), "expected_test": len(test)}


def inspect_h5(path, method, expected):
    import h5py
    import numpy as np
    metadata = read_dataset_metadata(path)
    configured_rbd = expected.get("rbd", {})
    saved_rbd = metadata.get("rbd_config", {})
    beam_mode = configured_rbd.get("beam_selection", saved_rbd.get("rbd_beam_selection"))
    frequency_mode = configured_rbd.get("frequency_estimation", saved_rbd.get("rbd_frequency_estimation"))
    norm_summary = {"mode": beam_mode, "active_columns": 0, "zero_columns": 0, "maximum_norm": 0.0,
                    "absolute_tolerance": 2e-6, "zero_threshold": 1e-8, "full_dc_exempt": frequency_mode == "full"}
    with h5py.File(path, "r") as h5:
        labels = vector(h5, "/y_range_km", finite=False)
        n = len(labels)
        require(n > 0, "empty generated split")
        valid = vector(h5, "/valid_sample", count=n)
        require(bool(np.isin(valid, [0, 1]).all()), "valid_sample must be binary")
        require(bool(np.isfinite(labels[valid.astype(bool)]).all()), "valid labels nonfinite")
        require(bool((labels[valid.astype(bool)] >= 0).all()), "valid labels negative")
        # Generation splits contain valid windows only; invalid labels must never enter acceptance data.
        require(bool(np.isfinite(labels).all()), "generated labels contain nonfinite values")
        require(bool(valid.any()), "no valid samples")
        depth = vector(h5, "/array/depth_m")
        elements = vector(h5, "/array/element_idx", count=len(depth))
        require(np.array_equal(elements, np.arange(1, len(depth) + 1)), "element indices must be 1-based ordered")
        frequencies = vector(h5, "/frequency/freq_hz")
        require(len(frequencies) > 0 and frequencies[0] >= 0 and bool((np.diff(frequencies) > 0).all()), "frequencies must be nonnegative, unique and ascending")
        height = len(depth)
        if method != "RBD":
            pairs = expected_pairs(method, len(depth))
            i = vector(h5, "/pair/numerator_element_idx", count=len(pairs))
            j = vector(h5, "/pair/denominator_element_idx", count=len(pairs))
            require(np.array_equal(np.column_stack((i, j)), np.array(pairs)), f"{method} pair order/semantics differ from column-major {'i<j' if method == 'ELM' else 'i<=j'}")
            height = len(pairs)
        require("/X" in h5, "missing /X")
        x = h5["/X"]
        stored_shape = tuple(x.shape)
        logical_axes = f"sample,{'element' if method == 'RBD' else 'pair'},frequency,real_imag"
        require("axis_order" in x.attrs, "missing /X axis_order attribute")
        require(attribute_text(x.attrs["axis_order"]) == logical_axes, f"/X axis_order must be {logical_axes!r}")
        logical_shape = (n, height, len(frequencies), 2)
        if "storage_axis_order" in x.attrs:
            stored_axes = tuple(attribute_text(x.attrs["storage_axis_order"]).split(","))
            declared_axes = tuple(logical_axes.split(","))
            require(len(stored_axes) == 4 and set(stored_axes) == set(declared_axes), "invalid /X storage_axis_order")
            axes = tuple(stored_axes.index(axis) for axis in declared_axes)
            require(axes in ((0, 1, 2, 3), (3, 2, 1, 0)), "generated /X storage order must be native or MATLAB-reversed")
            require(tuple(x.shape[a] for a in axes) == logical_shape, "stored /X shape disagrees with storage_axis_order and common metadata")
            storage = "native" if axes == (0, 1, 2, 3) else "MATLAB-reversed"
        elif x.shape == logical_shape:
            axes = (0, 1, 2, 3)
            storage = "native"
        elif x.shape == logical_shape[::-1]:
            axes = (3, 2, 1, 0)
            storage = "MATLAB-reversed"
        else:
            raise AssertionError(f"/X shape {x.shape} differs from logical {logical_shape} and MATLAB reversed {logical_shape[::-1]}")
        require(x.dtype.kind == "f", f"/X must be floating-point, got {x.dtype}")
        # Read a bounded number of samples per block, covering every stored value.
        bytes_per_sample = max(1, height * len(frequencies) * 2 * x.dtype.itemsize)
        block = max(1, (16 * 1024 * 1024) // bytes_per_sample)
        for start in range(0, n, block):
            selection = [slice(None)] * 4
            selection[axes[0]] = slice(start, min(n, start + block))
            raw = np.asarray(x[tuple(selection)])
            require(bool(np.isfinite(raw).all()), f"nonfinite /X features in sample block [{start},{min(n, start + block)})")
            if method == "RBD" and beam_mode in ("best", "multipath"):
                logical = np.transpose(raw, axes).astype(np.float64)
                norms = np.sqrt(np.sum(logical ** 2, axis=(1, 3)))
                active = norms > 1e-8
                norm_summary["zero_columns"] += int((~active).sum())
                norm_summary["maximum_norm"] = max(norm_summary["maximum_norm"], float(norms.max()))
                if frequency_mode == "full":
                    active[:, frequencies == 0] = False
                norm_summary["active_columns"] += int(active.sum())
                if beam_mode == "best":
                    require(np.allclose(norms[active], 1.0, rtol=0, atol=2e-6), f"RBD best Green column norm must be one; FFT amplitude may have been saved (block {start}, maximum norm {float(norms.max()):.8g})")
                else:
                    require(bool((norms <= 1 + 2e-6).all()), f"RBD multipath Green column norm exceeds one; FFT amplitude may have been saved (block {start})")
        starts = vector(h5, "/time/window_start_s", count=n)
        centers = vector(h5, "/time/window_center_s", count=n)
        stops = vector(h5, "/time/window_stop_s", count=n)
        require(bool(((starts <= centers) & (centers <= stops) & (starts < stops)).all()), "invalid window start/center/stop ordering")
        require(np.allclose(centers, (starts + stops) / 2, rtol=0, atol=1e-8), "window center is not midpoint")
        sample_starts = vector(h5, "/time/sample_start_idx", count=n)
        sample_stops = vector(h5, "/time/sample_stop_idx", count=n)
        sources = vector(h5, "/split/source_segment_idx", count=n)
        for key, values in (("sample_start_idx", sample_starts), ("sample_stop_idx", sample_stops), ("source_segment_idx", sources)):
            require(bool(((values >= 1) & (values == np.floor(values))).all()), f"{key} must be positive integer indices")
        require(bool((sample_starts <= sample_stops).all()), "sample windows have reversed endpoints")
        require(len(set(map(int, sources))) == n, "duplicate source_segment_idx within split")
        require(bool((np.diff(sources.astype(np.int64)) > 0).all()), "source_segment_idx must be strictly increasing")
        require(bool((np.diff(starts) > 0).all()) and bool((np.diff(sample_starts.astype(np.int64)) > 0).all()), "window source order not increasing")
        raw_times = vector(h5, "/range/range_time_s")
        raw_ranges = vector(h5, "/range/range_km_raw", count=len(raw_times))
        require(len(raw_times) > 1 and bool((np.diff(raw_times) > 0).all()), "raw range time must be strictly increasing")
        require(bool((raw_ranges >= 0).all()), "raw ranges negative")
        require(bool(((centers >= raw_times[0]) & (centers <= raw_times[-1])).all()), "window label times outside raw range coverage")
        interpolated = np.interp(centers, raw_times, raw_ranges)
        require(np.allclose(labels, interpolated, rtol=1e-6, atol=1e-6), "range labels disagree with linear interpolation of raw range at window centers")
        if "num_elements" in expected:
            require(len(depth) == expected["num_elements"], "num_elements differs from expected config")
        if "freq_hz" in expected:
            require(np.allclose(frequencies, expected["freq_hz"], rtol=0, atol=1e-8), "frequency axis differs from expected config")
        require(n >= expected.get("min_samples", 1), "split below expected min_samples")
    split = next((s for s in ("train", "val", "test") if path.stem.endswith("_" + s)), None)
    require(split is not None, f"generated HDF5 filename must end in _train/_val/_test: {path.name}")
    record = {"path": str(path), "method": method, "directory": str(path.parent), "split": split,
              "labels": labels, "valid": valid, "freq": frequencies, "depth": depth, "sources": sources,
              "starts": starts, "centers": centers, "stops": stops, "sample_starts": sample_starts, "sample_stops": sample_stops,
              "raw_times": raw_times, "raw_ranges": raw_ranges, "metadata": metadata}
    configured_details = configured_window_check(record, expected, metadata)
    details = {"path": str(path), "method": method, "split": split, "samples": n, "valid_samples": int(valid.sum()),
               "logical_shape": list(logical_shape), "stored_shape": list(stored_shape), "storage": storage,
               "frequency_hz": frequencies.tolist(), "source_segment_idx_min": int(sources.min()), "source_segment_idx_max": int(sources.max())}
    if method == "RBD" and beam_mode:
        details["green_column_norm"] = norm_summary
    if configured_details:
        details["expected_config"] = configured_details
    return record, details


def split_check(records, expected=None):
    import numpy as np
    require(any(r["split"] == "train" for r in records), "missing train split")
    require(any(r["split"] == "test" for r in records), "missing test split")
    for a_idx, a in enumerate(records):
        for b in records[a_idx + 1:]:
            require(np.array_equal(a["freq"], b["freq"]), "frequency axis differs among files in dataset directory")
            require(np.array_equal(a["depth"], b["depth"]), "array depth differs among files in dataset directory")
            if a["split"] == b["split"]:
                continue
            overlap = set(map(int, a["sources"])) & set(map(int, b["sources"]))
            require(not overlap, f"{a['split']}/{b['split']} source windows overlap: {sorted(overlap)[:10]}")
            intervals = sorted([(int(s), int(e), a["split"]) for s, e in zip(a["sample_starts"], a["sample_stops"])]+
                               [(int(s), int(e), b["split"]) for s, e in zip(b["sample_starts"], b["sample_stops"])])
            maxima = {}
            for start, stop, split in intervals:
                other = b["split"] if split == a["split"] else a["split"]
                require(maxima.get(other, 0) < start, f"{a['split']}/{b['split']} contain overlapping original sample intervals near {start}")
                maxima[split] = max(maxima.get(split, 0), stop)
    configured = configured_split_check(records, expected or {})
    return {"files": [r["path"] for r in records], "source_windows": "disjoint", "original_sample_intervals": "disjoint", "expected_config": configured}


def crossroute_check(records):
    import numpy as np
    groups = {}
    for record in records:
        groups.setdefault((record["method"], record["directory"]), []).append(record)
    require(set(m for m, _ in groups) == set(METHODS), "all RBD, ELM, SCM routes are required")
    def signature(group):
        return sorted((r["split"], tuple(r["sources"].tolist()), tuple(r["sample_starts"].tolist()), tuple(r["sample_stops"].tolist()), tuple(r["freq"].tolist())) for r in group)
    matched = []
    for (method, directory), group in groups.items():
        peers = []
        for other in METHODS:
            if other == method:
                continue
            candidates = [(d, g) for (m, d), g in groups.items() if m == other and signature(g) == signature(group)]
            require(candidates, f"{directory}: no {other} route shares splits, source windows and frequencies")
            for peer_directory, peer in candidates:
                for r in group:
                    corresponding = [p for p in peer if p["split"] == r["split"] and np.array_equal(p["sources"], r["sources"])]
                    require(len(corresponding) == 1, "ambiguous corresponding HDF5 split")
                    p = corresponding[0]
                    for key in ("labels", "valid", "starts", "centers", "stops", "depth"):
                        require(np.allclose(r[key], p[key], rtol=1e-6, atol=1e-7), f"{directory} vs {peer_directory}: common {key} differs")
                peers.append(peer_directory)
        matched.append({"method": method, "directory": directory, "matching_routes": sorted(set(peers))})
    return {"matched_datasets": matched}


def validator_controls_check():
    """Acceptance data must pass, and deliberately corrupted contracts must fail."""
    import h5py
    import numpy as np
    def rejected(function, expected_text):
        try:
            function()
        except AssertionError as exc:
            require(expected_text.casefold() in str(exc).casefold(), f"wrong rejection: {exc}; expected {expected_text}")
        else:
            raise AssertionError(f"corrupted fixture was accepted; expected {expected_text}")
    with tempfile.TemporaryDirectory(prefix="swellex-validator-") as tmp:
        directory = Path(tmp)
        records = []
        fixtures = {}
        for method in METHODS:
            route = directory / method
            route.mkdir()
            for split, offset in (("train", 1), ("test", 11)):
                path = route / f"{method}_fixture_{split}.h5"
                write_fixture(path, method, False)
                sources = np.arange(7) + offset
                starts = sources.astype(np.float64)
                with h5py.File(path, "r+") as h5:
                    h5["y_range_km"][...] = starts + 1.5
                    h5["valid_sample"][...] = np.ones(7)
                    h5["split/source_segment_idx"][...] = sources
                    h5["time/window_center_s"][...] = starts + 0.5
                    h5["time/window_start_s"] = starts
                    h5["time/window_stop_s"] = starts + 1.0
                    h5["time/sample_start_idx"] = sources * 100 + 1
                    h5["time/sample_stop_idx"] = sources * 100 + 100
                    h5["range/range_time_s"] = np.array([0.0, 100.0])
                    h5["range/range_km_raw"] = np.array([1.0, 101.0])
                record, _ = inspect_h5(path, method, {})
                records.append(record)
                fixtures[(method, split)] = path
            split_check([r for r in records if r["method"] == method])
        crossroute_check(records)
        path = fixtures[("RBD", "train")]
        with h5py.File(path, "r+") as h5:
            h5["X"][0, 0, 0, 0] = np.nan
        rejected(lambda: inspect_h5(path, "RBD", {}), "nonfinite /X")
        with h5py.File(path, "r+") as h5:
            h5["X"][0, 0, 0, 0] = 0.0
            del h5["X"].attrs["axis_order"]
        rejected(lambda: inspect_h5(path, "RBD", {}), "axis_order")
        path = fixtures[("ELM", "train")]
        with h5py.File(path, "r+") as h5:
            for key in ("pair/numerator_element_idx", "pair/denominator_element_idx"):
                h5[key][...] = np.asarray(h5[key])[::-1]
        rejected(lambda: inspect_h5(path, "ELM", {}), "pair order/semantics")
        path = fixtures[("SCM", "train")]
        with h5py.File(path, "r+") as h5:
            h5["pair/numerator_element_idx"][0] = 2
        require(classify(path) == "SCM", "corrupt SCM metadata was misclassified")
        rejected(lambda: inspect_h5(path, "SCM", {}), "pair order/semantics")
        rbd = [r for r in records if r["method"] == "RBD"]
        corrupted = dict(rbd[1])
        corrupted["sample_starts"] = corrupted["sample_starts"].copy()
        corrupted["sample_stops"] = corrupted["sample_stops"].copy()
        corrupted["sample_starts"][0] = rbd[0]["sample_starts"][-1]
        corrupted["sample_stops"][0] = rbd[0]["sample_stops"][-1]
        rejected(lambda: split_check([rbd[0], corrupted]), "overlapping original sample intervals")
        changed_label = [dict(r) for r in records]
        changed_label[-1]["labels"] = changed_label[-1]["labels"] + 1.0
        rejected(lambda: crossroute_check(changed_label), "common labels differs")
        # A separate physical RBD fixture distinguishes normalized Green columns from raw FFT amplitudes.
        rbd_path = fixtures[("RBD", "train")]
        canonical = normalize_expected({"schema_version": 1, "dataset": {"fs": 100, "segment_start_s": 1,
            "segment_end_s": 17, "frequency_selection_modes": ["deep"],
            "frequency_selection_config": {"deep_target_freq_hz": list(range(0, 51, 5))}},
            "rbd": {"beam_selection": "best"}, "result_path": "outputs/validation/control.json"})
        with h5py.File(rbd_path, "r+") as h5:
            physical = np.zeros_like(h5["X"])
            physical[..., 0] = 0.5  # Four elements: sum |g|^2 = 1.
            physical[:, :, 0, :] = 0.0  # Truly zero columns remain permitted.
            h5["X"][...] = physical
            h5["X"].attrs["axis_order"] = "sample,element,frequency,real_imag"
        inspect_h5(rbd_path, "RBD", canonical)
        with h5py.File(rbd_path, "r+") as h5:
            h5["X"][...] = physical * 16.0
        rejected(lambda: inspect_h5(rbd_path, "RBD", canonical), "RBD best Green column norm")
        multipath = normalize_expected({**canonical, "rbd": {**canonical["rbd"], "beam_selection": "multipath"}})
        with h5py.File(rbd_path, "r+") as h5:
            h5["X"][...] = physical * 0.5
        inspect_h5(rbd_path, "RBD", multipath)
        rejected(lambda: inspect_h5(rbd_path, "RBD", canonical), "RBD best Green column norm")
        with h5py.File(rbd_path, "r+") as h5:
            h5["X"][...] = physical * 2.0
        rejected(lambda: inspect_h5(rbd_path, "RBD", multipath), "RBD multipath Green column norm")
        rejected(lambda: normalize_expected({"dataset": {"segment_typo_s": 1}}), "unknown dataset fields")
        rejected(lambda: normalize_expected({"rbd": {"beam_selecton": "best"}}), "unknown rbd fields")
    return {"positive_controls": "three complete routes accepted", "rejected_corruptions": ["nonfinite features", "missing axis metadata",
            "ELM pair order", "SCM pair semantics", "split raw sample leakage", "cross-route label inconsistency",
            "FFT amplitude instead of normalized Green", "multipath norm above one", "unknown canonical config fields"]}


def datasets(report, root, expected):
    paths = sorted(p for p in root.rglob("*.h5") if not p.name.startswith("._"))
    report.check("datasets.discovery", lambda: require(paths, f"no generated HDF5 files under {root}") or {"root": str(root), "files": len(paths)})
    records = []
    for path in paths:
        def inspect(p=path):
            method = classify(p)
            directory = str(p.parent)
            if directory not in report.data["dataset_directories"][method]:
                report.data["dataset_directories"][method].append(directory)
            record, details = inspect_h5(p, method, expected)
            records.append(record)
            return details
        report.check(f"dataset.file.{path.relative_to(root)}", inspect)
    groups = {}
    for record in records:
        groups.setdefault((record["method"], record["directory"]), []).append(record)
    for (method, directory), group in sorted(groups.items()):
        report.check(f"dataset.splits.{method}.{Path(directory).name}", lambda g=group: split_check(g, expected))
    report.check("datasets.cross_route_consistency", lambda: crossroute_check(records))
    report.data["dataset_directories"] = {m: sorted(v) for m, v in report.data["dataset_directories"].items()}
    report.data["datasets"] = {m: dirs[0] for m, dirs in report.data["dataset_directories"].items() if len(dirs) == 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    quick_parser = sub.add_parser("quick", help="Validate imports, CLIs, loaders and all twelve models")
    quick_parser.add_argument("--device", required=True, help="Explicit cpu/cuda/cuda:N; CUDA never silently falls back")
    quick_parser.add_argument("--report", type=Path, required=True)
    dataset_parser = sub.add_parser("datasets", help="Validate all generated datasets and cross-route equivalence")
    dataset_parser.add_argument("--root", type=Path, required=True)
    dataset_parser.add_argument("--report", type=Path, required=True)
    dataset_parser.add_argument("--expected-config", type=Path, help="Canonical Python generation JSON, or flat num_elements/freq_hz/min_samples expectations; unknown fields fail")
    args = parser.parse_args()
    report = Report(args.command, args.report.resolve())
    try:
        if args.command == "quick":
            quick(report, args.device)
        else:
            expected = {}
            def read_expected():
                nonlocal expected
                raw = json.loads(args.expected_config.read_text()) if args.expected_config else {}
                expected = normalize_expected(raw)
                return {"format": "canonical Python generation" if "dataset" in expected else "flat expectations", "config": expected}
            if report.check("datasets.expected_config", read_expected):
                datasets(report, args.root.resolve(), expected)
    except Exception as exc:
        report.check("validation.unhandled_error", lambda: (_ for _ in ()).throw(exc))
    report.save()
    print(f"{report.data['status']} report={report.output}", flush=True)
    return 0 if report.data["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
