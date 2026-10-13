"""Remote Python, loader, model, and numerical contract checks."""
from __future__ import annotations
import ast
import importlib
import os
from pathlib import Path
import sys
import tempfile
from scripts.base_pack.localization_networks.common.paths import PROJECT_ROOT as ROOT
from scripts.base_pack.localization_networks.workflow.processes import run_process
from scripts.base_pack.localization_networks.workflow.tasks import METHODS, entry_script, package_name, resolve_selection
from scripts.base_pack.localization_networks.validation.reporting import require
from scripts.base_pack.localization_networks.validation.fixtures import write_fixture
from scripts.base_pack.localization_networks.validation.datasets import validator_controls_check

SCRIPTS = ROOT / "scripts" / "base_pack" / "localization_networks"
PACKAGE = "scripts.base_pack.localization_networks"
DATA_CLASSES = {"RBD": "RbdRangeH5Dataset", "ELM": "ElmRangeH5Dataset", "SCM": "ScmRangeH5Dataset"}


def data_module(method):
    return importlib.import_module(f"{PACKAGE}.{package_name(method)}.network.data")


def model_module(method):
    return importlib.import_module(f"{PACKAGE}.{package_name(method)}.network.model")


def grnn_module(name):
    return importlib.import_module(f"{PACKAGE}.grnn.{name}")


def syntax_check():
    scripts = ROOT / "scripts"
    paths = sorted({p for directory in (scripts / "base_pack", scripts / "tests")
                    for p in directory.rglob("*.py") if not p.name.startswith("._")}
                   | {p for p in scripts.glob("*.py") if not p.name.startswith("._")})
    failures = []
    for path in paths:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeError) as exc:
            failures.append(f"{path}: {exc}")
    require(not failures, "\n".join(failures))
    return {"files": len(paths)}


def imports_check():
    names = [f"{PACKAGE}.common.{path.stem}" for path in sorted((SCRIPTS / "common").glob("*.py"))
             if path.name != "__init__.py" and not path.name.startswith("._")]
    names += [f"{PACKAGE}.grnn.{name}" for name in ("bindings", "model", "data", "paths", "cli")]
    names.append("scripts.base_pack.data_processing.generation_config")
    for method in METHODS:
        for path in sorted((SCRIPTS / f"{package_name(method)}" / "network").rglob("*.py")):
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


def cli_check(method, *, grnn=False):
    selection = resolve_selection(method, "GRNN" if grnn else "complex_CNN")
    entry = ROOT / entry_script(selection)
    commands = ([], ["build" if grnn else "train"], ["predict"])
    results = []
    for command in commands:
        proc = run_process([sys.executable, str(entry), *command, "--help"], cwd=ROOT,
                              text=True, capture_output=True, timeout=90)
        require(proc.returncode == 0 and "usage:" in proc.stdout.lower(),
                f"{entry} {' '.join(command)} --help: exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        results.append({"arguments": [*command, "--help"], "exit_code": proc.returncode})
    parser_module = grnn_module("cli") if grnn else importlib.import_module(f"{PACKAGE}.{package_name(method)}.network.cli")
    parser = parser_module.build_parser()
    if not grnn:
        require(set(parser_module.MODEL_CHOICES) == set(model_module(method).MODEL_REGISTRY), "CLI model choices differ from registry")
    else:
        for command in ("build", "predict"):
            parsed = parser.parse_args([command, "--method", method, "--data", "fixture"])
            parser_module.validate_arguments(parsed, parser)
            require(parsed.method == method, "GRNN CLI lost explicit feature method")
    return {"method": method, "commands": results}


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
    proc = run_process([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", "test*.py", "-v"],
                          cwd=ROOT, text=True, capture_output=True, timeout=180)
    require(proc.returncode == 0, f"workflow unittest exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def package_tests_check(relative):
    directory = ROOT / relative
    require(directory.is_dir() and list(directory.glob("test*.py")), f"missing test files: {relative}")
    proc = run_process([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", "test*.py", "-v"],
                       cwd=ROOT, text=True, capture_output=True, timeout=240)
    require(proc.returncode == 0, f"{relative}: exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def workflow_registry_check():
    from scripts.base_pack.localization_networks.workflow.tasks import ARCHITECTURES, METHODS as TASK_METHODS, resolve_selection

    require(tuple(TASK_METHODS) == tuple(METHODS), "public workflow methods differ from validated methods")
    neural = [name for name in ARCHITECTURES if name != "GRNN"]
    selections = []
    for method in METHODS:
        registry = model_module(method).MODEL_REGISTRY
        mapped = [resolve_selection(method, name) for name in neural]
        require({row["model"] for row in mapped} == set(registry),
                f"{method} public architecture mapping differs from its actual model registry")
        require(not any(name.endswith("_grnn_range") for name in registry), "GRNN must remain outside trainable registries")
        selections.extend(mapped)
    selections.extend(resolve_selection(method, "GRNN") for method in METHODS)
    return {"selections": selections, "count": len(selections)}


def visualization_tests_check():
    directory = ROOT / "scripts" / "base_pack" / "visualization" / "tests"
    require(directory.is_dir() and list(directory.glob("test*.py")), "visualization unittest files are missing")
    proc = run_process([sys.executable, "-m", "unittest", "discover", "-s", str(directory), "-p", "test*.py", "-v"],
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
    from scripts.base_pack.localization_networks.common import h5_utils, paths
    from scripts.base_pack.localization_networks.common.training_utils import target_stats
    require(paths.PROJECT_ROOT == ROOT, "common project root points outside the checkout root")
    require(grnn_module("paths").PROJECT_ROOT == ROOT, "GRNN project root points outside the checkout root")
    for count in (2, 7, 41):
        train, val = h5_utils.split_indices(count, 0.25, 2026)
        require(not set(train) & set(val), f"split overlaps for n={count}")
        require(sorted(train + val) == list(range(count)), f"split loses samples for n={count}")
        require(train and val, f"empty split for n={count}")
        require((train, val) == h5_utils.split_indices(count, 0.25, 2026), "seed is not repeatable")
    for count, fraction in ((1, 0.25), (0, 0.25), (2, 0), (2, 1), (2, float("nan"))):
        try:
            h5_utils.split_indices(count, fraction, 2026)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid split accepted: n={count}, fraction={fraction}")
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
        junk = paths.with_time_suffix(fixed, "not-a-time"); junk.touch()
        invalid = paths.with_time_suffix(fixed, "0231_250000"); invalid.touch()
        os.utime(junk, (30, 30)); os.utime(invalid, (40, 40))
        require(paths.latest_time_suffixed_path(fixed) == newer, "latest checkpoint must use mtime and only valid timestamp suffixes")
        require(paths.dataset_dir(str(directory)) == directory, "explicit dataset path not honored")
        require(paths.train_output_dir(directory, "model") == directory / "model" / "train_outputs", "training path mismatch")
        require(paths.train_output_dir(directory / "model", "model") == directory / "model" / "train_outputs", "model path duplicated")
    return {"split_sizes": [2, 7, 41], "invalid_splits": "rejected", "path_resolution": "sorted unique, explicit paths, mtime checkpoints"}


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
            load_grnn_features = grnn_module("data").load_grnn_features
            bundle = load_grnn_features([fixture], method=method)
            expected = np.stack([features[s].transpose(2, 0, 1).reshape(-1) for s in keep])
            require(np.array_equal(bundle.x.numpy(), expected), f"GRNN {method} flatten order differs from CHW contract")
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
                small_grnn = load_grnn_features([small], method=method)
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
    GrnnRegressor = grnn_module("model").GrnnRegressor
    reference = torch.tensor([[-1.0, 2.0], [1.0, 2.0]])
    targets = torch.tensor([[2.0], [8.0]])
    queries = torch.tensor([[-1.0, 2.0], [0.0, 2.0], [1.0, 2.0], [100.0, 2.0]])
    model = GrnnRegressor(spread=0.01, device=device)
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
    restored = GrnnRegressor.from_artifact(model.to_artifact(), device=device)
    require(torch.equal(values, restored.predict(queries)), "artifact round trip changes predictions")
    permuted = GrnnRegressor(spread=0.01, device=device)
    permuted.fit_reference(reference.flip(0), targets.flip(0))
    require(torch.allclose(values, permuted.predict(queries), atol=1e-6), "reference order affects predictions")
    standardized = GrnnRegressor(spread=0.3, standardize_input=True, device=device)
    standardized.fit_reference(reference, targets)
    rescaled = GrnnRegressor(spread=0.3, standardize_input=True, device=device)
    scale = torch.tensor([3.0, 4.0]); shift = torch.tensor([9.0, -7.0])
    rescaled.fit_reference(reference * scale + shift, targets)
    require(torch.allclose(standardized.predict(queries), rescaled.predict(queries * scale + shift), atol=2e-5), "standardized GRNN not invariant under feature units/offset")
    constant = GrnnRegressor(spread=0.1, standardize_input=True, device=device)
    constant.fit_reference(reference, torch.full((2, 1), 3.25))
    require(torch.allclose(constant.predict(queries), torch.full((4, 1), 3.25)), "GRNN does not preserve constant targets")
    for spread in (0.0, -0.1):
        try:
            GrnnRegressor(spread=spread, device=device)
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
    for method in METHODS:
        report.check(f"cli.{method}", lambda m=method: cli_check(m))
        report.check(f"cli.GRNN.{method}", lambda m=method: cli_check(m, grnn=True))
    usable = report.check("runtime.device", lambda: device_check(device))
    report.check("shared.paths_and_splits", shared_check)
    report.check("workflow.unit_tests", workflow_tests_check)
    report.check("packs.dispatch_tests", lambda: package_tests_check("scripts/tests"))
    report.check("workflow.model_registry", workflow_registry_check)
    report.check("visualization.unit_tests", visualization_tests_check)
    report.check("network.contract_tests", lambda: package_tests_check("scripts/base_pack/localization_networks/common/tests"))
    report.check("grnn.contract_tests", lambda: package_tests_check("scripts/base_pack/localization_networks/grnn/tests"))
    report.check("generation.identity_tests", lambda: package_tests_check("scripts/base_pack/data_processing/tests"))
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
