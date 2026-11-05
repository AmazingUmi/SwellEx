"""Method selection for one shared joint-feature neural registry."""
from functools import partial
from types import SimpleNamespace

from scripts.network_training.joint_feature_pack.common.bindings import MethodBindings
from scripts.network_training.joint_feature_pack.common.paths import PROJECT_ROOT
from . import data, model
from .contract import components


def make_bindings(method: str) -> MethodBindings:
    components(method)
    paths = SimpleNamespace(DEFAULT_OUTPUT_DIR=PROJECT_ROOT / "outputs" / "networks_results" / method)
    return MethodBindings(method, model, data.data_bindings(method), paths,
                          partial(data.JointRangeH5Dataset, method=method), "input_features")
