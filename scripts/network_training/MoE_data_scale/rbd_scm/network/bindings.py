"""RBD_SCM dependencies for the shared neural runtime."""
from scripts.network_training.MoE_data_scale.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("RBD_SCM", model, data, paths, data.JointRangeH5Dataset, "input_features")
