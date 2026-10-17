"""RBD_SCM dependencies for the shared neural runtime."""
from scripts.MoE_data_scale.localization_networks.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("RBD_SCM", model, data, paths, data.JointRangeH5Dataset, "input_features")
