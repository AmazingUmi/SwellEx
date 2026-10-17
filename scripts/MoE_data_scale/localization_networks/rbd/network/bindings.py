"""RBD dependencies for the shared neural runtime."""
from scripts.MoE_data_scale.localization_networks.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("RBD", model, data, paths, data.RbdRangeH5Dataset, "input_elements")
