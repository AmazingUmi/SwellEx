"""RBD dependencies for the shared neural runtime."""
from scripts.network_training.data_scale_pack.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("RBD", model, data, paths, data.RbdRangeH5Dataset, "input_elements")
