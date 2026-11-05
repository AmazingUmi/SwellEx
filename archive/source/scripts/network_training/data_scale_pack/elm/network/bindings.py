"""ELM dependencies for the shared neural runtime."""
from scripts.network_training.data_scale_pack.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("ELM", model, data, paths, data.ElmRangeH5Dataset, "input_pairs")
