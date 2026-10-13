"""ELM dependencies for the shared neural runtime."""
from scripts.base_pack.localization_networks.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("ELM", model, data, paths, data.ElmRangeH5Dataset, "input_pairs")
