"""SCM dependencies for the shared neural runtime."""
from scripts.base_pack.localization_networks.common.bindings import MethodBindings
from . import data, model, paths

BINDINGS = MethodBindings("SCM", model, data, paths, data.ScmRangeH5Dataset, "input_pairs")
