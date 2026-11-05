"""Method training binding; shared runtime owns continuation and exports."""
from scripts.network_training.data_scale_pack.common.training import train as shared_train
from .bindings import BINDINGS


def train(args):
    return shared_train(args, BINDINGS)
