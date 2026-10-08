"""Method training binding; shared runtime owns continuation and exports."""
from scripts.localization_networks.common.training import train as shared_train
from .bindings import BINDINGS


def train(args):
    return shared_train(args, BINDINGS)
