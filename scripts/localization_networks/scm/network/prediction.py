"""Method prediction binding; shared runtime owns artifact validation/exports."""
from scripts.localization_networks.common.prediction import predict as shared_predict
from .bindings import BINDINGS


def predict(args):
    return shared_predict(args, BINDINGS)
