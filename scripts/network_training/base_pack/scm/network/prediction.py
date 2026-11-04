"""Method prediction binding; shared runtime owns artifact validation/exports."""
from scripts.network_training.base_pack.common.prediction import predict as shared_predict
from .bindings import BINDINGS


def predict(args):
    return shared_predict(args, BINDINGS)
