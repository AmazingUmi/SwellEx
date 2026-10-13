"""Method CLI backed by the shared neural argument contract."""
from scripts.base_pack.localization_networks.common.cli import build_parser as shared_parser, main as shared_main
from .bindings import BINDINGS

MODEL_CHOICES = sorted(BINDINGS.model.MODEL_REGISTRY)


def build_parser():
    return shared_parser(BINDINGS)


def main(argv=None):
    shared_main(BINDINGS, argv)
