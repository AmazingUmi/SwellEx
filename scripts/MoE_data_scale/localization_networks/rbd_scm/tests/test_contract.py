"""Standard-library checks for frozen joint physical axes and fitted identity."""
from copy import deepcopy
import unittest

from scripts.MoE_data_scale.localization_networks.rbd_scm.contract import validate_signature, validate_preprocessing
from scripts.MoE_data_scale.localization_networks.common.input_contract import validate_saved_input_signature


def joint_signature():
    axes = {"frequency_hz": [100., 105., 110.], "element_idx": [1, 2], "depth_m": [100., 110.]}
    rbd = dict(axes, method="RBD", input_shape=[2, 2, 3], pair_indices=[])
    scm = dict(deepcopy(axes), method="SCM", input_shape=[2, 3, 3], pair_indices=[[1, 1], [1, 2], [2, 2]])
    return {"format_version": 1, "method": "RBD_SCM", "input_shape": [2, 5, 3], "fusion_strategy": "block_rms",
            "blocks": [{"method": "RBD", "start": 0, "stop": 2, "input_signature": rbd},
                       {"method": "SCM", "start": 2, "stop": 5, "input_signature": scm}],
            "source_contract": {"input_files": [{"path": "signal.dat", "size_bytes": 12, "mtime_ns": 1, "sha256": "a" * 64}],
                                "effective_config": {"dataset": {"num_snapshots_per_segment": 1, "snapshot_overlap_count": 0}},
                                "split": {"strategy": "train_before"}, "generator_version": "test-v1"}}


class JointSchemaContracts(unittest.TestCase):
    def test_complete_joint_signature_uses_features_and_both_physical_blocks(self):
        validate_signature(joint_signature(), (2, 5, 3), "RBD_SCM")
        validate_saved_input_signature(joint_signature(), (2, 5, 3), "RBD_SCM")

    def test_axis_order_pair_order_fusion_and_snapshot_errors_reject(self):
        alterations = [lambda s: s["blocks"].reverse(),
                       lambda s: s["blocks"][1]["input_signature"]["pair_indices"].reverse(),
                       lambda s: s.update(fusion_strategy="raw_concat"),
                       lambda s: s["blocks"][1]["input_signature"]["frequency_hz"].__setitem__(1, 106.),
                       lambda s: s["source_contract"]["effective_config"]["dataset"].update(num_snapshots_per_segment=2),
                       lambda s: s["blocks"][1].update(start=3),
                       lambda s: s["source_contract"]["input_files"][0].update(sha256="invalid")]
        for mutate in alterations:
            with self.subTest(mutation=mutate):
                signature = deepcopy(joint_signature())
                mutate(signature)
                with self.assertRaises(ValueError):
                    validate_signature(signature, (2, 5, 3), "RBD_SCM")

    def test_preprocessing_requires_positive_scales_and_training_membership(self):
        state = {"strategy": "block_rms", "scales": [1., 2.], "fit_sample_count": 4,
                 "fit_source_identity_sha256": "b" * 64}
        validate_preprocessing(state, joint_signature())
        for key, bad in (("strategy", "raw_concat"), ("scales", [0., 1.]), ("scales", [float('nan'), 1.]),
                         ("scales", [True, 1.]), ("fit_sample_count", 0), ("fit_sample_count", True),
                         ("fit_source_identity_sha256", "oops")):
            with self.subTest(field=key, bad=bad):
                with self.assertRaises(ValueError):
                    validate_preprocessing(dict(state, **{key: bad}), joint_signature())
        del state["fit_source_identity_sha256"]
        with self.assertRaises(ValueError):
            validate_preprocessing(state, joint_signature())


if __name__ == "__main__":
    unittest.main()
