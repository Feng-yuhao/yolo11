"""Local-only CPU test; run after setting PYTHONPATH to the pinned source."""
from pathlib import Path
import json
import sys
import unittest

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))

import torch
from torch import nn
from parent_spatial_task_head import SpatialTaskBlock, SpatialTaskDetect
from depth_readout import install_readouts
from interventions import STATES, intervention, audit_interventions, state_digest


def make_head():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(45002)
        head = SpatialTaskDetect(nc=10, ch=(128, 128, 256, 512))
        blocks = nn.ModuleList(SpatialTaskBlock() for _ in range(2))
        for block in blocks:
            install_readouts(block)
        head.task_blocks = blocks
        head.enabled = True
        head.stride = torch.tensor([4., 8., 16., 32.])
        return head.float().eval()


class InterventionTests(unittest.TestCase):
    def test_all_states_cpu(self):
        report = audit_interventions(make_head(), 'cpu')
        self.assertEqual(report['device'], 'cpu')
        self.assertEqual(report['states'], list(STATES))
        print(json.dumps(report, indent=2), flush=True)

    def test_invalid_state_leaves_model_unchanged(self):
        head = make_head()
        before = state_digest(head)
        with self.assertRaises(ValueError):
            with intervention(head, 'invalid'):
                pass
        self.assertEqual(state_digest(head), before)

    def test_nested_and_exception_restore(self):
        head = make_head()
        before = state_digest(head)
        with intervention(head, 'normal'):
            with self.assertRaises(RuntimeError):
                with intervention(head, 'p2_cls_off'):
                    pass
            self.assertTrue(head.task_blocks[0]._d45_intervention_active)
        self.assertEqual(state_digest(head), before)
        self.assertTrue(all(not hasattr(b, '_d45_intervention_active') for b in head.task_blocks))

    def test_training_rejected(self):
        head = make_head().train()
        with self.assertRaises(RuntimeError):
            with intervention(head, 'normal'):
                pass


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
