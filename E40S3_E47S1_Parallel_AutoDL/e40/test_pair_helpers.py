"""Local standard-library guard tests; NOT server CUDA/memory proof."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from smoke_gate import validate_smoke_result
from smoke_memory import apply_pair_memory, bounded_smoke, check_headroom
from smoke_progress import SmokeProgress
from baseline_reference import EXPECTED_OFFICIAL, require_reproduction


def evidence():
    return dict(status='passed', real_batches=10, successful_optimizer_updates=8,
                updates=[dict(updated=True, status='finite_update', bad_parameter_count=0)]*8 +
                        [dict(updated=False, status='overflow_skip', bad_parameter_count=1)]*2,
                allocator_cap_GiB=13, other_job_reserve_GiB=2, peak_reserved_GiB=11.3,
                branch_updated=True, updated_groups=[str(i) for i in range(14)],
                post_update_inference=True, pair_context=dict(token='test'),
                validation=dict(validation_images=548, validation_batch=32, parallel_pair=True,
                                EMA_half=True, raw_logits_and_decoded_finite=True,
                                validation_loss_and_NMS=True, loss_precision='autocast_float16'))


class Guards(unittest.TestCase):
    def test_complete_evidence_and_fail_closed(self):
        self.assertTrue(validate_smoke_result(evidence()))
        for key, value in [('real_batches', 7), ('successful_optimizer_updates', 7),
                           ('allocator_cap_GiB', 14), ('other_job_reserve_GiB', 6),
                           ('peak_reserved_GiB', 13.1), ('pair_context', None),
                           ('branch_updated', False)]:
            changed = evidence(); changed[key] = value
            with self.assertRaises(RuntimeError, msg=key):
                validate_smoke_result(changed)
        changed = evidence(); changed['validation']['validation_images'] = 8
        with self.assertRaises(RuntimeError): validate_smoke_result(changed)

    def test_amp_skips_are_not_updates(self):
        progress = SmokeProgress()
        for i in range(10): progress.observe(evidence()['updates'][i])
        self.assertEqual((progress.attempts, progress.successful, progress.skipped), (10, 8, 2))
        progress.require_success()
        with self.assertRaises(RuntimeError): SmokeProgress().require_success()

    def test_cap_same_and_not_restored(self):
        fractions = []
        cuda = SimpleNamespace(mem_get_info=lambda device: (18*2**30, 32229*2**20),
                               set_per_process_memory_fraction=lambda fraction, device: fractions.append(fraction))
        with patch('parallel_safety.memory_policy', return_value=dict(allocator_cap_GiB=13)):
            apply_pair_memory(cuda, 'cuda:0')
        with bounded_smoke(cuda, 'cuda:0'): pass
        self.assertEqual(len(fractions), 1)
        self.assertEqual(fractions[0], 13*2**30/(32229*2**20))
        low = SimpleNamespace(mem_get_info=lambda device: (1.9*2**30, 32*2**30))
        with self.assertRaises(RuntimeError): check_headroom(low, 'cuda:0')

    def test_pinned_seed3_score(self):
        require_reproduction(EXPECTED_OFFICIAL)
        changed = dict(EXPECTED_OFFICIAL); changed['AP50_95_percent'] += .001
        with self.assertRaises(RuntimeError): require_reproduction(changed)


if __name__ == '__main__': unittest.main()

