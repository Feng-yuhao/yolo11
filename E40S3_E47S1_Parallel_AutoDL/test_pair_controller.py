"""Local controller tests with no torch, CUDA, subprocess training or remote IO."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import parallel_safety as safety
import run_pair


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='e40s3_e47s1_test_')
        self.project = Path(self.tmp.name).resolve()
        self.args = types.SimpleNamespace(project=self.project, source=self.project / 'source',
                                          data=self.project / 'data.yaml', dataset=self.project / 'val',
                                          device='cuda:0', workers=8, mode='smoke')
        self.gpu = dict(index=0, uuid='GPU-fixture', name='mock 32GB', total_MiB=32229,
                        free_MiB=31000, used_MiB=1229, utilization_percent=0, compute_pids=[])
        self.binding = dict(root='fixture', project=str(self.project), protocol=safety.protocol(),
                            memory_policy=safety.memory_policy(), package_digest='current-code',
                            GPU=safety.gpu_identity(self.gpu))
        self.binding_patch = patch.object(safety, 'runtime_binding', return_value=self.binding)
        self.binding_patch.start()
        self.pid_patch = patch.object(safety, 'process_identity', return_value=dict(pid=345, start_ticks=12, ppid=1))
        self.pid_patch.start()
        self.log = self.project / 'logs' / 'pair_e40s3_e47s1_smoke_fixture'
        self.log.mkdir(parents=True)
        self.context = safety.make_context(self.args, 'smoke', self.log, self.gpu, 'fixture-invocation')
        self.create_gates()

    def tearDown(self):
        self.binding_patch.stop()
        self.pid_patch.stop()
        self.tmp.cleanup()

    def markers(self):
        stages = {}
        for i, name in enumerate(safety.NAMES):
            stages[name] = {}
            for label, arrived, released in [('training_ready', 1 + i, 3 + i),
                                              ('before_validation', 10 + i, 12 + i),
                                              ('complete', 20 + i, 22 + i)]:
                stages[name][label] = dict(name=name, label=label, token=self.context['token'],
                                          invocation=self.context['invocation'], pid=100 + i,
                                          start_ticks=10 + i, entered_monotonic=arrived,
                                          released_monotonic=released)
        return stages

    def create_gates(self):
        stages = self.markers()
        for name in safety.NAMES:
            signature = dict(name=name, release='current-code', seed=3 if name == 'e40' else 1)
            for phase, filename in zip(('check', 'smoke'), safety.GATES[name]):
                out = self.project / 'comparison_reports' / f'{name}_{phase}'
                evidence = dict(status='passed', cuda_executed=True)
                if phase == 'smoke':
                    evidence = dict(status='passed', real_batches=8, successful_optimizer_updates=8,
                                    updates=[dict(updated=True) for _ in range(8)], post_update_inference=True,
                                    branch_updated=True, updated_groups=safety.expected_groups(name), allocator_cap_GiB=13.,
                                    other_job_reserve_GiB=2., peak_reserved_GiB=12.5,
                                    validation=dict(validation_batch=32, validation_images=548,
                                                    loss_precision='autocast_float16', raw_logits_and_decoded_finite=True,
                                                    parallel_pair=True, EMA_half=True, validation_loss_and_NMS=True,
                                                    square_batch32=True),
                                    pair_context=dict(token=self.context['token'], controller=self.context['controller'],
                                                      invocation=self.context['invocation'], binding=self.binding,
                                                      worker=name, phase='smoke', stages=stages,
                                                      context=str(self.log / 'pair_context.json'),
                                                      context_sha256=safety.file_hash(self.log / 'pair_context.json')))
                safety.write_json(out / f'{phase}.json', evidence)
                safety.write_json(self.project / 'comparison_reports' / filename,
                                  dict(signature=signature, report=str(out)))

    def alter_smoke(self, name, change):
        path = self.project / 'comparison_reports' / f'{name}_smoke' / 'smoke.json'
        value = safety.read_json(path)
        change(value)
        safety.write_json(path, value)

    def env(self, context=None):
        context = context or self.context
        return {safety.ENV_PREFIX + 'PAIR': '1', safety.ENV_PREFIX + 'PHASE': context['phase'],
                safety.ENV_PREFIX + 'TOKEN': context['token'],
                safety.ENV_PREFIX + 'CONTEXT': str(Path(context['log_dir']) / 'pair_context.json')}

    def test_memory_budget_and_idle_boundary(self):
        self.assertEqual(safety.memory_policy()['combined_allocator_cap_GiB'], 26)
        gpu = {**self.gpu, 'free_MiB': 30720}
        safety.require_idle_gpu(gpu)
        for changed in ({'free_MiB': 30719}, {'compute_pids': [900]}, {'total_MiB': 30000}):
            with self.subTest(changed=changed), self.assertRaises(RuntimeError):
                safety.require_idle_gpu({**gpu, **changed})

    def test_real_pair_receipt_roundtrip(self):
        receipt = safety.build_pair_receipt(self.args, self.context)
        self.assertEqual(receipt['overlap_seconds'], dict(training=6, validation=7))
        safety.write_json(self.project / 'comparison_reports' / safety.RECEIPT_NAME, receipt)
        self.assertEqual(safety.verify_pair_receipt(self.args)['status'], 'passed')

    def test_standalone_or_mixed_smoke_rejected(self):
        for mutate in (lambda v: v.pop('pair_context'),
                       lambda v: v['pair_context'].update(token='other-token'),
                       lambda v: v['pair_context'].update(invocation='other-invocation')):
            self.create_gates()
            self.alter_smoke('e47', mutate)
            with self.assertRaises(RuntimeError):
                safety.build_pair_receipt(self.args, self.context)

    def test_smoke_requires_updates_full_val_cap_and_gradients(self):
        changes = [dict(successful_optimizer_updates=7), dict(branch_updated=False), dict(real_batches=7),
                   dict(allocator_cap_GiB=28), dict(peak_reserved_GiB=13.01), dict(other_job_reserve_GiB=0),
                   dict(updated_groups=[]), dict(validation=dict(validation_batch=32, validation_images=547))]
        for change in changes:
            with self.subTest(change=change):
                self.create_gates()
                self.alter_smoke('e40', lambda v: v.update(change))
                with self.assertRaises(RuntimeError):
                    safety.build_pair_receipt(self.args, self.context)

    def test_overlapping_train_and_validation_required(self):
        for stage in ('before_validation', 'complete'):
            self.create_gates()
            for name in safety.NAMES:
                self.alter_smoke(name, lambda v: v['pair_context']['stages']['e40'][stage].update(entered_monotonic=0))
            with self.assertRaisesRegex(RuntimeError, 'overlapping'):
                safety.build_pair_receipt(self.args, self.context)

    def test_signature_cpu_check_failed_and_changed_receipts_rejected(self):
        check = self.project / 'comparison_reports' / 'e40_check' / 'check.json'
        safety.write_json(check, dict(status='passed', cuda_executed=False))
        with self.assertRaisesRegex(RuntimeError, 'CPU'):
            safety.build_pair_receipt(self.args, self.context)
        self.create_gates()
        receipt = safety.build_pair_receipt(self.args, self.context)
        safety.write_json(self.project / 'comparison_reports' / safety.RECEIPT_NAME, receipt)
        self.alter_smoke('e47', lambda v: v.update(peak_reserved_GiB=12.4))
        with self.assertRaisesRegex(RuntimeError, 'stale/changed'):
            safety.verify_pair_receipt(self.args)

    def test_worker_env_is_not_sufficient_without_ancestry(self):
        with patch.dict(os.environ, self.env()), patch.object(safety, 'is_descendant', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'owns'):
                safety.validate_worker_context(self.args, 'e40', 'smoke')
        with patch.dict(os.environ, {safety.ENV_PREFIX + 'PAIR': '1'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'PAIR flag'):
                safety.validate_worker_context(self.args, 'e40', 'smoke')

    def test_worker_context_validates_phase_token_pid_and_binding(self):
        with patch.dict(os.environ, self.env()), patch.object(safety, 'is_descendant', return_value=True):
            self.assertEqual(safety.validate_worker_context(self.args, 'e47', 'smoke')['token'], self.context['token'])
            with patch.dict(os.environ, {safety.ENV_PREFIX + 'TOKEN': 'stale'}):
                with self.assertRaisesRegex(RuntimeError, 'token'):
                    safety.validate_worker_context(self.args, 'e47', 'smoke')
            with patch.object(safety, 'process_identity', return_value=dict(pid=345, start_ticks=13, ppid=1)):
                with self.assertRaisesRegex(RuntimeError, 'start_ticks'):
                    safety.validate_worker_context(self.args, 'e47', 'smoke')
            with patch.object(safety, 'runtime_binding', return_value={'GPU': 'changed'}):
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    safety.validate_worker_context(self.args, 'e47', 'smoke')

    def test_peer_run_directory_does_not_break_live_context(self):
        receipt = safety.build_pair_receipt(self.args, self.context)
        safety.write_json(self.project / 'comparison_reports' / safety.RECEIPT_NAME, receipt)
        run_log = self.project / 'logs' / 'pair_e40s3_e47s1_run_fixture'
        run_log.mkdir()
        context = safety.make_context(self.args, 'run', run_log, self.gpu, 'run-invocation')
        (self.project / 'runs' / safety.EXPERIMENTS['e40']).mkdir(parents=True)
        with patch.dict(os.environ, self.env(context)), patch.object(safety, 'is_descendant', return_value=True):
            self.assertEqual(safety.validate_worker_context(self.args, 'e47', 'run')['phase'], 'run')
        with self.assertRaisesRegex(RuntimeError, 'existing formal run'):
            run_pair.check_run_destinations(self.args)

    def test_check_runs_sequentially_live_and_keeps_second_after_failure(self):
        calls = []
        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return types.SimpleNamespace(returncode=7 if len(calls) == 1 else 0)
        codes = [run_pair.run_one(name, 'check', self.args, runner) for name in safety.NAMES]
        self.assertEqual(codes, [7, 0])
        self.assertNotIn('capture_output', calls[0][1])
        self.assertIn('--workers', calls[0][0])

    def fake_children(self, fail_first=False, interrupt=False):
        children = []
        class Child:
            def __init__(self, idx):
                self.pid = 1000 + idx
                self.index = idx
                self.polls = 0
            def poll(self):
                self.polls += 1
                if interrupt and self.polls == 2:
                    raise KeyboardInterrupt()
                if self.index == 0:
                    return 9 if fail_first else 0
                return None if self.polls == 1 else 0
            def kill(self):
                raise AssertionError('Controller must not kill a worker')
            terminate = kill
        def popen(command, **kwargs):
            self.assertEqual(kwargs['stdin'], run_pair.subprocess.DEVNULL)
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['env'][safety.ENV_PREFIX + 'PAIR'], '1')
            child = Child(len(children)); children.append(child)
            return child
        return popen, children

    def test_formal_one_failure_other_finishes_without_termination(self):
        self.args.mode = 'run'
        safety.write_json(self.project / 'comparison_reports' / safety.RECEIPT_NAME, {'fixture': True})
        popen, children = self.fake_children(fail_first=True)
        with patch.object(safety, 'gpu_state', return_value=self.gpu):
            self.assertEqual(run_pair.launch_pair('run', self.args, self.gpu, popen, sleep=lambda _: None), 1)
        state = safety.read_json(next((self.project / 'logs').glob('pair_e40s3_e47s1_run_*/state.json')))
        self.assertEqual(state['exit_codes'], dict(e40=9, e47=0))
        self.assertEqual(state['status'], 'failed')
        self.assertGreaterEqual(children[1].polls, 2)

    def test_launch_error_still_launches_peer(self):
        launched = []
        popen, _ = self.fake_children()
        def partial(command, **kwargs):
            launched.append(command)
            if len(launched) == 1:
                raise OSError('fixture launch failure')
            return popen(command, **kwargs)
        safety.write_json(self.project / 'comparison_reports' / safety.RECEIPT_NAME, {'fixture': True})
        with patch.object(safety, 'gpu_state', return_value=self.gpu):
            self.assertEqual(run_pair.launch_pair('run', self.args, self.gpu, partial, sleep=lambda _: None), 1)
        self.assertEqual(len(launched), 2)

    def test_interrupted_controller_preserves_pid_and_orphan_notice(self):
        safety.write_json(self.project / 'comparison_reports' / safety.RECEIPT_NAME, {'fixture': True})
        popen, _ = self.fake_children()
        def interruption(_):
            raise KeyboardInterrupt()
        with patch.object(safety, 'gpu_state', return_value=self.gpu), self.assertRaises(KeyboardInterrupt):
            run_pair.launch_pair('run', self.args, self.gpu, popen, sleep=interruption)
        state = safety.read_json(next((self.project / 'logs').glob('pair_e40s3_e47s1_run_*/state.json')))
        self.assertEqual(state['status'], 'controller_interrupted')
        self.assertIn('not terminated', state['worker_notice'])
        self.assertEqual(len(state['pids']), 2)

    def test_cpu_missing_dataset_is_explicitly_ineligible(self):
        self.args.device = 'cpu'
        result = safety.dataset_fingerprint(self.args)
        self.assertFalse(result['CUDA_gate_eligible'])
        self.assertEqual(result['status'], 'unavailable_local_cpu')

    def test_barriers_wait_for_both_and_issue_same_invocation_evidence(self):
        with patch.dict(os.environ, self.env()), patch.object(safety, 'is_descendant', return_value=True):
            with ThreadPoolExecutor(max_workers=2) as pool:
                for label in ('training_ready', 'before_validation', 'complete'):
                    calls = [pool.submit(safety.barrier, self.args, name, label, 3) for name in safety.NAMES]
                    for call in calls:
                        marker = call.result(timeout=5)
                        self.assertGreaterEqual(marker['released_monotonic'], marker['entered_monotonic'])
            a = safety.smoke_evidence(self.args, 'e40')
            b = safety.smoke_evidence(self.args, 'e47')
            self.assertEqual(a['stages'], b['stages'])
            self.assertEqual(a['token'], self.context['token'])

    def test_barrier_failure_is_explicit_and_never_waits_indefinitely(self):
        safety.write_json(self.log / 'e47_failed.json', dict(exit_code=9))
        with patch.dict(os.environ, self.env()), patch.object(safety, 'is_descendant', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'peer e47 failed'):
                safety.barrier(self.args, 'e40', 'training_ready', 1)

    def test_fixed_worker_command_forwards_all_paths(self):
        command = run_pair.worker_command('e47', 'run', self.args)
        self.assertIn(str(self.args.source), command)
        self.assertIn(str(self.args.data), command)
        self.assertIn(str(self.args.dataset), command)
        self.assertEqual(command[-4:], ['--device', 'cuda:0', '--workers', '8'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
