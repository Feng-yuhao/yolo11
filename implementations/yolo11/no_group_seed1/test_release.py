"""Independent local unit tests, including failed/stale gate rejection."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(HERE.parent/'tmp/e58_e59_unit_settings'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(HERE.parent/'yolo11_src'))
import torch
torch.set_num_threads(2)
import spatial_task_head as target
import e57_spatial_task_head as e57
from checks import routing_unit
from pair_safety import CAPS,write_json,validate_smoke_evidence,check_pair_receipt


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ultralytics.nn.tasks import DetectionModel,yaml_model_load
        cfg=yaml_model_load(str(HERE/'e1_reference.yaml'));cfg['scale']='s'
        cls.base=DetectionModel(cfg,nc=10,verbose=False)

    def test_counts_widths_rng_and_common_initialization(self):
        parent=e57.attach(copy.deepcopy(self.base))
        for name,count in (('e58',1344672),('e59',486816)):
            target.configure(name)
            rng=torch.get_rng_state().clone()
            model=target.attach(copy.deepcopy(self.base))
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            self.assertEqual(sum(p.numel() for p in target.branch(model).parameters()),count)
            head=model.model[-1]
            self.assertEqual(head.nl,4)
            self.assertEqual([b.reduce[0].in_channels for b in head.task_blocks],[128,128,256,512])
            self.assertEqual([b.reduce[0].out_channels for b in head.task_blocks],[96]*4 if name=='e58' else [48]*4)
            for key,value in self.base.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
            if name=='e59':
                for key,value in parent.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)
                # Nonzero original projections cannot hide a changed spatial path.
                for i in (0,1):
                    a=head.task_blocks[i];b=parent.model[-1].task_blocks[i]
                    with torch.no_grad():
                        a.out_cls.weight.normal_(0,.02);a.out_reg.weight.normal_(0,.02)
                        b.load_state_dict({k:v for k,v in a.state_dict().items() if k in b.state_dict()})
                    x=torch.randn(2,128,9,11)
                    for actual,expected in zip(a(x),b(x)):
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            groups=target.branch_groups()
            for parameter,_ in target.branch(model).named_parameters():
                self.assertEqual(sum(parameter.startswith(g) for g in groups),1,parameter)
            self.assertEqual(len(groups),28 if name=='e58' else 34)

    def test_independent_task_depth_channel_oracles(self):
        for name in ('e58','e59'):
            target.configure(name)
            self.assertEqual(set(routing_unit('cpu')),{'128','256','512'})

    def test_new_branch_zero_start_and_real_updates(self):
        for name in ('e58','e59'):
            target.configure(name)
            model=target.attach(copy.deepcopy(self.base))
            for i,block in enumerate(target.branch(model)):
                cin=block.reduce[0].in_channels
                x=torch.randn(2,cin,9,11,requires_grad=True)
                original=x.detach().clone()
                objectives=[torch.randn_like(x),torch.randn_like(x)]
                opt=torch.optim.SGD(block.parameters(),lr=.03)
                before={n:p.detach().clone() for n,p in block.named_parameters()}
                for step in range(3):
                    opt.zero_grad(set_to_none=True)
                    c,r=block(x)
                    if step==0:
                        torch.testing.assert_close(c,x,rtol=0,atol=0)
                        torch.testing.assert_close(r,x,rtol=0,atol=0)
                    ((c*objectives[0]+r*objectives[1]).sum()/20).backward()
                    for n,p in block.named_parameters():
                        self.assertIsNotNone(p.grad,n)
                        self.assertTrue(torch.isfinite(p.grad).all(),n)
                        if step==0 and not n.startswith('out_'):
                            self.assertEqual(int(p.grad.count_nonzero()),0,n)
                    opt.step()
                for g in target.branch_groups():
                    if not g.startswith(str(i)+'.'): continue
                    local=g[len(str(i))+1:]
                    self.assertTrue(any(not torch.equal(p,before[n]) for n,p in block.named_parameters()
                                        if n.startswith(local)),name+'/'+g)
                self.assertTrue(torch.isfinite(x.grad).all() and x.grad.abs().sum()>0)
                torch.testing.assert_close(x.detach(),original,rtol=0,atol=0)

    def test_channel_weights_not_spatial_and_task_branch_gradients(self):
        block=target.initialize_channel_readout(target.parent.SpatialTaskBlock())
        x=torch.randn(2,128,9,11,requires_grad=True)
        _,_,_,bank=block.decompose(x)
        c,r,w=block.channel_decompose(bank)
        self.assertEqual(tuple(w.shape),(2,2,4,48,1,1))
        with torch.no_grad():
            block.out_channel_cls.weight.normal_(0,.02)
            block.out_channel_reg.weight.normal_(0,.02)
        cls,reg=block(x)
        grads=torch.autograd.grad(cls.square().mean(),[block.out_channel_cls.weight,block.out_channel_reg.weight],
                                  allow_unused=True,retain_graph=True)
        self.assertIsNotNone(grads[0]);self.assertIsNone(grads[1])
        grads=torch.autograd.grad(reg.square().mean(),[block.out_channel_cls.weight,block.out_channel_reg.weight],
                                  allow_unused=True)
        self.assertIsNone(grads[0]);self.assertIsNotNone(grads[1])


def smoke_fixture(name,session):
    count=28 if name=='e58' else 34
    groups=[str(i) for i in range(count)]
    return dict(status='passed',variant=name,pair_session=str(session),successful_optimizer_updates=8,
                branch_updated=True,updated_groups=groups,expected_groups=groups,
                allocator_cap_GiB=CAPS[name],peak_reserved_GiB=CAPS[name]-.5,
                validation=dict(validation_images=548,validation_batch=32,EMA_half=True,square_batch32=True,
                                validation_loss_and_NMS=True,parallel_pair=True))


class SafetyTests(unittest.TestCase):
    def test_bounded_smoke_uses_gib_over_gib_not_bytes_over_gib(self):
        import smoke_memory
        gib = 2**30
        for name in CAPS:
            cuda = mock.Mock()
            cuda.mem_get_info.return_value = (28*gib, 32229*2**20)
            smoke_memory.configure(name)
            free,total = smoke_memory.check_headroom(cuda,'cuda:0')
            self.assertEqual(free,28.)
            self.assertEqual(total,32229/1024)
            with smoke_memory.bounded_smoke(cuda,'cuda:0'):
                pass
            fraction,device = cuda.set_per_process_memory_fraction.call_args.args
            self.assertAlmostEqual(fraction,CAPS[name]/(32229/1024))
            self.assertTrue(0 < fraction < 1)
            self.assertEqual(device,'cuda:0')
        cuda = mock.Mock()
        cuda.mem_get_info.return_value = (1*gib,32*gib)
        with self.assertRaises(RuntimeError):
            with smoke_memory.bounded_smoke(cuda,'cuda:0'):
                self.fail('Insufficient headroom must not yield')
        cuda.set_per_process_memory_fraction.assert_not_called()
        cuda.mem_get_info.return_value = (8*gib,8*gib)
        smoke_memory.configure('e58')
        with self.assertRaises(RuntimeError):
            with smoke_memory.bounded_smoke(cuda,'cuda:0'):
                self.fail('Budget above device size must not yield')
        cuda.set_per_process_memory_fraction.assert_not_called()

    def test_smoke_rejects_skips_missing_validation_and_wrong_cap(self):
        for name in CAPS:
            value=smoke_fixture(name,Path('fixture'))
            validate_smoke_evidence(value,name,Path('fixture'))
            for field,changed in (('successful_optimizer_updates',7),('allocator_cap_GiB',28),
                                  ('branch_updated',False),('updated_groups',[]),('peak_reserved_GiB',29)):
                bad=copy.deepcopy(value);bad[field]=changed
                with self.assertRaises(RuntimeError): validate_smoke_evidence(bad,name,Path('fixture'))
            for field,changed in (('validation_images',547),('validation_batch',16),('parallel_pair',False)):
                bad=copy.deepcopy(value);bad['validation'][field]=changed
                with self.assertRaises(RuntimeError): validate_smoke_evidence(bad,name,Path('fixture'))

    def test_joint_receipt_rejects_stale_failed_or_missing_barriers(self):
        with tempfile.TemporaryDirectory() as temp:
            project=Path(temp);session=project/'logs/pair_smoke';session.mkdir(parents=True)
            signatures={n:{'signature':n} for n in CAPS};reports={}
            for name in CAPS:
                root=project/name;root.mkdir();reports[name]=str(root)
                write_json(root/'smoke.json',smoke_fixture(name,session))
                write_json(session/f'{name}_report.json',dict(report=str(root)))
                write_json(project/'comparison_reports'/f'{name}_smoke_gate.json',
                           dict(signature=signatures[name],report=str(root)))
                for stage in ('training_done','validation_start','validation_done'):
                    write_json(session/f'{name}_{stage}.json',dict(stage=stage))
            write_json(session/'state.json',dict(exit_codes={'e58':0,'e59':0}))
            write_json(project/'comparison_reports/e58_e59_pair_smoke_gate.json',
                       dict(status='passed',session=str(session),reports=reports,signatures=signatures))
            check_pair_receipt(project,signatures)
            with self.assertRaises(RuntimeError):check_pair_receipt(project,{'bad':True})
            write_json(session/'abort.json',dict(error='failed'))
            with self.assertRaises(RuntimeError):check_pair_receipt(project,signatures)
            (session/'abort.json').unlink()
            write_json(project/'comparison_reports/e59_smoke_gate.json',dict(signature=signatures['e59'],report='stale'))
            with self.assertRaises(RuntimeError):check_pair_receipt(project,signatures)
            write_json(project/'comparison_reports/e59_smoke_gate.json',dict(signature=signatures['e59'],report=reports['e59']))
            (session/'e59_validation_done.json').unlink()
            with self.assertRaises(RuntimeError):check_pair_receipt(project,signatures)

    def test_pair_signatures_are_symmetric_and_do_not_mutate(self):
        from run_experiment import both_signatures
        from pair_safety import EXPERIMENTS
        first=dict(variant='e58',allocator_cap_GiB=15.,params=dict(name=EXPERIMENTS['e58'],batch=16),
                   package={'source':'same'},GPU={'name':'fixture'})
        saved=copy.deepcopy(first)
        pair=both_signatures(first)
        self.assertEqual(first,saved)
        self.assertEqual(both_signatures(pair['e59']),pair)
        for name in CAPS:
            self.assertEqual(pair[name]['variant'],name)
            self.assertEqual(pair[name]['params']['name'],EXPERIMENTS[name])
            self.assertEqual(pair[name]['allocator_cap_GiB'],CAPS[name])

    def test_partial_launch_keeps_first_pid_and_does_not_kill_it(self):
        import run_pair
        first=SimpleNamespace(pid=12345)
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(project=Path(temp))
            with mock.patch.object(run_pair,'command',return_value=['fixture']), \
                    mock.patch.object(run_pair.subprocess,'Popen',side_effect=[first,OSError('fixture spawn failure')]):
                with self.assertRaises(OSError):run_pair.launch(args,'run')
            pointer=json.loads((args.project/'logs/latest_e58_e59_run.json').read_text())
            state=json.loads((Path(pointer['session'])/'state.json').read_text())
            self.assertEqual(state['pids'],{'e58':12345})
            self.assertIn('fixture spawn failure',state['controller_error'])

    def test_one_formal_failure_does_not_abort_other_job(self):
        import run_pair
        first=SimpleNamespace(pid=12345,poll=lambda:1)
        polls=iter([None,0])
        second=SimpleNamespace(pid=12346,poll=lambda:next(polls))
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(project=Path(temp))
            with mock.patch.object(run_pair,'command',return_value=['fixture']), \
                    mock.patch.object(run_pair.subprocess,'Popen',side_effect=[first,second]), \
                    mock.patch.object(run_pair,'gpu_state',return_value={}), \
                    mock.patch.object(run_pair.time,'sleep'):
                with self.assertRaises(RuntimeError):run_pair.launch(args,'run')
            pointer=json.loads((args.project/'logs/latest_e58_e59_run.json').read_text())
            session=Path(pointer['session'])
            state=json.loads((session/'state.json').read_text())
            self.assertEqual(state['exit_codes'],{'e58':1,'e59':0})
            self.assertFalse((session/'abort.json').exists())

    def test_formal_gpu_admission(self):
        from run_pair import require_idle
        require_idle(dict(total_MiB=32229,free_MiB=32229,compute_pids=[]))
        for state in (dict(total_MiB=16000,free_MiB=16000,compute_pids=[]),
                      dict(total_MiB=32229,free_MiB=25000,compute_pids=[]),
                      dict(total_MiB=32229,free_MiB=32229,compute_pids=[123])):
            with self.assertRaises(RuntimeError):require_idle(state)


if __name__=='__main__':unittest.main(verbosity=2)
