"""CPU-only regression tests; no SSH, GPU, real training or shutdown calls."""
import copy, tempfile, unittest,os,sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parent))
import common7 as c
import controller as ctl
import shutdown_guard as guard
import preservation7 as preservation
import summary7

if os.environ.get('ST_MDRF_TEST_TMP_DIR'):tempfile.tempdir=os.environ['ST_MDRF_TEST_TMP_DIR']

class SafetyTests(unittest.TestCase):
    def test_exact_seven_and_schedule(self):
        self.assertEqual(len(c.JOBS),7)
        self.assertEqual(sorted((v['variant'],v['seed']) for v in c.JOBS.values()),sorted([('shared',2),('shared',3),('uniform',2),('uniform',3),('e58',3),('tood_td',2),('tood_td',3)]))
        self.assertEqual([k for w in c.WAVES for k in w],list(c.JOBS))
        self.assertTrue(all(len(w)<=2 for w in ctl.planned_waves('parallel')))
        self.assertTrue(all(len(w)==1 for w in ctl.planned_waves('serial')))
    def test_oom_only_fallback(self):
        self.assertEqual(ctl.pair_decision([dict(exit_code=0),dict(exit_code=0)]),'parallel')
        for cat in ['cuda_oom','gpu_headroom']:
            self.assertEqual(ctl.pair_decision([dict(exit_code=42,category=cat),dict(exit_code=1,category='peer_failed')]),'serial')
        for cats in [['error'],['peer_failed'],['cuda_oom','error']]:
            with self.assertRaises(RuntimeError):ctl.pair_decision([dict(exit_code=1,category=v) for v in cats])
    def test_receipt_stale_corrupt_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);b=dict(seed=2,job='uniform')
            c.record_receipt(out,'smoke',b,dict(status='passed',epochs=2))
            self.assertEqual(c.verify_receipt(out,'smoke',b)['status'],'passed')
            with self.assertRaises(RuntimeError):c.verify_receipt(out,'smoke',dict(seed=3))
            c.write(out/'failure.json',dict(status='failed'))
            with self.assertRaises(RuntimeError):c.verify_receipt(out,'smoke',b)
            (out/'failure.json').unlink();c.write(out/'smoke.json',dict(status='passed',epochs=3))
            with self.assertRaises(RuntimeError):c.verify_receipt(out,'smoke',b)
    def test_all_gates_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c.write(root/'admission.json',dict(binding_id='A'))
            for key,job in c.JOBS.items():
                for phase in ['check','smoke']:
                    c.record_receipt(root/'jobs'/key/phase,phase,dict(admission='A',key=key,job=job),dict(status='passed'))
            self.assertEqual(ctl.gates_ready(root),'A')
            (root/'jobs/uniform_s3/smoke/smoke_receipt.json').unlink()
            with self.assertRaises(FileNotFoundError):ctl.gates_ready(root)
    def fixture_state(self):
        return dict(status='completed',stage='completed',auto_shutdown=True,all_7_formal200_saved=True,all_14_official_scores_completed=True,
                    completion_seal_sha256='seal',session='/p/logs/s',report='/p/comparison_reports/r',jobs={})
    def test_no_shutdown_if_failed_partial_live_cancelled_or_new_run(self):
        state=self.fixture_state();idx=dict(session=state['session'],report=state['report'])
        def check(s=state,i=idx,live=False,cancel=False):return guard.eligible(s,i,state['session'],state['report'],live,cancel)
        self.assertTrue(check())
        for change in [dict(status='failed'),dict(status='running'),dict(stage='check'),dict(auto_shutdown=False),
                       dict(all_7_formal200_saved=False),dict(all_14_official_scores_completed=False),dict(completion_seal_sha256='')]:
            with self.assertRaises(RuntimeError):check({**state,**change})
        with self.assertRaises(RuntimeError):check(live=True)
        with self.assertRaises(RuntimeError):check(cancel=True)
        with self.assertRaises(RuntimeError):check(i=dict(session='/p/logs/new',report=idx['report']))
        with self.assertRaises(RuntimeError):check({**state,'jobs':dict(x=dict(status='running'))})
        with self.assertRaises(RuntimeError):check({**state,'jobs':dict(x=dict(status='failed',exit_code=1,phase='train',out='/r/train',category='cuda_oom'))})
    def test_only_oom_joint_failure_allowed_in_final_state(self):
        s=self.fixture_state();idx=dict(session=s['session'],report=s['report'])
        s['jobs']['joint']=dict(status='failed',exit_code=42,phase='smoke',out='/r/pair_gates/wave1/shared_s2',category='cuda_oom')
        self.assertTrue(guard.eligible(s,idx,s['session'],s['report'],False,False))
        s['jobs']['joint']['category']='error'
        with self.assertRaises(RuntimeError):guard.eligible(s,idx,s['session'],s['report'],False,False)
    def test_idle_rejects_unknown_python_or_cuda(self):
        idle=dict(total_MiB=32760,free_MiB=32000,compute_pids=[]);p=dict(jobs=[],ignored_services=[])
        guard.safety.require_idle(idle,p)
        for gpu,procs in [({**idle,'compute_pids':[99]},p),(idle,dict(jobs=[dict(pid=1)])),({**idle,'free_MiB':20000},p)]:
            with self.assertRaises(RuntimeError):guard.safety.require_idle(gpu,procs)
    def test_no_generic_poweroff_and_no_shebang_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'shutdown'
            p.write_bytes(b'echo "user execute shutdown command in container" > /proc/1/fd/1\nsupervisord stop\n')
            v=guard.safety.shutdown_invocation(p)
            self.assertEqual(v['args'],['/bin/bash',str(p)])
            p.write_bytes(b'#!/bin/sh\npoweroff\n')
            with self.assertRaises(RuntimeError):guard.safety.shutdown_invocation(p)
            p.write_bytes(b'\x7fELF\x00')
            with self.assertRaises(RuntimeError):guard.safety.shutdown_invocation(p)
    def test_mocked_command_runs_once_only_with_unchanged_helper(self):
        command=dict(args=['/bin/bash','/usr/bin/shutdown'],sha256='abc',bytes=205)
        runner=Mock(return_value=Mock(returncode=0,stdout='accepted'))
        with patch.object(guard.sys,'platform','linux'),patch.object(guard.safety,'shutdown_invocation',return_value=command):
            self.assertEqual(guard.invoke_verified(command,runner)['status'],'shutdown_command_returned_zero')
        self.assertEqual(runner.call_count,1)
        for changed in [{**command,'sha256':'changed'},{**command,'args':['poweroff']}]:
            with patch.object(guard.sys,'platform','linux'),patch.object(guard.safety,'shutdown_invocation',return_value=changed):
                with self.assertRaises(RuntimeError):guard.invoke_verified(command,runner)
        self.assertEqual(runner.call_count,1)
    def test_seal_rejects_missing_or_modified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'result.json';c.write(f,dict(metric=1))
            with patch.object(preservation,'result_paths',return_value=[f]),patch.object(preservation,'package_audit',return_value='pkg'),patch.object(preservation,'completed_inventory',return_value={}):
                preservation.seal_results(root,root,proof={});value=c.sha(root/'COMPLETION_SEAL.json')
                preservation.verify_seal(root,root,value)
                c.write(f,dict(metric=2))
                with self.assertRaises(RuntimeError):preservation.verify_seal(root,root,value)
                f.unlink()
                with self.assertRaises(RuntimeError):preservation.verify_seal(root,root,value)
    def test_epoch_csv_exact_200(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'results.csv'
            c.text(f,'epoch,loss\n'+''.join(f'{i},1\n' for i in range(1,201)))
            self.assertEqual(len(c.csv_epochs(f)),200)
            c.text(f,'epoch,loss\n1,1\n200,1\n')
            with self.assertRaises(RuntimeError):c.csv_epochs(f)
    def test_cached_seed_evidence_and_statistics(self):
        rows=summary7.cached_rows();groups,paired=summary7.aggregate(rows)
        self.assertAlmostEqual(groups['val/e60']['metrics']['AP50_95_percent']['mean'],26.846717573229904,places=9)
        self.assertNotIn('test-dev/shared',groups)
        r=next(r for r in paired if r['split']=='val' and r['candidate']=='e60' and r['reference']=='e1' and r['metric']=='AP50_95_percent')
        self.assertAlmostEqual(r['mean_pp'],1.341636084189457,places=8)
        self.assertEqual(r['positive_seeds'],3)
    def test_incomplete_artifacts_never_count_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):c.completed_inventory(Path(tmp),Path(tmp))
    def test_native_fp16_diagnostic_is_not_ap_or_train_failure(self):
        values,diag=c.native_metrics_for_json({'val/cls_loss':float('inf'),'metrics/mAP50-95(B)':.26})
        self.assertEqual(values['val/cls_loss'],'inf');self.assertEqual(values['metrics/mAP50-95(B)'],.26);self.assertTrue(diag)
        for bad in [{'metrics/mAP50-95(B)':float('inf')},{'val/box_loss':float('inf')},{'val/cls_loss':float('nan')}]:
            with self.assertRaises(RuntimeError):c.native_metrics_for_json(bad)
    def test_actual_controller_order_and_oom_downgrade(self):
        class Fake(ctl.Controller):
            def arm_guard(self):self.events.append(('armed',None))
            def mark_scored(self,key):self.events.append(('scored',key))
            def single(self,key,phase,out,split=None):self.events.append((phase,key))
            def start(self,key,phase,out,cap=28.,pair=None,split=None):
                tag=str(len(self.pending));self.pending[tag]=dict(key=key,phase=phase,out=str(out),exit_code=0)
                self.events.append((phase,key));return tag
            def wait(self,tags):
                results=[self.pending[t] for t in tags]
                if self.oom and results[0]['phase']=='smoke':
                    results[0].update(exit_code=42,category='cuda_oom');results[1].update(exit_code=1,category='peer_failed')
                return results
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/'report';session=root/'logs';report.mkdir();session.mkdir()
            c.write(report/'admission.json',dict(binding_id='A'))
            for oom in [False,True]:
                # Each simulated run has its own pair-gate path and plan.
                r=report/str(oom);r.mkdir();c.write(r/'admission.json',dict(binding_id='A'))
                obj=Fake(SimpleNamespace(serial=False,enable_shutdown=True,project=root),session,r,
                         dict(admission=dict(gpu=dict(total_MiB=32760)),jobs={},stage='init'))
                obj.events=[];obj.pending={};obj.oom=oom
                def mock_seal(*args):c.write(r/'COMPLETION_SEAL.json',dict(status='mock_only'))
                with patch.object(ctl,'gates_ready',return_value='A'),patch.object(ctl,'completed_inventory',return_value={}),patch.object(summary7,'summarize'),patch.object(preservation,'seal_results',side_effect=mock_seal):
                    obj.execute()
                phase_list=[event[0] for event in obj.events]
                first_train=phase_list.index('train')
                self.assertEqual(phase_list[:first_train].count('check'),7)
                self.assertGreaterEqual(phase_list[:first_train].count('smoke'),7)
                self.assertEqual(phase_list.count('train'),7);self.assertEqual(phase_list.count('score'),14)
                self.assertEqual(obj.state['status'],'completed')
                plan=c.read(r/'execution_plan.json')
                self.assertEqual(plan['execution'],'serial' if oom else 'parallel')
                self.assertTrue(all(len(w)==1 for w in plan['waves']) if oom else max(map(len,plan['waves']))==2)
    def test_preflight_failure_never_starts_formal_or_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            obj=ctl.Controller(SimpleNamespace(serial=False,enable_shutdown=True,project=root),root,root,dict(stage='init',jobs={}))
            def fail(key,phase,out,split=None):
                events.append((phase,key))
                if phase=='check':raise RuntimeError('synthetic engineering failure')
            with patch.object(obj,'arm_guard'),patch.object(obj,'single',side_effect=fail),patch.object(obj,'start') as start,patch.object(guard,'invoke_verified') as shutdown:
                with self.assertRaises(RuntimeError):obj.execute()
                self.assertFalse(start.called);self.assertFalse(shutdown.called)
            self.assertNotIn('train',[p for p,k in events])
    def test_full_completion_schema_and_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);report=project/'comparison_reports/synthetic';report.mkdir(parents=True)
            protocol=dict(imgsz=640,device='cuda:0',conf=.001,iou=.7,max_det=500,half=False,augment=False,
                          verbose=False,save=False,agnostic_nms=False,classes=None,batch=4)
            for key,job in c.JOBS.items():
                root=report/'jobs'/key;run=project/'runs'/job['run'];(run/'weights').mkdir(parents=True)
                c.text(run/'results.csv','epoch,loss\n'+''.join(f'{i},1\n' for i in range(1,201)))
                for name in ['best.pt','last.pt']:(run/'weights'/name).write_bytes(b'test_fixture_not_torch')
                cp_sha=c.sha(run/'weights/best.pt')
                c.write(root/'train/training_completion.json',dict(status='completed',completed_epochs=200,seed=job['seed'],key=key,
                    official_scores_completed=True,best_sha256=cp_sha,last_sha256=cp_sha,
                    binding=dict(key=key,job=job,admission='A',params=dict(seed=job['seed'],epochs=200,batch=16))))
                for split,count in [('val',548),('test-dev',1610)]:
                    out=root/split;c.write(out/'area_predictions.json',[]);pred_sha=c.sha(out/'area_predictions.json')
                    b=dict(checkpoint_sha256=cp_sha,split=split,images=count,package_sha256='P',admission_id='A',seed=job['seed'],
                           scorer_sha256=c.sha(c.HERE/'frozen'/job['kind']/'d27_official.py'),protocol=protocol,smoke_only=False)
                    c.write(out/'official_metrics.json',dict(status='completed',key=key,seed=job['seed'],binding=b,prediction_sha256=pred_sha,official={k:25. for k in c.KEYS}))
                    c.write(out/'prediction_manifest.json',dict(status='completed',binding=b,seen_images=count,sha256=pred_sha))
            c.write(report/'summary.json',dict(status='completed'));c.text(report/'REPORT.md','synthetic unit test only')
            def mock_checkpoint(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=c.sha(p))
            with patch.object(c,'checkpoint_check',side_effect=mock_checkpoint),patch.object(c,'package_audit',return_value='P'):
                self.assertEqual(len(c.completed_inventory(project,report)),7)
                p=report/'jobs/shared_s2/test-dev/official_metrics.json';original=c.read(p)
                for field,value in [('images',548),('checkpoint_sha256','wrong'),('seed',3),('package_sha256','wrong')]:
                    changed=copy.deepcopy(original);changed['binding'][field]=value;c.write(p,changed)
                    with self.assertRaises(RuntimeError):c.completed_inventory(project,report)
                c.write(p,original)
                (report/'jobs/uniform_s3/test-dev/area_predictions.json').unlink()
                with self.assertRaises(FileNotFoundError):c.completed_inventory(project,report)
    def guard_fixture(self,project):
        project=Path(project);session=project/'logs/synthetic';report=project/'comparison_reports/synthetic'
        session.mkdir(parents=True);report.mkdir(parents=True)
        command=dict(args=['/bin/bash','/usr/bin/shutdown'],sha256='abc',bytes=205)
        state=self.fixture_state();state.update(session=str(session),report=str(report),controller_identity=dict(pid=99,start_ticks=1),
            admission=dict(shutdown_invocation=command))
        c.write(session/'state.json',state);c.write(project/'logs/latest_stmdrf_supp7.json',dict(session=str(session),report=str(report)))
        return session,report,command,state
    def test_whole_guard_success_calls_mock_only_after_two_seal_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            session,report,command,state=self.guard_fixture(tmp)
            events=[]
            def seal(*args):events.append('seal')
            def invoke(*args):events.append('shutdown');return dict(status='mock_command_only')
            with patch.object(guard.safety,'process_identity',return_value=None),patch.object(guard.safety,'shutdown_invocation',return_value=command),\
                 patch.object(guard,'verify_seal',side_effect=seal),patch.object(guard,'assert_lock_free'),patch.object(guard,'idle_now',return_value=dict(idle=True)),\
                 patch.object(guard.time,'monotonic',side_effect=[0.,121.]),patch.object(guard.time,'sleep'),patch.object(guard.os,'sync',create=True),\
                 patch.object(guard,'invoke_verified',side_effect=invoke):
                guard.run(tmp,session,report)
            self.assertEqual(events,['seal','seal','shutdown'])
    def test_whole_guard_failed_and_cancelled_never_calls_command(self):
        for mode in ['failed','cancelled','corrupt']:
            with tempfile.TemporaryDirectory() as tmp:
                session,report,command,state=self.guard_fixture(tmp)
                if mode=='failed':state['status']='failed';c.write(session/'state.json',state)
                if mode=='cancelled':(Path(tmp)/'KEEP_RUNNING_ST_MDRF_SUPP7').touch()
                with patch.object(guard.safety,'process_identity',return_value=None),patch.object(guard.safety,'shutdown_invocation',return_value=command),\
                     patch.object(guard,'verify_seal',side_effect=RuntimeError('Corrupted preservation seal')),patch.object(guard,'assert_lock_free'),\
                     patch.object(guard,'invoke_verified') as invoke:
                    guard.run(tmp,session,report);self.assertFalse(invoke.called)
                self.assertIn(c.read(session/'shutdown_status.json')['status'],['disarmed_incomplete_or_failed','cancelled_by_user','blocked_safety_check'])
    def test_whole_guard_unknown_active_job_prevents_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            session,report,command,state=self.guard_fixture(tmp)
            with patch.object(guard.safety,'process_identity',return_value=None),patch.object(guard.safety,'shutdown_invocation',return_value=command),\
                 patch.object(guard,'verify_seal'),patch.object(guard,'assert_lock_free'),patch.object(guard,'idle_now',side_effect=RuntimeError('Other GPU/Python tasks active')),\
                 patch.object(guard.time,'sleep',side_effect=RuntimeError('end mock wait')),patch.object(guard,'invoke_verified') as invoke:
                with self.assertRaisesRegex(RuntimeError,'end mock wait'):guard.run(tmp,session,report)
                self.assertFalse(invoke.called)
            self.assertEqual(c.read(session/'shutdown_status.json')['status'],'waiting_idle')

if __name__=='__main__':unittest.main(verbosity=2)
