"""New seed3/aggregation/shutdown integration tests; no actual shutdown."""
import copy
import csv
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import common as c
import protocol as p
import shutdown_after as s
import summarize as summary

ROOT=c.HERE.parents[1]/'audit/yolo26_seed3_serial/extra_tests'
ROOT.mkdir(parents=True,exist_ok=True)


def scores(report,seed):
    for split in ('val','test-dev'):
        for variant in p.VARIANTS:
            value=copy.deepcopy(p.read(p.REF/(variant+'_seed1_'+split.replace('-','_')+'.json')))
            value.update(seed=seed,variant=variant,split=split,local_fixture=False,
                         protocol=dict(imgsz=640,batch=4,half=False,one2many_NMS=True,conf=.001,iou=.7,max_det=500))
            destination=report/(variant+'_'+split.replace('-','_'))
            p.write(destination/'area_predictions.json',[])
            value['prediction_sha256']=c.sha256(destination/'area_predictions.json')
            p.write(destination/'official_metrics.json',value)
            p.write(destination/'completion.json',dict(status='completed',seed=seed,local_fixture=False))


def whole_fixture(project,seed=3):
    report=project/'comparison_reports'/('YOLO26_P2_E60_S'+str(seed)+'_ALL_PRIVATE_FIXTURE')
    session=project/'logs'/('yolo26_seed'+str(seed)+'_all_fixture')
    report.mkdir(parents=True);session.mkdir(parents=True)
    scores(report,seed)
    for variant in p.VARIANTS:
        run=project/'runs'/('yolo26s_p2_'+variant+'_img640_seed'+str(seed)+'_200e')
        (run/'weights').mkdir(parents=True)
        for name in ('best.pt','last.pt'):
            with zipfile.ZipFile(run/'weights'/name,'w') as archive:
                archive.writestr('PRIVATE_NOT_A_MODEL',b'0'*(1024*1024+1))
        with (run/'results.csv').open('w',newline='') as file:
            writer=csv.writer(file);writer.writerow(['epoch']);writer.writerows([n] for n in range(1,201))
        best=run/'weights/best.pt';digest=c.sha256(best)
        p.write(report/(variant+'_train')/'training_completion.json',dict(status='training_completed',epochs=200,
            seed=seed,variant=variant,run=str(run),best=str(best),best_sha256=digest))
        for split in ('val','test_dev'):
            path=report/(variant+'_'+split)/'official_metrics.json';value=p.read(path)
            value['checkpoint_sha256']=digest;p.write(path,value)
    state=dict(status='completed',seed=seed,admitted=True,auto_shutdown=True,child_active=False,execution='serial',
               session=str(session),report_root=str(report),training_report=str(report),local_fixture=False)
    p.write(session/'state.json',state)
    p.write(project/'logs'/('latest_yolo26_seed'+str(seed)+'_serial.json'),dict(session=str(session),report_root=str(report)))
    p.write(report/'comparison.json',dict(status='completed',seeds=[seed]));(report/'REPORT.md').write_text('PRIVATE MOCK')
    return report,session,state


class Seed3Tests(unittest.TestCase):
    def temp(self): return tempfile.TemporaryDirectory(dir=ROOT)

    def test_seed1_and_seed2_numeric_sources_byte_identical(self):
        for name,digest in p.read(p.REF/'frozen_seed2_files.json').items():
            self.assertEqual(c.sha256(c.HERE/name),digest,name)

    def test_named_start_script_cannot_omit_shutdown_flag(self):
        content=(c.HERE/'start_all_auto_shutdown.sh').read_text()
        self.assertIn('all --auto-shutdown',content)
        self.assertNotIn('seed2',content.lower())
        self.assertNotIn(b'\r',(c.HERE/'start_all_auto_shutdown.sh').read_bytes())

    def test_seed3_paths_do_not_touch_old_runs_or_gates(self):
        args=SimpleNamespace(project=Path('/project'),local_test=False)
        self.assertIn('seed3',str(p.gate_path(args)))
        for variant in p.VARIANTS:
            self.assertIn('seed3',p.actual_run_name(args,variant))
            self.assertNotIn('seed2',p.actual_run_name(args,variant))

    def test_completed_proof_needs_all_four_scores_and_predictions(self):
        with self.temp() as tmp:
            project=Path(tmp);report,session,state=whole_fixture(project)
            self.assertEqual(s.verify_completed(project,report),[])
            (report/'e60_test_dev/official_metrics.json').unlink()
            self.assertTrue(s.verify_completed(project,report))

    def test_changed_predictions_block_shutdown_proof(self):
        with self.temp() as tmp:
            project=Path(tmp);report,session,state=whole_fixture(project)
            p.write(report/'baseline_val/area_predictions.json',[{'changed':True}])
            self.assertTrue(any('Prediction' in x for x in s.verify_completed(project,report)))

    def test_preserved_artifact_change_is_detected(self):
        with self.temp() as tmp:
            project=Path(tmp);report,session,state=whole_fixture(project)
            before=s.preserve(project,session,report,'completed')
            self.assertTrue(s.artifacts_unchanged(before,full=True))
            (report/'REPORT.md').write_text('CHANGED PRIVATE MOCK')
            self.assertFalse(s.artifacts_unchanged(before))

    def test_missing_seed2_is_explicit_not_fake_three_seed_mean(self):
        with self.temp() as tmp:
            project=Path(tmp);report=project/'comparison_reports/fresh';report.mkdir(parents=True)
            scores(report,3);summary.summarize(report)
            value=p.read(report/'comparison.json')
            self.assertEqual(value['seeds'],[1,3])
            self.assertEqual(value['seed2_reference']['status'],'not_available')

    def test_complete_negative_seed2_is_included_not_selected_away(self):
        with self.temp() as tmp:
            project=Path(tmp);old,session,state=whole_fixture(project,2)
            p.write(session/'signature.json',dict(payload=dict(seed=2,layout='serial',train_batch=16,
                validation_batch=32,prediction_batch=4,imgsz=640,epochs=200,pretrained=c.WEIGHT_SHA,
                source=c.sha256(c.HERE/'training_source_sha256.json'))))
            for split in ('val','test_dev'):
                path=old/('e60_'+split)/'official_metrics.json';value=p.read(path)
                base=p.read(old/('baseline_'+split)/'official_metrics.json')
                for key in summary.KEYS:value['official_metrics'][key]=base['official_metrics'][key]-.25
                p.write(path,value)
            fresh=project/'comparison_reports/fresh';fresh.mkdir();scores(fresh,3)
            summary.summarize(fresh);value=p.read(fresh/'comparison.json')
            self.assertEqual(value['seeds'],[1,2,3])
            self.assertEqual(value['seed2_reference']['status'],'included')
            for split in ('val','test-dev'):
                stats=value['splits'][split]['aggregate']['AP50_95_percent']
                self.assertAlmostEqual(stats['paired_delta_pp']['2'],-.25)
            self.assertTrue((fresh/'references_used/e60_seed2_test_dev.json').is_file())

    def test_mismatched_seed2_protocol_is_flagged_not_mixed(self):
        with self.temp() as tmp:
            project=Path(tmp);old,session,state=whole_fixture(project,2)
            p.write(session/'signature.json',dict(payload=dict(seed=2,layout='parallel')))
            fresh=project/'comparison_reports/fresh';fresh.mkdir();scores(fresh,3)
            summary.summarize(fresh);value=p.read(fresh/'comparison.json')
            self.assertEqual(value['seeds'],[1,3])
            self.assertEqual(value['seed2_reference']['status'],'not_included_requires_review')

    def test_complete_pipeline_shutdown_failure_and_missing_score_mocks(self):
        from shutdown_safety import shutdown_invocation
        for scenario in ('completed','failed','missing_score','changing_artifact'):
            with self.subTest(scenario=scenario),self.temp() as tmp:
                project=Path(tmp);report,session,state=whole_fixture(project)
                command=project/'MOCK_SHUTDOWN_NEVER_EXECUTED'
                command.write_text('echo "user execute shutdown command in container" > /proc/1/fd/1\n# supervisord\n')
                state.update(controller_identity=dict(pid=123,start_ticks=456),shutdown_command=shutdown_invocation(command))
                if scenario=='failed': state['status']='failed'
                if scenario in ('failed','missing_score'):
                    (report/'e60_test_dev/official_metrics.json').unlink()
                p.write(session/'state.json',state)
                elapsed=[0.];calls=[]
                def sleep(seconds):
                    elapsed[0]+=seconds
                    if scenario=='changing_artifact': (report/'REPORT.md').write_text('CHANGED DURING COUNTDOWN')
                def runner(argv,**kwargs):
                    calls.append(argv);return SimpleNamespace(returncode=0)
                result=s.run_watch(session,project,runner=runner,sleeper=sleep,clock=lambda:elapsed[0],
                    identity=lambda pid:None,locks=lambda path:True,platform_name='linux',command=command,delay=5,
                    gpu=lambda:dict(total_MiB=32760,free_MiB=32760,compute_pids=[]),
                    processes=lambda:dict(jobs=[],ignored_services=[]))
                self.assertEqual(len(calls),int(scenario in ('completed','failed')))
                if calls:
                    self.assertEqual(calls[0],['/bin/bash',str(command)])
                    self.assertTrue((session/'PRESERVATION.json').is_file())
                elif scenario=='missing_score':
                    self.assertEqual(result['status'],'blocked_incomplete_completed_outputs')
                else:
                    self.assertIn('blocked_artifacts_changed',result['status'])

    def test_controller_dispatches_shutdown_only_after_four_score_attempts_and_log_copy(self):
        import controller
        for failed_stage in (None,'baseline_test_dev'):
            with self.subTest(failed_stage=failed_stage),self.temp() as tmp:
                project=Path(tmp)
                args=SimpleNamespace(mode='all',auto_shutdown=True,local_test=False,local_size=128,local_epochs=2,
                    project=project,source=project/'source',data=project/'data.yaml',val_dataset=project/'val',
                    dataset=project/'test',training_report=None)
                calls=[];spawned=[]
                class FakeProcess:
                    def __init__(self,command,**kwargs):
                        self.pid=123;self.stdout=iter(['PRIVATE MOCK NOT A REAL TRAINING\n'])
                        if Path(command[2]).name=='shutdown_after.py':
                            spawned.append(command)
                            self.key='shutdown'
                            if calls[-4:]!=['baseline_val','baseline_test_dev','e60_val','e60_test_dev']:
                                raise AssertionError('Shutdown dispatched before all scoring attempts')
                            info=p.read(project/'logs/latest_yolo26_seed3_serial.json')
                            report=Path(info['report_root'])
                            if not all((report/(key+'.log')).is_file() for key in calls):
                                raise AssertionError('Stage logs not copied before dispatch')
                            if not kwargs.get('start_new_session'): raise AssertionError('Helper is not independent')
                            return
                        mode=command[3];variant=command[command.index('--variant')+1]
                        split=command[command.index('--split')+1] if '--split' in command else None
                        self.key='check' if mode=='check' else variant+'_'+(split.replace('-','_') if split else mode)
                        calls.append(self.key)
                    def wait(self): return int(self.key==failed_stage)
                    def poll(self): return self.wait()
                with patch.object(controller,'arguments',return_value=args),patch.object(controller,'runtime'),\
                     patch('process_lock.acquire',return_value=SimpleNamespace(close=lambda:None)),\
                     patch('shutdown_safety.readiness',return_value=dict(command={'PRIVATE_MOCK':True})),\
                     patch('shutdown_safety.process_identity',return_value=dict(pid=123,start_ticks=456)),\
                     patch.object(controller,'signature',return_value=dict(fingerprint='private',payload={})),\
                     patch.object(controller,'issue_gate'),patch.object(controller,'verify_gate'),\
                     patch.object(controller.subprocess,'Popen',FakeProcess),patch('summarize.summarize'):
                    if failed_stage:
                        with self.assertRaisesRegex(RuntimeError,'Some stages failed'):controller.main()
                    else: controller.main()
                self.assertEqual(len(spawned),1)
                info=p.read(project/'logs/latest_yolo26_seed3_serial.json')
                final=p.read(Path(info['session'])/'state.json')
                self.assertTrue(final['auto_shutdown'])
                self.assertEqual(final['status'],'failed' if failed_stage else 'completed')


if __name__=='__main__': unittest.main()
