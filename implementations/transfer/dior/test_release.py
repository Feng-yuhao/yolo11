"""Synthetic regression checks; never choose settings from target AP."""
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common import HERE,KEYS,PROTOCOL,boot,parser,require,write

class ReleaseTests(unittest.TestCase):
    temp_root=HERE
    def test_all12_keys(self):
        self.assertEqual(len(KEYS),12);self.assertEqual(len(set(KEYS)),12)
    def test_mapping_numpy_both_and_source_unmodified(self):
        import torch,numpy as np
        from mapping import map_scores
        r=torch.rand(2,14,21);original=r.clone();a=r.numpy()
        for t,indices in [('dior',[7,8,9,12]),('tiny',[4,5])]:
            expected=np.concatenate((a[:,:4],a[:,indices].max(axis=1,keepdims=True)),axis=1)
            np.testing.assert_array_equal(map_scores(r,t).numpy(),expected)
        torch.testing.assert_close(r,original,rtol=0,atol=0)
    def test_merge_before_nms(self):
        import torch
        from mapping import mapped_nms
        for target,i,j in [('dior',7,12),('tiny',4,5)]:
            r=torch.zeros(1,14,7);r[:,:2]=50;r[:,2:4]=10;r[0,i,0]=.9;r[0,j,1]=.8;saved=r.clone()
            self.assertEqual(len(mapped_nms(r,target)[0]),1);torch.testing.assert_close(r,saved,rtol=0,atol=0)
    def test_unmapped_scores_cannot_hide_target(self):
        import torch
        from mapping import mapped_nms
        r=torch.zeros(1,14,7);r[:,:2]=50;r[:,2:4]=10;r[0,10,0]=.999;r[0,7,0]=.8
        self.assertEqual(len(mapped_nms(r,'dior')[0]),1)
    def test_nonfinite_rejected(self):
        import torch
        from mapping import map_scores
        r=torch.zeros(1,14,7);r[0,4,0]=float('nan')
        with self.assertRaises(RuntimeError):map_scores(r,'tiny')
    def test_inverse_non_square_and_portrait(self):
        import torch
        from mapping import inverse_boxes
        r=torch.tensor([[50.,62.,150.,112.,.7,0.]])
        torch.testing.assert_close(inverse_boxes(r,(384,640),(720,1280))[:,:4],torch.tensor([[100.,100.,300.,200.]]),rtol=0,atol=0)
        r=torch.tensor([[62.,50.,112.,150.,.7,0.]])
        torch.testing.assert_close(inverse_boxes(r,(640,384),(1280,720))[:,:4],torch.tensor([[100.,100.,200.,300.]]),rtol=0,atol=0)
    def fixture(self,out,ignore=False):
        gt=dict(info={},images=[dict(id=i,width=200,height=200,file_name=str(i)+'.jpg') for i in (0,1)],
                categories=[dict(id=1,name='person')],annotations=[dict(id=1,image_id=0,category_id=1,bbox=[10,10,10,10],area=100,iscrowd=0,ignore=False)])
        if ignore:gt['annotations'].append(dict(id=2,image_id=0,category_id=1,bbox=[50,50,100,100],area=10000,iscrowd=0,ignore=True,uncertain=True))
        write(out/'gt.json',gt)
        return [dict(image_id=0,category_id=1,bbox=[10,10,10,10],score=.8)]
    def test_perfect_and_empty_both_evaluators(self):
        from scoring import evaluate
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            out=Path(d);pred=self.fixture(out)
            for target in ('dior','tiny'):
                a=evaluate(out/'gt.json',pred,target,out)
                self.assertAlmostEqual(a['AP_percent'],100.,places=8)
                self.assertEqual(evaluate(out/'gt.json',[],target,out)['AP_percent'],0.)
                if target=='tiny':self.assertAlmostEqual(a['native_tiny']['AP50_tiny_percent'],100.,places=8)
    def test_native_ignore_region_multiple_detections_not_false_positives(self):
        from scoring import evaluate
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            out=Path(d);pred=self.fixture(out,ignore=True)
            pred=[dict(image_id=0,category_id=1,bbox=[60,60,10,10],score=.99),
                  dict(image_id=0,category_id=1,bbox=[90,90,10,10],score=.98)]+pred
            native=evaluate(out/'gt.json',pred,'tiny',out)
            vanilla=evaluate(out/'gt.json',pred,'dior',out)
            self.assertAlmostEqual(native['AP_percent'],100.,places=8);self.assertLess(vanilla['AP_percent'],60.)
    def test_background_images_are_not_discarded(self):
        from scoring import evaluate
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            out=Path(d);pred=self.fixture(out)
            pred.insert(0,dict(image_id=1,category_id=1,bbox=[10,10,10,10],score=.99))
            for t in ('dior','tiny'):self.assertAlmostEqual(evaluate(out/'gt.json',pred,t,out)['AP_percent'],50.,places=7)
    def test_author_and_coco_equal_when_no_ignored_objects(self):
        from scoring import evaluate
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            out=Path(d);pred=self.fixture(out)
            pred.insert(0,dict(image_id=1,category_id=1,bbox=[30,30,11,11],score=.91))
            a=evaluate(out/'gt.json',pred,'tiny',out);b=evaluate(out/'gt.json',pred,'dior',out)
            for f in ('AP_percent','AP50_percent','AP75_percent','AP_small_percent'):self.assertAlmostEqual(a[f],b[f],places=10)
    def test_gzip_lossless_exact_floats(self):
        from storage import PredictionWriter,read_predictions
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            out=Path(d);rows=[dict(image_id=0,bbox=[.12345678901234567,1.,2.000000000000003,4.],score=.0012345678901234)]
            w=PredictionWriter(out/'predictions.json.gz',out);w.add(rows);w.close()
            self.assertEqual(read_predictions(out/'predictions.json.gz'),rows)
    def test_reserve_guard(self):
        from storage import disk_guard
        with patch('storage.shutil.disk_usage',return_value=type('Disk',(),dict(free=1024))()):
            with self.assertRaisesRegex(RuntimeError,'1GiB'):disk_guard(HERE)
    def test_initial_disk_guard(self):
        from controller import disk_admission
        with patch('controller.shutil.disk_usage',return_value=type('Disk',(),dict(free=4*1024**3))()):
            with self.assertRaisesRegex(RuntimeError,'4.5GiB'):disk_admission(HERE)
    def test_BN_mutation_detected(self):
        import torch
        from modeling import fingerprint
        m=torch.nn.BatchNorm2d(4).eval();before=fingerprint(m);m.running_mean.add_(1);self.assertNotEqual(before,fingerprint(m))
    def test_busy_gpu_admission(self):
        from admission import check_idle
        with patch('admission.capture',side_effect=['32760,20000\n','1234\n']):
            with self.assertRaisesRegex(RuntimeError,'GPU is in use'):check_idle()
    def test_partial_failure_never_shuts_down(self):
        from shutdown_after import run_watch
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            project=Path(d);report=project/'comparison_reports/VISDRONE_TO_DIOR_TINY_S123_SOURCE_ONLY_fixture';report.mkdir(parents=True)
            write(report/'state.json',dict(status='failed',auto_shutdown=True));runner=Mock()
            self.assertEqual(run_watch(report,project,platform='linux',runner=runner)['status'],'not_authorized_or_not_complete');runner.assert_not_called()
    def test_negative_seed_gain_is_retained(self):
        from summary import summarize
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            out=Path(d)
            for k in KEYS:
                v=10+([-1,2,3][int(k[-1])-1] if '_e60_' in k else 0)
                s=dict(status='completed',**{f+'_percent':v for f in ('AP','AP50','AP75','AP_small','AP_medium','AP_large')})
                s['native_tiny']={f'AP{t}_tiny_percent':v for t in (25,50,75)};write(out/'evaluation'/k/'metrics.json',s)
            s=summarize(out)
            for t in ('dior','tiny'):self.assertEqual(s['targets'][t]['paired']['AP_percent'],[-1,2,3])

def run_tests(out):
    out.mkdir(parents=True,exist_ok=True);ReleaseTests.temp_root=out
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseTests))
    print(log.getvalue(),flush=True);(out/'unit_checks.txt').write_text(log.getvalue(),encoding='utf-8')
    proof=dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,errors=len(result.errors),failures=len(result.failures))
    write(out/'unit_checks.json',proof);require(result.wasSuccessful(),'Engineering regression failed');return proof
if __name__=='__main__':
    p=parser();p.add_argument('--report',type=Path,required=True);args=p.parse_args();write(args.report/'check_environment.json',boot(args,args.report/'settings/check'));run_tests(args.report/'checks')
