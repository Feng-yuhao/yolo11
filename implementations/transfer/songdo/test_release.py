"""Independent small-fixture mapping/NMS/COCO regression tests; no target AP selection."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from common import HERE, CLASSES, KEYS, boot, parser, require, write


class ReleaseTests(unittest.TestCase):
    temp_root = HERE
    def fixture(self,n=7):
        import torch
        raw = torch.zeros(1,14,n,dtype=torch.float32)
        raw[:,0] = torch.arange(n)*30+20
        raw[:,1] = 20; raw[:,2:4] = 10
        return raw

    def test_merge_all_four_score_columns(self):
        import torch
        from mapping import map_scores
        raw = self.fixture()
        raw[:,7] = .2; raw[:,8] = .8; raw[:,12] = .6; raw[:,9] = .4; raw[:,13] = .3
        actual = map_scores(raw)
        torch.testing.assert_close(actual[:,4:,0],torch.tensor([[.8,.6,.4,.3]]),rtol=0,atol=0)
        torch.testing.assert_close(actual[:,:4],raw[:,:4],rtol=0,atol=0)

    def test_no_alias_or_raw_mutation(self):
        import torch
        from mapping import map_scores,mapped_nms
        raw = self.fixture(); raw[:,7] = .8; saved = raw.clone()
        mapped = map_scores(raw); mapped.zero_()
        torch.testing.assert_close(raw,saved,rtol=0,atol=0)
        mapped_nms(raw)
        torch.testing.assert_close(raw,saved,rtol=0,atol=0)

    def test_merged_car_van_are_deduplicated(self):
        from mapping import mapped_nms
        raw = self.fixture(); raw[:,:,1] = raw[:,:,0]
        raw[0,7,0] = .9; raw[0,8,1] = .8
        result = mapped_nms(raw)[0]
        self.assertEqual(len(result),1); self.assertEqual(int(result[0,5]),0)

    def test_unannotated_high_scores_do_not_hide_mapped_score(self):
        from mapping import mapped_nms
        raw = self.fixture(); raw[:,4] = .999; raw[0,12,0] = .75
        result = mapped_nms(raw)[0]
        self.assertEqual(len(result),1); self.assertEqual(int(result[0,5]),1)

    def test_unmapped_classes_filtered_before_top500(self):
        from mapping import mapped_nms
        raw = self.fixture(620); raw[0,4,:610] = .99; raw[0,7,619] = .2
        result = mapped_nms(raw)[0]
        self.assertEqual(len(result),1); self.assertEqual(int(result[0,5]),0)

    def test_class_aware_nms_keeps_overlapping_bus(self):
        from mapping import mapped_nms
        raw = self.fixture(); raw[:,:,1] = raw[:,:,0]
        raw[0,7,0] = .9; raw[0,12,1] = .8
        self.assertEqual(set(mapped_nms(raw)[0][:,5].tolist()),{0.,1.})

    def test_independent_numpy_greedy_nms(self):
        import numpy as np
        from mapping import mapped_nms
        raw = self.fixture(9)
        raw[:,:,1] = raw[:,:,0]; raw[:,:,3] = raw[:,:,2]
        for i,(cls,score) in enumerate([(3,.91),(4,.81),(8,.71),(8,.61),(5,.51),(9,.41),(0,.99),(3,.21),(5,.11)]):
            raw[0,4+cls,i] = score
        r = raw[0].numpy(); candidates = []
        for i in range(r.shape[1]):
            scores = [max(float(r[7,i]),float(r[8,i])),float(r[12,i]),float(r[9,i]),float(r[13,i])]
            cls = int(np.argmax(scores)); score = scores[cls]
            if score > .001:
                x,y,w,h = r[:4,i]; candidates.append([x-w/2,y-h/2,x+w/2,y+h/2,score,cls])
        remaining = sorted(candidates,key=lambda v:-v[4]); kept = []
        while remaining:
            first = remaining.pop(0); kept.append(first); filtered = []
            for row in remaining:
                w = max(0,min(first[2],row[2])-max(first[0],row[0]))
                h = max(0,min(first[3],row[3])-max(first[1],row[1]))
                inter = w*h
                union = (first[2]-first[0])*(first[3]-first[1])+(row[2]-row[0])*(row[3]-row[1])-inter
                if row[5] != first[5] or inter/union <= .7:
                    filtered.append(row)
            remaining = filtered
        np.testing.assert_allclose(mapped_nms(raw)[0].numpy(),np.array(kept),atol=1e-6,rtol=0)

    def test_bad_nc_and_nonfinite_rejected(self):
        import torch
        from mapping import map_scores
        with self.assertRaises(RuntimeError): map_scores(torch.zeros(1,8,7))
        raw = self.fixture(); raw[0,7,0] = float('nan')
        with self.assertRaises(RuntimeError): map_scores(raw)

    def test_inverse_landscape_and_portrait(self):
        import torch
        from mapping import inverse_boxes
        for original in ((2160,3840),(3840,2160)):
            oh,ow = original; gain = 640/max(original)
            px = round((640-ow*gain)/2-.1); py = round((640-oh*gain)/2-.1)
            row = torch.tensor([[100*gain+px,200*gain+py,300*gain+px,400*gain+py,.9,0]])
            expected = torch.tensor([[100.,200.,300.,400.,.9,0.]])
            torch.testing.assert_close(inverse_boxes(row,(640,640),original),expected,atol=.001,rtol=0)

    def test_convert_original_id_and_category(self):
        from coco_eval import convert
        item = dict(id=123456,width=3840,height=2160)
        row = convert(item,[[100,200,120,230,.5,3]])[0]
        self.assertEqual(row,dict(image_id=123456,category_id=4,bbox=[100,200,20,30],score=.5))
        with self.assertRaises(RuntimeError): convert(item,[[0,0,4000,20,.5,0]])

    def test_coco_perfect_wrong_class_empty_and_subset(self):
        from coco_eval import coco_score
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp)
            categories = [dict(id=i+1,name=n) for i,n in enumerate(CLASSES)]
            images = [dict(id=i+100,width=100,height=100,file_name=str(i)+'.jpg') for i in range(4)]
            annotations = [dict(id=i+1,image_id=i+100,category_id=i+1,bbox=[10,10,20,20],area=400,iscrowd=0)
                           for i in range(4)]
            write(root/'gt.json',dict(info={},images=images,categories=categories,annotations=annotations))
            perfect = [dict(image_id=i+100,category_id=i+1,bbox=[10,10,20,20],score=.9) for i in range(4)]
            score = coco_score(root/'gt.json',perfect,root/'perfect')
            self.assertAlmostEqual(score['AP_percent'],100.)
            self.assertEqual(score['maxDets'],[1,10,100])
            wrong = [dict(p,category_id=p['category_id']%4+1) for p in perfect]
            self.assertEqual(coco_score(root/'gt.json',wrong,root/'wrong')['AP_percent'],0.)
            self.assertEqual(coco_score(root/'gt.json',[],root/'empty')['AP_percent'],0.)
            self.assertAlmostEqual(coco_score(root/'gt.json',perfect[:1],root/'subset',[100])['AP_percent'],100.)

    def test_summary_keeps_negative_paired_seed(self):
        from controller import summarize
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp)
            for key in KEYS:
                seed = int(key[-1]); val = 10. + ([-1.,2.,3.][seed-1] if key.startswith('e60') else 0.)
                metric = {name+'_percent':val for name in ('AP','AP50','AP75','AP_small','AP_medium','AP_large')}
                write(root/'evaluation'/key/'coco_metrics.json',dict(metric,status='completed',images=1084))
            summary = summarize(root)
            self.assertEqual(summary['paired']['AP_percent'],[-1.,2.,3.])
            self.assertAlmostEqual(summary['aggregates']['AP_percent']['paired_gain_mean_pp'],4/3)

    def test_busy_gpu_blocks_without_process_action(self):
        from unittest.mock import patch
        from admission import check_idle
        with patch('admission.capture',side_effect=['32760, 20000\n','1234\n']):
            with self.assertRaisesRegex(RuntimeError,'GPU is in use'):
                check_idle()

    def test_other_python_and_kernel_block_but_readonly_services_do_not(self):
        import os
        from unittest.mock import patch
        from admission import check_idle
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp); process = root/str(os.getpid()+100000)
            process.mkdir(); (process/'stat').write_text('1234 (python) S '+'0 '*25)
            command = process/'cmdline'
            for argv in (['python','-u','train.py'],['python','-m','ipykernel_launcher']):
                command.write_bytes(b'\0'.join(v.encode() for v in argv))
                with patch('admission.capture',side_effect=['32760, 32000\n','']):
                    with self.assertRaisesRegex(RuntimeError,'Other Python'):
                        check_idle(root)
            command.write_bytes(b'python\0-m\0jupyterlab\0')
            with patch('admission.capture',side_effect=['32760, 32000\n','']):
                self.assertEqual(check_idle(root)['other_GPU_PIDs'],[])

    def test_wrong_source_weight_bytes_fail_closed(self):
        from unittest.mock import patch
        from common import read,weights_audit
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            root = Path(tmp); spec = read(HERE/'references/checkpoints.json')['e1_seed1']
            path = root/spec['relative_path']; path.parent.mkdir(parents=True)
            path.write_bytes(b'not a trained VisDrone checkpoint')
            with self.assertRaisesRegex(RuntimeError,'Wrong/modified source checkpoint'):
                weights_audit(root)

    def test_model_fingerprint_detects_BN_update(self):
        import torch
        from modeling import fingerprint
        model = torch.nn.BatchNorm2d(4).eval()
        before = fingerprint(model)
        with torch.no_grad():
            model.running_mean.add_(1)
        self.assertNotEqual(fingerprint(model),before)


def run_tests(out):
    out.mkdir(parents=True,exist_ok=True)
    ReleaseTests.temp_root = out
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReleaseTests)
    with io.StringIO() as captured:
        result = unittest.TextTestRunner(stream=captured,verbosity=2).run(suite)
        text = captured.getvalue()
    print(text,flush=True)
    proof = dict(status='passed' if result.wasSuccessful() else 'failed',tests=result.testsRun,
                 errors=len(result.errors),failures=len(result.failures))
    write(out/'unit_checks.json',proof)
    (out/'unit_checks.txt').write_text(text,encoding='utf-8')
    require(result.wasSuccessful(),'Mapping/NMS/COCO engineering checks failed')
    return proof


if __name__ == '__main__':
    p = parser(); p.add_argument('--report',type=Path,required=True); args = p.parse_args()
    environment = boot(args,args.report/'settings/check')
    write(args.report/'check_environment.json',environment)
    run_tests(args.report)
