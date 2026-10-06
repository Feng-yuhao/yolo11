"""Independent synthetic CPU checks, not server CUDA/official reproduction."""
import json
import math
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from readout_stats import ReadoutStats, js


def record(width=517, height=333, gt=None, image_id=1):
    if gt is None:
        gt = [[100., 80., 20., 16., 0., 1.], [0., 0., 10., 10., 1., 2.]]
    return SimpleNamespace(image_id=image_id, width=width, height=height,
                           eval_gt=np.asarray(gt, dtype=np.float64).reshape(-1, 6))


def captured(x, weights=None, old=(2., 3.), new=(1., -1.)):
    if weights is None:
        weights = torch.full((len(x), 2, 4, *x.shape[-2:]), .25)
    decomposition = (x[:, :48], x[:, :48], weights, [x[:, :48]] * 4)
    return decomposition, tuple(x + x * v for v in old), tuple(x * v for v in new)


class StatsTests(unittest.TestCase):
    def test_JS_depth_axis(self):
        a = torch.tensor([[1., .25], [0., .25], [0., .25], [0., .25]])
        b = torch.tensor([[0., .25], [1., .25], [0., .25], [0., .25]])
        torch.testing.assert_close(js(a, b), torch.tensor([math.log(2), 0.]), rtol=1e-6, atol=1e-7)

    def test_rectangular_projection_and_signed_cosine(self):
        # 517x333 -> rounded resize 640x412 -> actual rect input 640x416.
        rec = record()
        expected_x = int((110 * 640 / 517) // 4)
        expected_y = int((88 * 640 / 517 + 2) // 4)
        x = torch.ones(1, 128, 104, 160)
        weights = torch.full((1, 2, 4, 104, 160), .25)
        weights[0, 0, :, expected_y, expected_x] = torch.tensor([1., 0., 0., 0.])
        stats = ReadoutStats(); stats.set_context([rec], (416, 640))
        stats(0, None, x, *captured(x, weights))
        self.assertEqual(len(stats.rows), 1)  # ignored official GT never sampled
        row = stats.rows[0]
        self.assertEqual(row['feature_cell'], [expected_x, expected_y])
        self.assertEqual(row['cls_depth_weights'], [1., 0., 0., 0.])
        self.assertAlmostEqual(row['cls_depth_weights_3x3'][0], (1 + 8 * .25) / 9, places=6)
        self.assertAlmostEqual(sum(row['cls_depth_weights_3x3']), 1., places=6)
        self.assertAlmostEqual(row['input_RMS'], 1., places=6)
        self.assertAlmostEqual(row['cls_e40_residual_RMS'], 2., places=6)
        self.assertAlmostEqual(row['reg_e40_residual_RMS'], 3., places=6)
        self.assertAlmostEqual(row['cls_new_to_e40_ratio'], .5, places=6)
        self.assertAlmostEqual(row['cls_residual_cosine'], 1., places=6)
        self.assertAlmostEqual(row['reg_residual_cosine'], -1., places=6)
        self.assertAlmostEqual(row['size'], math.sqrt(320) * 640 / 517, places=6)
        # Description includes undefined empty groups as null, never fabricated 0.
        summary = stats.summary()
        self.assertEqual(summary['P3/all']['gt_count'], 0)
        self.assertIsNone(summary['P3/all']['input_RMS']['mean'])
        json.dumps(summary, allow_nan=False)

    def test_rounded_letterbox_odd_padding(self):
        # Height rounds 315.284... to 315; 384-315=69 => top=34.
        rec = record(width=517, height=255, gt=[[100, 80, 20, 16, 0, 1]])
        x = torch.ones(1, 128, 96, 160)
        stats = ReadoutStats(); stats.set_context([rec], (384, 640))
        stats(0, None, x, *captured(x))
        self.assertAlmostEqual(stats.rows[0]['input_center_xy'][1], 88 * 640 / 517 + 34)

    def test_edge_clamp_and_cropped_neighborhood(self):
        rec = record(width=32, height=32, gt=[[-6, -6, 4, 4, 0, 1], [32, 32, 4, 4, 0, 2]])
        x = torch.ones(1, 128, 8, 8)
        stats = ReadoutStats(); stats.set_context([rec], (32, 32))
        stats(0, None, x, *captured(x))
        self.assertEqual([r['feature_cell'] for r in stats.rows], [[0, 0], [7, 7]])
        for row in stats.rows:
            self.assertEqual(row['input_RMS'], 1.)  # border has no padded-zero dilution
            self.assertEqual(row['cls_new_readout_RMS'], 1.)

    def test_zero_residual_is_measured_but_cosine_is_undefined(self):
        x = torch.ones(1, 128, 8, 8)
        stats = ReadoutStats(); stats.set_context([record(width=32, height=32, gt=[[0, 0, 8, 8, 0, 1]])], (32, 32))
        stats(0, None, x, *captured(x, old=(0., 0.), new=(0., 0.)))
        row = stats.rows[0]
        self.assertEqual(row['cls_new_to_input_ratio'], 0.)
        self.assertIsNone(row['cls_new_to_e40_ratio'])
        self.assertIsNone(row['cls_residual_cosine'])
        json.dumps(stats.summary(), allow_nan=False)

    def test_geometry_join_and_outcome_groups(self):
        x = torch.ones(1, 128, 8, 8)
        rec = record(width=32, height=32, gt=[[0, 0, 8, 8, 0, 1], [8, 8, 8, 8, 0, 2]])
        stats = ReadoutStats(); stats.set_context([rec], (32, 32)); stats(0, None, x, *captured(x))
        geom = [dict(image_id=1, gt_index=j, cls=j + 1, size=160., iou={'top500': value})
                for j, value in enumerate((.75, .74))]
        summary = stats.summary(geom)
        self.assertEqual(summary['P2/all/normal_final_correct75']['gt_count'], 1)
        self.assertEqual(summary['P2/all/normal_final_missing75']['gt_count'], 1)
        self.assertIsNone(summary['P3/all/normal_final_correct75']['input_RMS']['mean'])
        flat = [dict(image_id=r['image_id'], gt_index=r['gt_index'], cls=r['cls'],
                     size=r['size'], final_correct=r['iou']['top500']) for r in geom]
        self.assertEqual(stats.summary(flat), summary)
        del flat[0]['final_correct']
        with self.assertRaisesRegex(RuntimeError, 'Missing finite'):
            stats.summary(flat)
        with self.assertRaisesRegex(RuntimeError, 'absent'):
            stats.summary(geom[:1])
        geom[0]['cls'] = 3
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            stats.summary(geom)

    def test_reject_bad_context_no_capture_and_duplicate_forward(self):
        x = torch.ones(1, 128, 8, 8)
        rec = record(width=32, height=32, gt=[[0, 0, 8, 8, 0, 1]])
        stats = ReadoutStats()
        with self.assertRaisesRegex(RuntimeError, 'context'):
            stats(0, None, x, *captured(x))
        with self.assertRaisesRegex(RuntimeError, 'aligned'):
            stats.set_context([rec], (31, 32))
        stats.set_context([rec], (32, 32))
        decomposition, outputs, _ = captured(x)
        with self.assertRaisesRegex(RuntimeError, 'invent'):
            stats(0, None, x, decomposition, outputs, None)
        stats(0, None, x, *captured(x))
        with self.assertRaisesRegex(RuntimeError, 'Duplicate'):
            stats(0, None, x, *captured(x))
        mismatch = ReadoutStats(); mismatch.set_context([rec], (64, 64))
        with self.assertRaisesRegex(RuntimeError, 'stride'):
            mismatch(0, None, x, *captured(x))


if __name__ == '__main__':
    unittest.main(verbosity=2)
