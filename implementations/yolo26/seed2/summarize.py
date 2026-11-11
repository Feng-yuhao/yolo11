"""Paired seed1/2 official results. Retain every sign; no seed selection."""
import math
from pathlib import Path
import statistics
from protocol import read, write, require, REF, VARIANTS

KEYS = ('AP50_95_percent', 'AP50_percent', 'AP75_percent', 'AR500_percent')


def completed_pair(report, split, local=False):
    result = {v:read(Path(report) / (v + '_' + split.replace('-', '_')) / 'official_metrics.json')
              for v in VARIANTS}
    for variant, value in result.items():
        require(value['status'] == 'completed' and value['seed'] == 2 and value['variant'] == variant
                and value['split'] == split and value['local_fixture'] == local, 'Wrong seed2 score')
        if not local:
            require(value['dataset']['images'] == (548 if split == 'val' else 1610), 'Incomplete evaluation')
    require(result['baseline']['dataset'] == result['e60']['dataset'] and
            result['baseline']['protocol'] == result['e60']['protocol'] and
            result['baseline']['scorer_sha256'] == result['e60']['scorer_sha256'], 'Pair protocol/data/scorer mismatch')
    return result


def summarize(report, local=False):
    report = Path(report)
    output = dict(status='completed', seeds=[2] if local else [1,2], local_fixture=local,
                  sample_standard_deviation=True, limited_n_not_significance=True, splits={})
    lines = ['# YOLO26s-P2 / E60 / seed2 serial + paired seed1/2 summary', '']
    for split in ('val', 'test-dev'):
        fresh = completed_pair(report, split, local)
        rows = {2:{v:fresh[v]['official_metrics'] for v in VARIANTS}}
        if not local:
            original = {v:read(REF / (v + '_seed1_' + split.replace('-', '_') + '.json')) for v in VARIANTS}
            for v in VARIANTS:
                require(original[v]['status'] == 'completed' and original[v]['seed'] == 1 and
                        original[v]['dataset']['annotation_dimension_sha256'] == fresh[v]['dataset']['annotation_dimension_sha256']
                        and original[v]['scorer_sha256'] == fresh[v]['scorer_sha256'], 'Seed1 reference protocol differs')
                old_pred = original[v]['prediction']
                require(old_pred['FP32'] and old_pred['one2many_NMS'] and old_pred['batch'] == 4
                        and old_pred['conf'] == .001 and old_pred['iou'] == .7 and old_pred['max_det'] == 500,
                        'Seed1 reference prediction settings differ')
            rows[1] = {v:original[v]['official_metrics'] for v in VARIANTS}
        aggregate = {}
        for key in KEYS:
            values = {v:[rows[s][v][key] for s in sorted(rows)] for v in VARIANTS}
            deltas = [rows[s]['e60'][key] - rows[s]['baseline'][key] for s in sorted(rows)]
            aggregate[key] = dict(baseline_mean=statistics.mean(values['baseline']),
                                 e60_mean=statistics.mean(values['e60']), delta_mean_pp=statistics.mean(deltas),
                                 baseline_sample_std=statistics.stdev(values['baseline']) if len(rows)>1 else None,
                                 e60_sample_std=statistics.stdev(values['e60']) if len(rows)>1 else None,
                                 paired_delta_sample_std=statistics.stdev(deltas) if len(rows)>1 else None,
                                 paired_delta_pp={str(s):rows[s]['e60'][key]-rows[s]['baseline'][key] for s in sorted(rows)})
        output['splits'][split] = dict(seed2=fresh, aggregate=aggregate)
        lines += ['## ' + split, '', '|Metric (%)|Baseline seed2|+E60 seed2|Seed2 delta pp|Seed1/2 mean delta pp|',
                  '|---|---:|---:|---:|---:|']
        for key in KEYS:
            b,e = rows[2]['baseline'][key], rows[2]['e60'][key]
            lines.append(f'|{key}|{b:.4f}|{e:.4f}|{e-b:+.4f}|{aggregate[key]["delta_mean_pp"]:+.4f}|')
        lines += ['']
        if not local:
            lines += ['|Metric (%)|Baseline mean +/- sample SD|+E60 mean +/- sample SD|Mean paired delta pp|',
                      '|---|---:|---:|---:|']
            for key in KEYS:
                a = aggregate[key]
                lines.append(f'|{key}|{a["baseline_mean"]:.4f} +/- {a["baseline_sample_std"]:.4f}|'
                             f'{a["e60_mean"]:.4f} +/- {a["e60_sample_std"]:.4f}|{a["delta_mean_pp"]:+.4f}|')
            lines += ['']
    lines += ['Both trained from original YOLO26s weights, not seed1 best fine-tuning.',
              '640/b16/w8/200 epochs; original progressive dual loss/STAL; one2many+NMS/max500.',
              'Only global training seed and independent output paths change from seed1.',
              'The fixed common model initialization and isolated E60 initialization stay identical to seed1.',
              'Validation-best checkpoint fixed before test-dev scoring. All positive and negative seeds retained.',
              'Two seeds provide repeatability evidence, not a claim of statistical significance.', '']
    write(report / 'comparison.json', output)
    (report / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')

