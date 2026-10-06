"""Independent train-reader regression fixtures; no real-data claims."""
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import d27_official as official
from train_annotations import parse_training_annotation, load_training_dataset


def run_checks():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'annotations').mkdir()
        (root/'images').mkdir()
        path = root/'annotations'/'fixture.txt'
        # Fixtures are test data, not modifications to original annotations.
        Image.new('RGB', (1100,500)).save(root/'images'/'fixture.png')
        normal = '10,20,12,14,1,1,0,0\n20,30,15,15,0,2,0,0\n100,100,4,4,0,0,0,0\n'
        path.write_text(normal, encoding='utf-8')
        raw, audit = parse_training_annotation(path)
        np.testing.assert_array_equal(raw, official.parse_annotation(path))
        assert audit == []
        records, summary = load_training_dataset(root, 1)
        old = official.EXPECTED_IMAGES
        try:
            official.EXPECTED_IMAGES = 1
            original_records, original_summary = official.load_dataset(root)
        finally:
            official.EXPECTED_IMAGES = old
        for key, value in original_summary.items():
            assert summary[key] == value, key
        for field in ('raw_gt','eval_gt','ignore_rects','others_rects'):
            np.testing.assert_array_equal(getattr(records[0],field),getattr(original_records[0],field))

        row = '1008,374,3,0,0,0,0,0'
        path.write_text(normal+row+'\n100,100,0,2,0,0,0,0\n200,200,0,0,0,0,0,0\n', encoding='utf-8')
        raw, audit = parse_training_annotation(path)
        assert len(audit) == 3 and audit[0]['line'] == 4
        np.testing.assert_array_equal(raw[3], [1008,374,3,0,0,0,0,0])
        records, summary = load_training_dataset(root, 1)
        assert summary['zero_extent_class0_count'] == 3
        assert len(records[0].ignore_rects) == 4
        assert np.all(records[0].eval_gt[:,2:4] > 0)
        # Analytic raster strip: columns1009..1011, row375 are covered.
        assert official.official_region_coverage([1008,374,3,1],records[0].ignore_rects,1100,500) == 1
        assert official.official_region_coverage([1008,380,3,1],records[0].ignore_rects,1100,500) == 0
        try:
            official.parse_annotation(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Frozen validation parser was relaxed')

        bad_rows = ['10,20,3,0,1,1,0,0','10,20,-3,1,0,0,0,0',
                    '10,20,3,1,1,1.5,0,0','10,20,3,1,1,12,0,0',
                    '10,20,nan,1,0,0,0,0','10,20,3']
        for bad in bad_rows:
            path.write_text(bad,encoding='utf-8')
            try:
                parse_training_annotation(path)
            except ValueError:
                pass
            else:
                raise AssertionError('Accepted invalid annotation: '+bad)
        path.write_text('\ufeff10,20,3,4,1,1,0,0,\n\n',encoding='utf-8')
        assert parse_training_annotation(path)[0].shape == (1,8)
    return dict(normal_reader_equivalence=True, zero_extent_class0_preserved=True,
                official_raster_strip=True, invalid_rows_rejected=True,
                frozen_val_parser_unchanged=True)


if __name__ == '__main__':
    print(run_checks())
