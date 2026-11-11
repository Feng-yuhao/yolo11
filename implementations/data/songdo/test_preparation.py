"""Tiny semantic/unit fixtures only, no original data modifications."""
import hashlib
import unittest
from convert_songdo import validation_ids, parse_label, validate_triplet


class PreparationTests(unittest.TestCase):
    def test_holdout_is_fixed_and_input_order_independent(self):
        images=[{'id':i} for i in range(1,4336)]
        chosen=validation_ids(images)
        self.assertEqual(len(chosen),434)
        self.assertEqual(chosen,validation_ids(list(reversed(images))))
        expected=set(sorted(range(1,4336),key=lambda i:hashlib.sha256(
            ('songdo-v2-holdout:20261017:'+str(i)).encode()).digest())[:434])
        self.assertEqual(chosen,expected)

    def test_background_is_not_dropped(self):
        self.assertEqual(parse_label(b'\n',3840,2160),[])

    def test_portrait_uses_its_own_dimensions(self):
        cls,box=parse_label(b'0 0.5 0.5 0.1 0.2\n',2160,3840)[0]
        self.assertEqual(cls,0)
        for actual,expected in zip(box,(972.,1536.,1188.,2304.)):
            self.assertAlmostEqual(actual,expected)

    def test_duplicate_and_invalid_labels_rejected(self):
        for data in (b'4 .5 .5 .1 .1',b'0 nan .5 .1 .1',b'0 .99 .5 .2 .2',
                     b'0 .5 .5 0 .1',b'0 .5 .5 .1 .1\n0 .5 .5 .1 .1',b'0 .5 .5 .1'):
            with self.subTest(data=data),self.assertRaises((ValueError,IndexError)):
                parse_label(data,3840,2160)

    def test_original_integer_geometry_and_class_binding(self):
        image=dict(id=9,file_name='00009.jpg',width=2160,height=3840)
        label=b'0 0.5 0.5 0.1 0.2\n'
        xml=b'<annotation><filename>00009.jpg</filename><size><width>2160</width><height>3840</height></size><object><name>car</name><bndbox><xmin>972</xmin><ymin>1536</ymin><xmax>1188</xmax><ymax>2304</ymax></bndbox></object></annotation>'
        anns=[dict(id=1,image_id=9,category_id=1,bbox=[972,1536,216,768],area=216*768,iscrowd=0)]
        rows,error=validate_triplet(image,label,xml,anns)
        self.assertEqual(len(rows),1); self.assertLess(error,1e-8)
        with self.assertRaises(ValueError): validate_triplet(image,label,xml,[dict(anns[0],category_id=2)])
        shifted=xml.replace(b'<xmin>972',b'<xmin>973').replace(b'<xmax>1188',b'<xmax>1189')
        with self.assertRaises(ValueError):
            validate_triplet(image,label,shifted,[dict(anns[0],bbox=[973,1536,216,768])])


if __name__=='__main__': unittest.main()
