import unittest
from pcg.footprint import Footprint

RECT = [(0,0),(12,0),(12,10),(0,10)]
HOLE = [(2,2),(4,2),(4,4),(2,4)]
HOLE2 = [(7,2),(9,2),(9,4),(7,4)]


class FootprintTests(unittest.TestCase):
    def test_shapes(self):
        cases = [
            ([(0,0),(12,0),(12,5),(6,5),(6,10),(0,10)], (), 90),
            ([(0,0),(12,0),(12,10),(8,10),(8,4),(4,4),(4,10),(0,10)], (), 96),
            ([(0,0),(12,0),(12,4),(8,4),(8,10),(4,10),(4,4),(0,4)], (), 72),
            (RECT, [HOLE], 116),
        ]
        for outer, holes, area in cases:
            with self.subTest(area=area):
                f = Footprint("fixture", outer, holes)
                self.assertEqual(f.polygon().area, area)
                self.assertTrue(f.polygon().exterior.is_ccw)
                self.assertTrue(all(not r.is_ccw for r in f.polygon().interiors))
                self.assertEqual(f, Footprint.from_json(f.canonical()))

    def test_canonical_order(self):
        a = Footprint("fixture", RECT, [HOLE, HOLE2])
        b = Footprint("fixture", list(reversed(RECT[1:]+RECT[:1])), [HOLE2[::-1], HOLE[2:]+HOLE[:2]])
        self.assertEqual(a.digest(), b.digest())
        self.assertEqual(a, Footprint("fixture", RECT+[RECT[0]], [HOLE, HOLE2]))
        self.assertNotEqual(a.digest(), Footprint("other", RECT, [HOLE, HOLE2]).digest())

    def test_reject_invalid(self):
        cases = [
            ([(0,0),(8,8),(0,8),(8,0)], []),
            ([(0,0),(1,0),(2,0)], []),
            ([(0,0),(1,0),(1,0),(0,1)], []),
            ([(0,0),(1,0),(0,float("nan"))], []),
            ([(0,0),(True,0),(0,1)], []),
            (RECT, [[(11,1),(13,1),(13,2),(11,2)]]),
            (RECT, [[(0,0),(2,1),(1,2)]]),
            (RECT, [HOLE, [(4,4),(6,4),(6,6),(4,6)]]),
            (RECT, [HOLE, HOLE]),
        ]
        for outer, holes in cases:
            with self.subTest(outer=outer, holes=holes), self.assertRaises(ValueError):
                Footprint("fixture", outer, holes)

    def test_contract_metadata(self):
        for kwargs in [{"schema_version":True},{"units":"degrees"},{"frame":"world"},{"building_id":""}]:
            args = dict(building_id="fixture", exterior=RECT)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                Footprint(**args)
        with self.assertRaises(ValueError):
            Footprint.from_json('{}')

    def test_preserve_collinear_vertices(self):
        f = Footprint("fixture", [(0,0),(3,0),(12,0),(12,10),(0,10)])
        self.assertEqual(len(f.exterior), 5)
        self.assertEqual(f.polygon().area, 120)
