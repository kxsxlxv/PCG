import json
import unittest
from pcg.contracts import Rectangle

class RectangleTests(unittest.TestCase):
    def fixture(self):
        return Rectangle("house-1", 12, 8, 6, "gable", 2)

    def test_roundtrip(self):
        a = self.fixture()
        b = Rectangle.from_json(a.canonical())
        self.assertEqual(a, b)
        self.assertEqual(a.digest(), b.digest())
        self.assertEqual(a.digest(), Rectangle("house-1", 12.0, 8.0, 6.0, "gable", 2.0).digest())

    def test_rejections(self):
        for key, value in [("width_m", 0), ("depth_m", -1), ("eave_m", float("inf")),
                           ("width_m", True), ("roof_rise_m", float("nan")),
                           ("roof", "hip"), ("roof_rise_m", 0), ("schema_version", 2),
                           ("schema_version", True), ("units", "ft"), ("frame", "unknown"),
                           ("building_id", "")]:
            with self.subTest(key=key, value=value):
                data = json.loads(self.fixture().canonical())
                data[key] = value
                with self.assertRaises(ValueError):
                    Rectangle.from_json(json.dumps(data))

    def test_required_fields(self):
        data = json.loads(self.fixture().canonical())
        del data["frame"]
        with self.assertRaises(ValueError):
            Rectangle.from_json(json.dumps(data))

    def test_flat(self):
        self.assertEqual(Rectangle("flat", 12, 8, 6, "flat", 0).roof_rise_m, 0)
        with self.assertRaises(ValueError):
            Rectangle("flat", 12, 8, 6, "flat", 1)
