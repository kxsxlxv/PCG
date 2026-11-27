import unittest
from pcg.quality_budget import ceiling,entry_budget,equipment_ceiling
class QualityBudgetTests(unittest.TestCase):
    def test_entry_budget(self):
        self.assertEqual(entry_budget(100,0),0)
        self.assertEqual(entry_budget(100,2),32)
        self.assertEqual(entry_budget(100,20),96)
        self.assertEqual(entry_budget(1790,20),10)
    def test_equipment_ceiling(self):
        self.assertEqual(equipment_ceiling(500),644)
        self.assertEqual(equipment_ceiling(1750),1800)
    def test_invalid_inputs(self):
        for args in ((-1,4),(1801,4),(0,-1),(1.2,4),(1,False)):
            with self.assertRaises(ValueError):ceiling(*args)
        with self.assertRaises(ValueError):entry_budget(0,-1)
