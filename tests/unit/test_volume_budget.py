import unittest
from pcg.volume_budget import Offer, select, detail_reserve
class BudgetTests(unittest.TestCase):
    def test_priority_and_atomicity(self):
        offers=[Offer('stack',90,1,1),Offer('core',24,2,2),Offer('bay',10,3,3)]
        self.assertEqual(select(offers,110),(('bay','core'),34))
        self.assertEqual(select(reversed(offers),110),select(offers,110))
        self.assertEqual(select(offers,124),(('bay','core','stack'),124))
    def test_invalid(self):
        for offers,cap in (([Offer('x',0,1,0)],10),([Offer('x',1,1,0)]*2,10),([], -1)):
            with self.assertRaises(ValueError):select(offers,cap)
    def test_reserve(self):
        self.assertGreater(detail_reserve([range(8)],4,'flat'),detail_reserve([range(4)],0,'flat'))
