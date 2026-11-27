import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
import numpy as np
from reference_intake33 import Part
from reference_low33 import indices_by_material,material
class Low33Tests(unittest.TestCase):
    def test_material_index_filter(self):
        p=Part(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2],[0,0,0]]),None,2,'a')
        q=Part(p.xyz,np.array([[0,1,2]]),None,4,'b')
        r=indices_by_material([p,q])
        np.testing.assert_array_equal(r[2],[0]);np.testing.assert_array_equal(r[4],[1])
    def test_unbound_maps(self):
        class G:doc={'materials':[{'pbrMetallicRoughness':{'roughnessFactor':.5}}]}
        r=material(G(),0)
        self.assertEqual(r['bindings'],dict(basecolor=False,normal=False,metallic_roughness=False))
        self.assertEqual(int(r['maps'][2][0,0,1]),127)

    def test_basecolor_normal_alias(self):
        from unittest.mock import patch
        class G:
            doc={'textures':[{'source':0}],'materials':[{'pbrMetallicRoughness':{'baseColorTexture':{'index':0}},'normalTexture':{'index':0}}]}
        with patch('reference_low33.image',return_value=np.full((4,4,4),80,np.uint8)):
            r=material(G(),0)
        self.assertTrue(r['normal_alias']);self.assertFalse(r['normal_used'])
        np.testing.assert_array_equal(r['maps'][1][0,0],[128,128,255])
