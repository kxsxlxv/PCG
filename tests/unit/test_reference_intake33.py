import base64,json,os,struct,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lab'))
from reference_intake33 import Gltf,triangulate,measure,extract,material_inventory,Part,run

def fixture(path,strip=False):
    pos=np.array([[0,0,0],[2,0,0],[0,3,0],[2,3,0]],dtype='<f4')
    uv=np.array([[0,0],[1,0],[0,1],[1,1]],dtype='<f4')
    idx=np.array([0,1,2,3] if strip else [0,1,2,1,3,2],dtype='<u2')
    buf=pos.tobytes()+uv.tobytes()+idx.tobytes()
    doc={'asset':{'version':'2.0'},'buffers':[{'byteLength':len(buf),'uri':'data:application/octet-stream;base64,'+base64.b64encode(buf).decode()}],
      'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':48},{'buffer':0,'byteOffset':48,'byteLength':32},{'buffer':0,'byteOffset':80,'byteLength':len(idx)*2}],
      'accessors':[{'bufferView':0,'componentType':5126,'count':4,'type':'VEC3'},{'bufferView':1,'componentType':5126,'count':4,'type':'VEC2'},{'bufferView':2,'componentType':5123,'count':len(idx),'type':'SCALAR'}],
      'meshes':[{'primitives':[{'attributes':{'POSITION':0,'TEXCOORD_0':1},'indices':2,'mode':5 if strip else 4,'material':0}]}],
      'nodes':[{'translation':[10,20,30],'children':[1]},{'name':'synthetic','mesh':0,'scale':[2,1,1]}],
      'scenes':[{'nodes':[0]}],'scene':0,'materials':[{'name':'synthetic'}]}
    path.write_text(json.dumps(doc));return path

class IntakeTests(unittest.TestCase):
    def test_strip_and_fan(self):
        np.testing.assert_array_equal(triangulate(np.arange(5),5),[[0,1,2],[2,1,3],[2,3,4]])
        np.testing.assert_array_equal(triangulate(np.arange(4),6),[[0,1,2],[0,2,3]])
        with self.assertRaises(ValueError):triangulate(np.arange(4),1)
    def test_transform_uv(self):
        with tempfile.TemporaryDirectory() as d:
            g=Gltf(fixture(Path(d)/'a.gltf',True));groups=list(g.groups())
            self.assertEqual(len(groups),1);p=groups[0][1][0]
            np.testing.assert_allclose(p.xyz[0],[10,-30,20])
            np.testing.assert_allclose(p.xyz[3],[14,-30,23])
            r,_,_=measure([p]);self.assertEqual(r['triangles_nondegenerate'],2)
            self.assertEqual(r['uv0_nonzero_triangles'],2);self.assertEqual(r['vertical_area_fraction'],1.)
            self.assertEqual(material_inventory(g)[0]['bindings'],{})
    def test_archive_path(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);z=root/'x.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../escape','bad')
            with self.assertRaises(ValueError):extract(z,root/'out')
    def test_actual_inventory(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=fixture(root/'a.gltf');q=fixture(root/'b.gltf',True)
            z=root/'x.zip'
            with zipfile.ZipFile(z,'w') as f:f.write(p,'a.gltf')
            with patch.dict(os.environ,{'PCG_PRIVATE_LOCAL':'1'}):
                r=run(z,q,root/'results')
            self.assertEqual(len(r['objects']),2)
            self.assertEqual(sum(x['triangles_nondegenerate'] for x in r['objects']),4)
            self.assertTrue((root/'results/catalog.json').is_file())

class AtlasTests(unittest.TestCase):
    def test_candidate_threshold(self):
        from reference_atlas33 import candidates
        gray=np.full((128,128),170,np.uint8)
        gray[8:48,8:48]=0;gray[64:112,64:112]=12;gray[0:8,64:72]=0
        r=candidates(gray)
        self.assertEqual(r['components'],3);self.assertEqual(r['rectangular_candidates'],2)
        self.assertEqual(r['dark_fraction'],(25+36+1)/256)
