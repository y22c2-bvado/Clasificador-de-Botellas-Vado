import io
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from unittest.mock import patch
import torch
from PIL import Image
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from importador import importar_zip, ErrorDataset
from referencias import MemoriaVisual
from entrenamiento import extender_cabeza


def zipfotos(clases):
    b=io.BytesIO()
    with ZipFile(b,'w') as z:
        for clase,nombre,color in clases:
            foto=io.BytesIO()
            Image.new('RGB',(64,64),color=color).save(foto,format='PNG')
            z.writestr(f'Raiz/{clase}/{nombre}.png',foto.getvalue())
    return b.getvalue()

class ModeloFalso:
    def __init__(self, file):
        self.archivo=file
    def vectorizar(self, foto):
        if isinstance(foto, (Path,str)):
            with Image.open(foto) as im:
                x=im.convert('RGB').getpixel((10,10))
        else:
            x=foto.convert('RGB').getpixel((10,10))
        return torch.nn.functional.normalize(torch.tensor([[float(x[0]),float(x[1]),float(x[2])]]),dim=-1)

class Pruebas(unittest.TestCase):
    def test_nuevas_categorias_sin_configuracion(self):
        with tempfile.TemporaryDirectory() as d:
            r=importar_zip(zipfotos([('PEEK','a',(200,30,20)),('PUR','b',(20,150,70)),('OTRO','c',(5,5,5))]),Path(d))
            self.assertEqual(r['imagenes_nuevas'],3)
            self.assertEqual(sorted(r['categorias_importadas']),['OTRO','PEEK','PUR'])
            self.assertTrue((Path(d)/'PEEK').exists())
    def test_duplicados_misma_clase(self):
        with tempfile.TemporaryDirectory() as d:
            contenido=zipfotos([('PC','a',(200,10,20)),('PC','a2',(200,10,20))])
            r=importar_zip(contenido,Path(d))
            self.assertEqual(r['imagenes_nuevas'],1)
            self.assertEqual(r['duplicadas'],1)
    def test_rechaza_foto_sin_carpeta(self):
        b=io.BytesIO()
        with ZipFile(b,'w') as z: z.writestr('foto.jpg',b'fake')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ErrorDataset): importar_zip(b.getvalue(),Path(d))
    def test_rechaza_zip_slip(self):
        b=io.BytesIO()
        with ZipFile(b,'w') as z: z.writestr('../PC/foto.jpg',b'fake')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ErrorDataset): importar_zip(b.getvalue(),Path(d))
    def test_misma_imagen_2_categorias_se_rechaza(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ErrorDataset):
                importar_zip(zipfotos([('PC','a',(100,100,100)),('PUR','b',(100,100,100))]),Path(d))
    def test_ajuste_cabeza_preserva_clases(self):
        ant=torch.nn.Linear(512,6)
        with torch.no_grad(): ant.weight.fill_(0.42); ant.bias.fill_(-0.24)
        nuevo=extender_cabeza(ant,['PET','HDPE','PVC','LDPE','PP','PS'],['PET','HDPE','PVC','LDPE','PP','PS','PUR'],512,'cpu')
        self.assertEqual(nuevo.weight.shape,(7,512))
        self.assertTrue(torch.equal(nuevo.weight[:6],ant.weight))
        self.assertTrue(torch.equal(nuevo.bias[:6],ant.bias))
    def test_memoria_compara_automaticamente(self):
        with tempfile.TemporaryDirectory() as d:
            ruta=Path(d)
            base=ruta/'dummy.pth'; base.write_bytes(b'x')
            datos=ruta/'dataset'
            fotos=[]
            for i in range(6):
                fotos.append(('PUR',f'rojo_{i}',(220-i*2,15+i,10)))
                fotos.append(('PC',f'azul_{i}',(15,10+i,225-i*2)))
            importar_zip(zipfotos(fotos),datos)
            modelo=ModeloFalso(base)
            with patch('referencias.RESULTADOS',ruta/'resultados'):
                memoria=MemoriaVisual(modelo,carpeta=datos)
                self.assertEqual(memoria.indexar(),12)
                resp=memoria.sugerir(Image.new('RGB',(64,64),(220,15,10)))
                self.assertEqual(resp['categoria'],'PUR')
                self.assertTrue('similitud_coseno' in resp)

if __name__=='__main__': unittest.main()
