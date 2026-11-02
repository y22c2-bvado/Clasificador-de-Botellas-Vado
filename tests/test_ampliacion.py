import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
import torch
from PIL import Image
BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from config import CATEGORIAS_OBJETIVO
from importador import importar_zip
from ia_plasticos import IAPlasticos
from entrenamiento import extender_cabeza

class FakeModel:
    def __init__(self, *args, **kwargs):
        self.clases=['PET','HDPE','PVC','LDPE','PP','PS']
        self.archivo=BASE/'models'/'Clasificador_Botellas_Aprendido.pth'
    @staticmethod
    def abrir_imagen(foto): return foto.convert('RGB')
    def vectorizar(self, foto): return torch.tensor([[1.0,0.0]])
    def predecir_vector(self, vector): return {'material':'PVC','categoria_sugerida':'PVC','estado':'DETERMINADO'}

class FakeRefs:
    respuesta={'estado':'SIN_EJEMPLOS'}
    def __init__(self, *args, **kwargs): pass
    def sugerir_vector(self, vector): return self.respuesta
    def fotos(self): return []

class Casos(unittest.TestCase):
    def test_diez_clases_objetivo(self):
        self.assertEqual(CATEGORIAS_OBJETIVO,('PET','HDPE','PVC','LDPE','PP','PS','PC','PUR','ABS','OTRO'))
    def test_expandir_cabeza_hasta_10(self):
        anterior=torch.nn.Linear(512,6)
        m=extender_cabeza(anterior, list(CATEGORIAS_OBJETIVO[:6]),list(CATEGORIAS_OBJETIVO),512,'cpu')
        self.assertEqual(m.weight.shape,(10,512))
        self.assertTrue(torch.equal(anterior.weight,m.weight[:6]))
    def test_abs_ingresa_por_zip(self):
        from tempfile import TemporaryDirectory
        b=io.BytesIO()
        with ZipFile(b,'w') as z:
            im=io.BytesIO(); Image.new('RGB',(48,48),(230,70,60)).save(im,'PNG')
            z.writestr('Ejemplos/ABS/teclado.png',im.getvalue())
        with TemporaryDirectory() as tmp:
            r=importar_zip(b.getvalue(),Path(tmp))
            self.assertEqual(r['categorias_importadas'], ['ABS'])
            self.assertEqual(r['imagenes_nuevas'],1)
    def test_no_plastico_con_referencias_de_otro(self):
        with patch('ia_plasticos.ClasificadorPlasticos',FakeModel),patch('ia_plasticos.MemoriaVisual',FakeRefs):
            FakeRefs.respuesta={'estado':'SUGERENCIA_CON_REFERENCIAS','categoria':'OTRO','similitud_coseno':0.92}
            ia=IAPlasticos()
            r=ia.clasificar(Image.new('RGB',(32,32),'white'))
            self.assertEqual(r['resultado'], 'NO ES PLÁSTICO')
            self.assertFalse(r['es_plastico_sugerido'])
            self.assertEqual(r['origen'],'ejemplos_etiquetados')
    def test_sin_pruebas_otro_no_inventar_rechazo(self):
        with patch('ia_plasticos.ClasificadorPlasticos',FakeModel),patch('ia_plasticos.MemoriaVisual',FakeRefs):
            FakeRefs.respuesta={'estado':'EVIDENCIA_INSUFICIENTE','categoria':'OTRO'}
            ia=IAPlasticos()
            r=ia.clasificar(Image.new('RGB',(32,32),'white'))
            self.assertNotEqual(r['resultado'],'NO ES PLÁSTICO')
            self.assertFalse('OTRO' in ia.catalogo()['categorias_entrenadas_pth'])
    def test_base_ambiguo_no_forzar_pvc(self):
        class FakeAmbiguo(FakeModel):
            def predecir_vector(self,v): return {'material':'NO DETERMINADO','categoria_sugerida':'PVC','estado':'REVISAR'}
        with patch('ia_plasticos.ClasificadorPlasticos',FakeAmbiguo),patch('ia_plasticos.MemoriaVisual',FakeRefs):
            FakeRefs.respuesta={'estado':'SIN_EJEMPLOS'}
            ia=IAPlasticos()
            r=ia.clasificar(Image.new('RGB',(32,32),'white'))
            self.assertEqual(r['resultado'],'NO DETERMINADO')
            self.assertIsNone(r['categoria_sugerida'])

if __name__=='__main__': unittest.main()
