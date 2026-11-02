"""Interfaz para usar CLIP+memoria en CUALQUIER API Python ya existente.
No crea servidores. No requiere ngrok, cámaras ni SQLite.

    ia=IAPlasticos()
    ia.clasificar(imagen_en_bytes) -> dict listo para JSON
    ia.aprender_zip(zip_en_bytes) -> dict
    ia.reentrenar() -> dict     # proceso lento, ejecutar en tarea administrada
    ia.recargar() -> None
"""
from pathlib import Path
import io
import subprocess
import sys
import threading
from PIL import Image, ImageOps
try:
    from .config import DATOS, MODELO_ORIGINAL, MODELO_MEJORADO, CATEGORIAS_OBJETIVO, CATEGORIA_RECHAZO, MENSAJE_NO_PLASTICO, MENSAJE_INCIERTO, ruta_modelo
    from .importador import importar_zip
    from .modelo import ClasificadorPlasticos
    from .referencias import MemoriaVisual
except ImportError:
    from config import DATOS, MODELO_ORIGINAL, MODELO_MEJORADO, CATEGORIAS_OBJETIVO, CATEGORIA_RECHAZO, MENSAJE_NO_PLASTICO, MENSAJE_INCIERTO, ruta_modelo
    from importador import importar_zip
    from modelo import ClasificadorPlasticos
    from referencias import MemoriaVisual

class IAPlasticos:
    def __init__(self, modelo=None, dispositivo=None, datos=DATOS):
        self._archivo = Path(modelo) if modelo else None
        self._dispositivo = dispositivo
        self._datos = Path(datos)
        self._lock = threading.RLock()
        self._entrenando = False
        self.recargar()

    def recargar(self):
        with self._lock:
            ruta=self._archivo if self._archivo else ruta_modelo()
            self.modelo=ClasificadorPlasticos(ruta, dispositivo=self._dispositivo)
            self.memoria=MemoriaVisual(self.modelo,carpeta=self._datos)

    @staticmethod
    def _imagen(imagen):
        if isinstance(imagen, (bytes,bytearray)):
            if len(imagen)>10*1024*1024:
                raise ValueError('Imagen superior a 10 MB')
            with Image.open(io.BytesIO(imagen)) as im:
                if im.width*im.height > 24_000_000:
                    raise ValueError('Imagen demasiado grande')
                return ImageOps.exif_transpose(im).convert('RGB')
        return ClasificadorPlasticos.abrir_imagen(imagen)

    def catalogo(self):
        """Info para la API: esperado vs ya entrenado, no confundirlos."""
        with self._lock:
            entrenadas = list(self.modelo.clases)
            encontradas = sorted({clase for clase, _ in self.memoria.fotos()})
        return {
            'categorias_objetivo': list(CATEGORIAS_OBJETIVO),
            'categorias_entrenadas_pth': entrenadas,
            'categorias_aun_no_entrenadas': [c for c in CATEGORIAS_OBJETIVO if c not in entrenadas],
            'categorias_con_ejemplos': encontradas,
            'rechazo_entrenado': CATEGORIA_RECHAZO in entrenadas,
        }

    def clasificar(self, imagen):
        """Devuelve dict JSON sin forzar clasificación cuando la evidencia es débil.

        OTRO = imágenes comprobadas de OBJETOS NO PLÁSTICOS, no plástico desconocido.
        Un rechazo por simple umbral no prueba que algo no sea plástico.
        """
        foto = self._imagen(imagen)
        with self._lock:
            # Vectorizar una sola vez, relevante para APIs en CPU.
            vector = self.modelo.vectorizar(foto)
            base = self.modelo.predecir_vector(vector)
            refs = self.memoria.sugerir_vector(vector)
            entrenadas = list(self.modelo.clases)

        if refs.get('estado') == 'SUGERENCIA_CON_REFERENCIAS':
            sugerencia = refs['categoria']
            origen = 'ejemplos_etiquetados'
            # La semejanza visual no certifica composición.
            if sugerencia == CATEGORIA_RECHAZO:
                material = MENSAJE_NO_PLASTICO
                estado = 'NO_PLASTICO_SUGERIDO'
            else:
                material = sugerencia
                estado = 'PLASTICO_SUGERIDO'
        else:
            sugerencia = base.get('categoria_sugerida')
            origen = 'modelo_pth'
            material = base['material']
            estado = base['estado']
            if material == MENSAJE_INCIERTO:
                sugerencia = None  # no forzar que una mano sea PVC

        return {
            'resultado': material,
            'estado': estado,
            'es_plastico_sugerido': False if estado in ('NO_PLASTICO_SUGERIDO', 'NO_PLASTICO') else (
                True if estado in ('DETERMINADO', 'PLASTICO_SUGERIDO') else None),
            'categoria_sugerida': sugerencia,
            'origen': origen,
            'requiere_verificacion': True,
            'prediccion_modelo': base,
            'comparacion_ejemplos': refs,
            'categorias_entrenadas': entrenadas,
            'categorias_objetivo': list(CATEGORIAS_OBJETIVO),
            'nota': 'Estimación visual, no prueba química. Un rechazo incierto NO demuestra que un objeto no sea plástico.'
        }

    def aprender_zip(self, archivo_zip: bytes):
        """Enseña inmediatamente por comparación, y guarda fotos para reentrenar."""
        with self._lock:
            r=importar_zip(archivo_zip,self._datos)
            r['referencias_indexadas']=self.memoria.indexar(forzar=True)
            r['categorias_objetivo']=list(CATEGORIAS_OBJETIVO)
            r['pesos_actualizados']=False
            return r

    def reentrenar(self, minimo=10, epocas=40, salida=MODELO_MEJORADO, datos=None):
        """Actualiza pesos en OTRO .pth después de disponer de fotos en TODAS las clases.
        Es lento, ejecutar desde un trabajador y no bloquear una solicitud HTTP.
        Lanza Python externo para que la API quede disponible mientras entrena.
        Retorna ruta del checkpoint y mensaje; se recarga automáticamente al finalizar.
        """
        carpeta=Path(datos) if datos else self._datos
        salida=Path(salida)
        if self._entrenando:
            raise RuntimeError('Ya existe un entrenamiento en curso')
        self._entrenando=True
        try:
            fuente=self.modelo.archivo
            cmd=[sys.executable,str(Path(__file__).with_name('entrenamiento.py')),
                 '--base',str(fuente),'--datos',str(carpeta),'--salida',str(salida),
                 '--minimo',str(minimo),'--epocas',str(epocas)]
            proc=subprocess.run(cmd,capture_output=True,text=True,check=False)
            if proc.returncode:
                raise RuntimeError('Error de entrenamiento:\n'+(proc.stderr+'\n'+proc.stdout)[-4000:])
            # Solo recarga si el archivo que sigue IAPlasticos es salida automática.
            if self._archivo is None and salida.resolve()==MODELO_MEJORADO.resolve():
                self.recargar()
            elif self._archivo and self._archivo.resolve()==salida.resolve():
                self.recargar()
            return {'archivo_pth':str(salida),'pesos_actualizados':True,
                    'categorias_entrenadas':list(self.modelo.clases) if (
                        (self._archivo is None and salida.resolve()==MODELO_MEJORADO.resolve()) or
                        (self._archivo and self._archivo.resolve()==salida.resolve())
                    ) else None,'registro':proc.stdout[-2500:]}
        finally:
            self._entrenando=False
