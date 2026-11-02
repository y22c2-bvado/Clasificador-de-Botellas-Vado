"""Configuración de IA únicamente. No crea ni modifica servidores."""
from pathlib import Path
RAIZ = Path(__file__).resolve().parent
MODELOS = RAIZ / 'models'
DATOS = RAIZ / 'dataset'
RESULTADOS = RAIZ / 'resultados'
MODELO_ORIGINAL = MODELOS / 'Clasificador_Botellas_Aprendido.pth'
MODELO_MEJORADO = MODELOS / 'Clasificador_Ampliado.pth'
# Las categorías objetivo son configurables: subir otra carpeta al ZIP permite
# agregar otras clases, pero PC/PUR/ABS/OTRO están incluidas desde el principio.
CATEGORIAS_PLASTICOS = ('PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS', 'PC', 'PUR', 'ABS')
CATEGORIA_RECHAZO = 'OTRO'    # únicamente fotos VERIFICADAS de objetos no plásticos
CATEGORIAS_OBJETIVO = (*CATEGORIAS_PLASTICOS, CATEGORIA_RECHAZO)
MENSAJE_NO_PLASTICO = 'NO ES PLÁSTICO'
MENSAJE_INCIERTO = 'NO DETERMINADO' 
UMBRAL_MINIMO = 0.55
MARGEN_MINIMO = 0.15
EXTENSIONES_IMAGEN = {'.jpg','.jpeg','.png','.webp'}

def ruta_modelo(preferir_mejorado=True):
    if preferir_mejorado and MODELO_MEJORADO.is_file():
        return MODELO_MEJORADO
    return MODELO_ORIGINAL
