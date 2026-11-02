"""Inspecciona las clases realmente entrenadas sin cargar el modelo CLIP completo."""
import torch
try:
    from .config import CATEGORIAS_OBJETIVO, ruta_modelo
except ImportError:
    from config import CATEGORIAS_OBJETIVO, ruta_modelo

def main():
    ruta = ruta_modelo()
    datos = torch.load(ruta, map_location='cpu', weights_only=True)
    entrenadas = list(datos['clases'])
    print('Checkpoint:', ruta)
    print('Clases entrenadas:', ', '.join(entrenadas))
    print('Nuevas pendientes:', ', '.join(c for c in CATEGORIAS_OBJETIVO if c not in entrenadas) or 'ninguna')
    print('Capa de salida:', tuple(datos['cabeza']['weight'].shape))
    print('Original protegido; las clases nuevas se incorporan después del reentrenamiento.')

if __name__ == '__main__':
    main()
