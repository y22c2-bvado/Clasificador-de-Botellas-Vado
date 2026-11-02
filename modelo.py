"""Recupera el CLIP del proyecto Colab y clasifica recortes o fotografías.

Se reproducen exactamente las operaciones esenciales del notebook:
CLIPProcessor -> get_image_features -> normalización L2 -> Linear -> softmax.
"""
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

try:
    from .config import CATEGORIA_RECHAZO, MARGEN_MINIMO, UMBRAL_MINIMO, MENSAJE_NO_PLASTICO, MENSAJE_INCIERTO, ruta_modelo
except ImportError:
    from config import CATEGORIA_RECHAZO, MARGEN_MINIMO, UMBRAL_MINIMO, MENSAJE_NO_PLASTICO, MENSAJE_INCIERTO, ruta_modelo


class ClasificadorPlasticos:
    def __init__(self, archivo=None, dispositivo=None):
        self.archivo = Path(archivo) if archivo else ruta_modelo()
        if not self.archivo.is_file():
            raise FileNotFoundError(
                f"No existe el modelo: {self.archivo}. "
                "Coloca el archivo .pth en la carpeta models/."
            )
        self.dispositivo = torch.device(
            dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Seguridad: solo lee tensores/tipos permitidos en checkpoints confiables.
        checkpoint = torch.load(
            self.archivo, map_location="cpu", weights_only=True
        )
        obligatorias = {"clases", "dimension", "cabeza"}
        if not isinstance(checkpoint, dict) or not obligatorias.issubset(checkpoint):
            raise ValueError("Formato de checkpoint no compatible con Colab.")
        self.clases = list(checkpoint["clases"])
        base = checkpoint.get("modelo_base", "openai/clip-vit-base-patch32")
        print(f"Cargando CLIP {base} en {self.dispositivo}...")

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "Falta transformers. Ejecuta: pip install -r requirements.txt"
            ) from error

        # En primera ejecución se descargará el modelo base de Hugging Face.
        self.clip = CLIPModel.from_pretrained(base)
        self.procesador = CLIPProcessor.from_pretrained(base)

        # El .pth solo almacena los parámetros visuales reajustados de CLIP.
        # El resto debe provenir del modelo base, no inicializarse al azar.
        pesos = checkpoint.get("pesos_clip", {})
        if not isinstance(pesos, dict):
            raise ValueError("La clave 'pesos_clip' no es un diccionario.")
        parametros = self.clip.state_dict()
        invalidos = [
            k for k, v in pesos.items()
            if k not in parametros
            or not isinstance(v, torch.Tensor)
            or parametros[k].shape != v.shape
        ]
        if invalidos:
            raise ValueError(f"Pesos visuales incompatibles: {invalidos[:5]}")
        self.clip.load_state_dict(pesos, strict=False)
        self.clip = self.clip.to(self.dispositivo).eval()
        self.cabeza = nn.Linear(
            int(checkpoint["dimension"]), len(self.clases)
        ).to(self.dispositivo)
        self.cabeza.load_state_dict(checkpoint["cabeza"], strict=True)
        self.cabeza.eval()
        print("Clases disponibles:", ", ".join(self.clases))
        if CATEGORIA_RECHAZO not in self.clases:
            print(
                "AVISO: el modelo actual aún no aprendió OTRO. "
                "Los umbrales pueden rechazar predicciones ambiguas, "
                "pero no evitan todos los falsos positivos."
            )

    @staticmethod
    def abrir_imagen(imagen: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(imagen, Image.Image):
            return ImageOps.exif_transpose(imagen).convert("RGB")
        with Image.open(imagen) as img:
            return ImageOps.exif_transpose(img).convert("RGB").copy()

    @torch.inference_mode()
    def vectorizar(self, imagen) -> torch.Tensor:
        foto = self.abrir_imagen(imagen)
        entradas = self.procesador(images=foto, return_tensors="pt")
        entradas = {k: v.to(self.dispositivo) for k, v in entradas.items()}
        features = self.clip.get_image_features(pixel_values=entradas["pixel_values"])
        # Compatibilidad con versiones de Transformers que devuelven un ModelOutput.
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return F.normalize(features, dim=-1)

    @torch.inference_mode()
    def predecir(self, imagen, umbral=UMBRAL_MINIMO, margen=MARGEN_MINIMO):
        return self.predecir_vector(self.vectorizar(imagen), umbral=umbral, margen=margen)

    @torch.inference_mode()
    def predecir_vector(self, vector, umbral=UMBRAL_MINIMO, margen=MARGEN_MINIMO):
        if not 0 <= umbral <= 1 or not 0 <= margen <= 1:
            raise ValueError("umbral y margen deben pertenecer al intervalo [0, 1].")
        probs = F.softmax(self.cabeza(vector), dim=-1)[0].cpu()
        indices = torch.argsort(probs, descending=True)
        mejor = int(indices[0])
        segundo = int(indices[1]) if len(indices) > 1 else mejor
        puntuacion = float(probs[mejor])
        diferencia = puntuacion - float(probs[segundo]) if segundo != mejor else puntuacion
        sugerida = self.clases[mejor]

        es_otro = sugerida == CATEGORIA_RECHAZO
        confiable = puntuacion >= umbral and diferencia >= margen
        determinado = (not es_otro) and confiable
        if es_otro:
            razon = "El modelo entrenado reconoce semejanza con objetos NO plásticos etiquetados OTRO."
        elif not confiable:
            razon = "Predicción ambigua o debajo del umbral; revisar manualmente."
        else:
            razon = "Categoría visual sugerida; composición no verificada."
        return {
            "material": MENSAJE_NO_PLASTICO if es_otro and confiable else (sugerida if determinado else MENSAJE_INCIERTO),
            "categoria_sugerida": sugerida,
            "estado": "NO_PLASTICO" if es_otro and confiable else ("DETERMINADO" if determinado else "REVISAR"),
            "motivo": razon,
            "puntuacion_pct": round(puntuacion * 100, 2),
            "margen_pct": round(diferencia * 100, 2),
            "probabilidades_pct": {
                c: round(float(p) * 100, 2) for c, p in zip(self.clases, probs)
            },
            "composicion_verificada": False,
        }
