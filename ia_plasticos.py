
"""
Clasificador de plasticos: CLIP + memoria visual.

Categorias:
PET, HDPE, PVC, LDPE, PP, PS, PC, PUR, ABS, OTRO.

- Una sola categoria como resultado.
- Numero de vecinos configurable.
- Comparacion de modelo y referencias.
- Rechazo conservador ante incertidumbre.
- Compatible con probar_imagen.py.
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps

try:
    from .config import (
        DATOS,
        MODELO_MEJORADO,
        CATEGORIAS_OBJETIVO,
        CATEGORIA_RECHAZO,
        ruta_modelo,
    )
    from .importador import importar_zip
    from .modelo import ClasificadorPlasticos
    from .referencias import MemoriaVisual

except ImportError:
    from config import (
        DATOS,
        MODELO_MEJORADO,
        CATEGORIAS_OBJETIVO,
        CATEGORIA_RECHAZO,
        ruta_modelo,
    )
    from importador import importar_zip
    from modelo import ClasificadorPlasticos
    from referencias import MemoriaVisual


class IAPlasticos:

    def __init__(
        self,
        modelo=None,
        dispositivo=None,
        datos=DATOS,
        top_k=None,
    ):
        self._archivo = Path(modelo) if modelo else None
        self._dispositivo = dispositivo
        self._datos = Path(datos)

        if top_k is None:
            top_k = os.environ.get(
                "REFERENCIAS_TOP_K", "25"
            )

        self.top_k = max(
            1, min(100, int(top_k))
        )

        self._lock = threading.RLock()
        self._entrenando = False

        self.recargar()

    # ==============================================
    # CARGAR MODELO Y REFERENCIAS
    # ==============================================

    def recargar(self):

        with self._lock:

            ruta = (
                self._archivo
                if self._archivo is not None
                else ruta_modelo()
            )

            nuevo_modelo = ClasificadorPlasticos(
                ruta,
                dispositivo=self._dispositivo,
            )

            nueva_memoria = MemoriaVisual(
                nuevo_modelo,
                carpeta=self._datos,
                top_k=self.top_k,
            )

            self.modelo = nuevo_modelo
            self.memoria = nueva_memoria

    # ==============================================
    # CAMBIAR CANTIDAD DE VECINOS
    # ==============================================

    def configurar_vecinos(self, cantidad):

        cantidad = int(cantidad)

        if not 1 <= cantidad <= 100:
            raise ValueError(
                "La cantidad debe estar entre 1 y 100."
            )

        with self._lock:
            self.top_k = cantidad
            self.memoria.top_k = cantidad

        return self.top_k

    # ==============================================
    # PREPARAR IMAGEN
    # ==============================================

    @staticmethod
    def _imagen(imagen):

        if isinstance(imagen, (bytes, bytearray)):

            if len(imagen) > 10 * 1024 * 1024:
                raise ValueError(
                    "La imagen supera los 10 MB."
                )

            with Image.open(
                io.BytesIO(imagen)
            ) as im:

                if im.width * im.height > 24_000_000:
                    raise ValueError(
                        "La imagen es demasiado grande."
                    )

                return ImageOps.exif_transpose(
                    im
                ).convert("RGB")

        return ClasificadorPlasticos.abrir_imagen(
            imagen
        )

    # ==============================================
    # CATALOGO
    # ==============================================

    def catalogo(self):

        with self._lock:

            entrenadas = list(
                self.modelo.clases
            )

            encontradas = sorted({
                categoria
                for categoria, _ in self.memoria.fotos()
            })

        return {
            "categorias_objetivo": list(
                CATEGORIAS_OBJETIVO
            ),
            "categorias_entrenadas_pth": entrenadas,
            "categorias_aun_no_entrenadas": [
                c for c in CATEGORIAS_OBJETIVO
                if c not in entrenadas
            ],
            "categorias_con_ejemplos": encontradas,
            "rechazo_entrenado": (
                CATEGORIA_RECHAZO in entrenadas
            ),
            "vecinos_configurados": self.top_k,
        }

    # ==============================================
    # CLASIFICACION
    # ==============================================

    def clasificar(self, imagen, detalles=False):

        foto = self._imagen(imagen)

        with self._lock:

            vector = self.modelo.vectorizar(foto)

            base = self.modelo.predecir_vector(
                vector
            )

            refs = self.memoria.sugerir_vector(
                vector
            )

            entrenadas = list(
                self.modelo.clases
            )

        categoria_modelo = base.get(
            "categoria_sugerida"
        )

        estado_modelo = base.get(
            "estado", "NO_DETERMINADO"
        )

        puntuacion = float(
            base.get("puntuacion_pct") or 0
        )

        margen_modelo = float(
            base.get("margen_pct") or 0
        )

        probabilidades = base.get(
            "probabilidades_pct"
        ) or {}

        categoria_refs = refs.get(
            "categoria"
        )

        estado_refs = refs.get(
            "estado", "EVIDENCIA_INSUFICIENTE"
        )

        similitud = float(
            refs.get("similitud_coseno") or 0
        )

        margen_refs = float(
            refs.get("margen_coseno") or 0
        )

        consenso = float(
            refs.get("consenso_vecinos_pct") or 0
        )

        permitidas = set(
            CATEGORIAS_OBJETIVO
        )

        permitidas.discard(
            CATEGORIA_RECHAZO
        )

        permitidas.intersection_update(
            entrenadas
        )

        modelo_seguro = (
            estado_modelo == "DETERMINADO"
        )

        referencias_seguras = (
            estado_refs
            == "SUGERENCIA_CON_REFERENCIAS"
        )

        acuerdo = (
            categoria_modelo is not None
            and categoria_modelo == categoria_refs
        )

        modelo_es_plastico = (
            categoria_modelo in permitidas
        )

        refs_es_plastico = (
            categoria_refs in permitidas
        )

        # Evidencia fuerte del modelo.
        # Estos umbrales son experimentales;
        # deben evaluarse con datos independientes.
        modelo_muy_fuerte = (
            modelo_seguro
            and puntuacion >= 80
            and margen_modelo >= 35
        )

        # Evidencia moderada de las referencias.
        # No equivale a composicion certificada.
        referencias_moderadas = (
            similitud >= 0.78
            and margen_refs >= 0.025
            and consenso >= 55
        )

        desacuerdo_fuerte = (
            modelo_seguro
            and referencias_seguras
            and not acuerdo
        )

        modelo_rechaza = (
            modelo_seguro
            and categoria_modelo == CATEGORIA_RECHAZO
        )

        refs_rechazan = (
            referencias_seguras
            and categoria_refs == CATEGORIA_RECHAZO
        )

        prob_otro = float(
            probabilidades.get(
                CATEGORIA_RECHAZO, 0
            ) or 0
        )

        # ==========================================
        # DECISION
        # ==========================================

        resultado = CATEGORIA_RECHAZO
        estado = "INCIERTO"
        origen = "evaluacion_conjunta"

        motivo = (
            "Evidencia insuficiente para asignar "
            "con fiabilidad una categoria conocida."
        )

        # 1. Modelo determina OTRO.
        if modelo_rechaza:

            resultado = CATEGORIA_RECHAZO
            estado = "OTRO"
            origen = "modelo_pth"

            motivo = (
                "El modelo identifica OTRO."
            )

        # 2. Referencias fuertes determinan OTRO.
        elif refs_rechazan:

            resultado = CATEGORIA_RECHAZO
            estado = "OTRO"
            origen = "memoria_visual"

            motivo = (
                "La memoria visual aporta "
                "evidencia fuerte de OTRO."
            )

        # 3. Dos fuentes fuertes coinciden.
        elif (
            modelo_seguro
            and modelo_es_plastico
            and referencias_seguras
            and acuerdo
        ):

            resultado = categoria_modelo
            estado = "CLASIFICADO"
            origen = "modelo_y_referencias"

            motivo = (
                "El modelo y las referencias "
                "coinciden con evidencia suficiente."
            )

        # 4. Dos fuentes fuertes discrepan.
        elif desacuerdo_fuerte:

            resultado = CATEGORIA_RECHAZO
            estado = "INCIERTO"
            origen = "conflicto"

            motivo = (
                "El modelo y las referencias "
                "presentan una contradiccion fuerte."
            )

        # 5. Modelo muy fuerte, sin contradiccion.
        elif (
            modelo_muy_fuerte
            and modelo_es_plastico
        ):

            resultado = categoria_modelo
            estado = "CLASIFICADO"
            origen = "modelo_pth"

            motivo = (
                "El modelo tiene una puntuacion "
                "y un margen elevados."
            )

        # 6. Modelo determinado con apoyo moderado.
        elif (
            modelo_seguro
            and modelo_es_plastico
            and acuerdo
            and referencias_moderadas
        ):

            resultado = categoria_modelo
            estado = "CLASIFICADO"
            origen = "modelo_y_referencias"

            motivo = (
                "El modelo y la memoria visual "
                "coinciden con apoyo moderado."
            )

        # 7. Modelo indeciso, referencias fuertes.
        elif (
            not modelo_seguro
            and referencias_seguras
            and refs_es_plastico
            and prob_otro < 40
        ):

            resultado = categoria_refs
            estado = "CLASIFICADO"
            origen = "memoria_visual"

            motivo = (
                "La memoria visual aporta "
                "evidencia fuerte de la categoria."
            )

        # 8. En los demas casos se devuelve OTRO
        # por compatibilidad con la interfaz,
        # pero el estado interno sera INCIERTO.

        respuesta = {
            "resultado": resultado
        }

        if not detalles:
            return respuesta

        respuesta.update({

            "estado": estado,

            "origen": origen,

            "motivo": motivo,

            "categoria_modelo": categoria_modelo,

            "categoria_referencias": categoria_refs,

            "estado_modelo": estado_modelo,

            "estado_referencias": estado_refs,

            "acuerdo": acuerdo,

            "desacuerdo_fuerte": desacuerdo_fuerte,

            "puntuacion_modelo_pct": round(
                puntuacion, 2
            ),

            "margen_modelo_pct": round(
                margen_modelo, 2
            ),

            "probabilidad_otro_pct": round(
                prob_otro, 2
            ),

            "similitud_coseno": round(
                similitud, 4
            ),

            "margen_referencias": round(
                margen_refs, 4
            ),

            "consenso_vecinos_pct": round(
                consenso, 2
            ),

            "vecinos_configurados": self.top_k,

            "modelo_muy_fuerte": modelo_muy_fuerte,

            "referencias_moderadas": (
                referencias_moderadas
            ),

            "prediccion_modelo": base,

            "comparacion_ejemplos": refs,

            "categorias_entrenadas": entrenadas,

            "requiere_verificacion": True,

            "nota": (
                "OTRO puede significar material ajeno "
                "a las clases conocidas o incertidumbre. "
                "Las puntuaciones visuales no verifican "
                "la composicion quimica."
            ),
        })

        return respuesta

    # ==============================================
    # IMPORTAR FOTOGRAFIAS
    # ==============================================

    def aprender_zip(self, archivo_zip: bytes):

        with self._lock:

            resultado = importar_zip(
                archivo_zip,
                self._datos
            )

            resultado["referencias_indexadas"] = (
                self.memoria.indexar(forzar=True)
            )

            resultado["categorias_objetivo"] = list(
                CATEGORIAS_OBJETIVO
            )

            resultado["pesos_actualizados"] = False

            return resultado

    # ==============================================
    # REENTRENAMIENTO
    # ==============================================

    def reentrenar(
        self,
        minimo=10,
        epocas=40,
        salida=MODELO_MEJORADO,
        datos=None,
    ):

        carpeta = (
            Path(datos)
            if datos is not None
            else self._datos
        )

        salida = Path(salida)

        with self._lock:

            if self._entrenando:
                raise RuntimeError(
                    "Ya existe un entrenamiento en curso."
                )

            self._entrenando = True
            fuente = Path(self.modelo.archivo)

        respaldo_temporal = None

        try:

            # Evitar utilizar como entrada y salida
            # el mismo checkpoint durante el proceso.
            if fuente.resolve() == salida.resolve():

                with tempfile.NamedTemporaryFile(
                    suffix=".pth",
                    prefix="base_entrenamiento_",
                    delete=False,
                    dir=salida.parent,
                ) as temporal:

                    respaldo_temporal = Path(
                        temporal.name
                    )

                shutil.copy2(
                    fuente,
                    respaldo_temporal
                )

                fuente = respaldo_temporal

            comando = [
                sys.executable,
                str(
                    Path(__file__).with_name(
                        "entrenamiento.py"
                    )
                ),
                "--base", str(fuente),
                "--datos", str(carpeta),
                "--salida", str(salida),
                "--minimo", str(minimo),
                "--epocas", str(epocas),
            ]

            proceso = subprocess.run(
                comando,
                capture_output=True,
                text=True,
                check=False,
            )

            if proceso.returncode != 0:

                detalle = (
                    proceso.stderr
                    + "\n"
                    + proceso.stdout
                )[-4000:]

                raise RuntimeError(
                    "Error de entrenamiento:\n"
                    + detalle
                )

            debe_recargar = (
                (
                    self._archivo is None
                    and salida.resolve()
                    == Path(MODELO_MEJORADO).resolve()
                )
                or
                (
                    self._archivo is not None
                    and self._archivo.resolve()
                    == salida.resolve()
                )
            )

            if debe_recargar:
                self.recargar()

            return {
                "archivo_pth": str(salida),
                "pesos_actualizados": True,
                "categorias_entrenadas": (
                    list(self.modelo.clases)
                    if debe_recargar else None
                ),
                "registro": proceso.stdout[-2500:],
            }

        finally:

            if respaldo_temporal is not None:
                respaldo_temporal.unlink(
                    missing_ok=True
                )

            with self._lock:
                self._entrenando = False
