
"""
Memoria visual CLIP para clasificacion de plasticos.

Categorias:
PET, HDPE, PVC, LDPE, PP, PS, PC, PUR, ABS, OTRO.

Caracteristicas:
- Hasta 100 vecinos configurables.
- Comparacion de similitud coseno.
- Ranking equilibrado por categoria.
- Consenso ponderado de vecinos.
- Deteccion de evidencia insuficiente.
- Cache automatico de embeddings.
- Detecta imagenes nuevas en Dataset.
- Compatible con IAPlasticos.

IMPORTANTE:
La similitud visual no certifica la composicion
quimica de un material.
"""

from collections import defaultdict
from pathlib import Path
import hashlib
import threading

import torch
import torch.nn.functional as F

try:
    from .config import (
        DATOS,
        RESULTADOS,
        EXTENSIONES_IMAGEN,
    )
except ImportError:
    from config import (
        DATOS,
        RESULTADOS,
        EXTENSIONES_IMAGEN,
    )


class MemoriaVisual:

    VERSION_CACHE = 3

    def __init__(
        self,
        modelo,
        carpeta=DATOS,
        top_k=25,
        minimo_ejemplos=5,
        umbral_similitud=0.83,
        umbral_margen=0.04,
        consenso_minimo=0.60,
    ):

        self.modelo = modelo
        self.carpeta = Path(carpeta)

        self.top_k = max(
            1, min(100, int(top_k))
        )

        self.minimo_ejemplos = max(
            1, int(minimo_ejemplos)
        )

        self.umbral_similitud = float(
            umbral_similitud
        )

        self.umbral_margen = float(
            umbral_margen
        )

        self.consenso_minimo = float(
            consenso_minimo
        )

        if not 0 <= self.consenso_minimo <= 1:
            raise ValueError(
                "consenso_minimo debe estar entre 0 y 1"
            )

        if not -1 <= self.umbral_similitud <= 1:
            raise ValueError(
                "umbral_similitud fuera de rango"
            )

        if self.umbral_margen < 0:
            raise ValueError(
                "umbral_margen no puede ser negativo"
            )

        self._index = []
        self._matriz = None
        self._etiquetas = []
        self._firma = None

        self._lock = threading.RLock()

        RESULTADOS.mkdir(
            parents=True,
            exist_ok=True
        )

        self._cache = (
            RESULTADOS / "referencias.pt"
        )

    # ==============================================
    # LOCALIZAR IMAGENES
    # ==============================================

    def fotos(self):

        if not self.carpeta.exists():
            return []

        imagenes = []

        extensiones = {
            str(ext).lower()
            for ext in EXTENSIONES_IMAGEN
        }

        for carpeta_clase in sorted(
            self.carpeta.iterdir()
        ):

            if not carpeta_clase.is_dir():
                continue

            categoria = carpeta_clase.name

            for archivo in sorted(
                carpeta_clase.rglob("*")
            ):

                if not archivo.is_file():
                    continue

                if archivo.suffix.lower() not in extensiones:
                    continue

                imagenes.append(
                    (categoria, archivo)
                )

        return imagenes

    # ==============================================
    # IDENTIFICAR CAMBIOS EN EL DATASET
    # ==============================================

    def _fingerprint(self, fotos):

        archivo = Path(self.modelo.archivo)

        partes = [
            f"cache:{self.VERSION_CACHE}"
        ]

        if archivo.exists():

            st = archivo.stat()

            partes.append(
                f"{archivo.resolve()}:"
                f"{st.st_size}:"
                f"{st.st_mtime_ns}"
            )

        else:
            partes.append(
                str(archivo.resolve())
            )

        for categoria, ruta in fotos:

            st = ruta.stat()

            partes.append(
                f"{categoria}/{ruta.resolve()}:"
                f"{st.st_size}:"
                f"{st.st_mtime_ns}"
            )

        contenido = "\n".join(partes)

        return hashlib.sha256(
            contenido.encode("utf-8")
        ).hexdigest()

    # ==============================================
    # PREPARAR MATRIZ
    # ==============================================

    def _preparar_matriz(self):

        vectores = []
        etiquetas = []

        for categoria, vector in self._index:

            tensor = torch.as_tensor(
                vector,
                dtype=torch.float32
            ).detach().cpu().flatten()

            if not bool(
                torch.isfinite(tensor).all()
            ):
                continue

            norma = float(
                torch.linalg.vector_norm(tensor)
            )

            if norma <= 0:
                continue

            tensor = F.normalize(
                tensor,
                dim=0
            )

            vectores.append(tensor)
            etiquetas.append(categoria)

        if not vectores:

            self._matriz = None
            self._etiquetas = []
            return

        try:

            self._matriz = torch.stack(
                vectores,
                dim=0
            )

        except RuntimeError as error:

            raise ValueError(
                "Las referencias tienen dimensiones "
                "incompatibles."
            ) from error

        self._etiquetas = etiquetas

    # ==============================================
    # INDEXAR Y ACTUALIZAR CACHE
    # ==============================================

    def indexar(self, forzar=False):

        with self._lock:

            fotos = self.fotos()
            firma = self._fingerprint(fotos)

            if (
                not forzar
                and firma == self._firma
            ):
                return len(self._etiquetas)

            if (
                not forzar
                and self._cache.exists()
            ):

                try:

                    datos = torch.load(
                        self._cache,
                        map_location="cpu",
                        weights_only=True
                    )

                    if (
                        datos.get("firma") == firma
                        and datos.get("version")
                        == self.VERSION_CACHE
                    ):

                        self._index = datos[
                            "referencias"
                        ]

                        self._preparar_matriz()
                        self._firma = firma

                        return len(self._etiquetas)

                except (
                    KeyError,
                    OSError,
                    RuntimeError,
                    ValueError,
                    TypeError,
                    EOFError,
                ):
                    pass

            referencias = []

            print(
                "Actualizando memoria visual..."
            )

            for numero, (categoria, ruta) in enumerate(
                fotos,
                start=1
            ):

                try:

                    vector = (
                        self.modelo
                        .vectorizar(ruta)
                        .detach()
                        .cpu()
                        .float()
                        .flatten()
                    )

                    if not bool(
                        torch.isfinite(vector).all()
                    ):
                        continue

                    if float(
                        torch.linalg.vector_norm(vector)
                    ) <= 0:
                        continue

                    vector = F.normalize(
                        vector,
                        dim=0
                    )

                    referencias.append(
                        (categoria, vector)
                    )

                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                ) as error:

                    print(
                        f"Imagen omitida: {ruta}"
                    )

                    print(
                        f"Motivo: {error}"
                    )

                if numero % 100 == 0:

                    print(
                        f"  {numero}/{len(fotos)}"
                    )

            self._index = referencias
            self._preparar_matriz()
            self._firma = firma

            temporal = self._cache.with_suffix(
                ".tmp"
            )

            try:

                torch.save(
                    {
                        "version": self.VERSION_CACHE,
                        "firma": firma,
                        "referencias": self._index,
                    },
                    temporal
                )

                temporal.replace(
                    self._cache
                )

            except OSError as error:

                print(
                    f"No se pudo guardar cache: {error}"
                )

            print(
                f"Referencias indexadas: "
                f"{len(self._etiquetas)}"
            )

            return len(self._etiquetas)

    # ==============================================
    # COMPARAR VECTOR CON REFERENCIAS
    # ==============================================

    def sugerir_vector(self, vector):

        self.indexar()

        with self._lock:

            if self._matriz is None:

                return {
                    "categoria": None,
                    "estado": "SIN_EJEMPLOS",
                    "similitud_coseno": 0.0,
                    "margen_coseno": 0.0,
                    "ejemplos_clase": 0,
                    "referencias_usadas": 0,
                    "consenso_vecinos_pct": 0.0,
                    "ranking": [],
                    "motivos": [
                        "No hay referencias disponibles"
                    ],
                }

            consulta = torch.as_tensor(
                vector,
                dtype=torch.float32
            ).detach().cpu().flatten()

            if not bool(
                torch.isfinite(consulta).all()
            ):
                raise ValueError(
                    "Vector de consulta no valido."
                )

            if float(
                torch.linalg.vector_norm(consulta)
            ) <= 0:

                raise ValueError(
                    "Vector de consulta vacio."
                )

            consulta = F.normalize(
                consulta,
                dim=0
            )

            if (
                consulta.numel()
                != self._matriz.shape[1]
            ):
                raise ValueError(
                    "Dimension del vector incompatible."
                )

            similitudes = (
                self._matriz @ consulta
            ).clamp(-1.0, 1.0)

            # ======================================
            # AGRUPAR POR CATEGORIA
            # ======================================

            puntajes = defaultdict(list)

            for categoria, similitud in zip(
                self._etiquetas,
                similitudes.tolist()
            ):

                puntajes[categoria].append(
                    float(similitud)
                )

            ranking = []

            for categoria, valores in puntajes.items():

                mejores = sorted(
                    valores,
                    reverse=True
                )[:self.top_k]

                promedio = (
                    sum(mejores) / len(mejores)
                )

                # Ponderacion adicional para
                # destacar referencias mas cercanas.
                pesos = [
                    max(0.0, valor) ** 4
                    for valor in mejores
                ]

                suma_pesos = sum(pesos)

                if suma_pesos > 0:

                    promedio_ponderado = (
                        sum(
                            valor * peso
                            for valor, peso
                            in zip(mejores, pesos)
                        ) / suma_pesos
                    )

                else:
                    promedio_ponderado = promedio

                ranking.append({
                    "categoria": categoria,
                    "similitud": promedio_ponderado,
                    "mejor_similitud": max(valores),
                    "ejemplos": len(valores),
                    "referencias_usadas": len(mejores),
                })

            ranking.sort(
                key=lambda x: x["similitud"],
                reverse=True
            )

            ganador = ranking[0]

            segunda = (
                ranking[1]
                if len(ranking) > 1
                else None
            )

            categoria = ganador["categoria"]

            similitud = ganador["similitud"]

            margen = (
                similitud - segunda["similitud"]
                if segunda is not None
                else 0.0
            )

            # ======================================
            # CONSENSO PONDERADO DE VECINOS
            # ======================================

            cantidad_vecinos = min(
                self.top_k,
                len(self._etiquetas)
            )

            valores, indices = torch.topk(
                similitudes,
                k=cantidad_vecinos
            )

            votos = defaultdict(float)
            peso_total = 0.0

            for valor, indice in zip(
                valores.tolist(),
                indices.tolist()
            ):

                etiqueta = self._etiquetas[
                    indice
                ]

                peso = max(
                    0.0, float(valor)
                ) ** 4

                votos[etiqueta] += peso
                peso_total += peso

            consenso = (
                votos.get(categoria, 0.0)
                / peso_total
                if peso_total > 0
                else 0.0
            )

            # ======================================
            # VERIFICAR EVIDENCIA
            # ======================================

            suficientes = (
                ganador["ejemplos"]
                >= self.minimo_ejemplos
            )

            similitud_aceptable = (
                similitud >= self.umbral_similitud
            )

            margen_aceptable = (
                margen >= self.umbral_margen
            )

            consenso_aceptable = (
                consenso >= self.consenso_minimo
            )

            hay_competencia = (
                segunda is not None
            )

            condiciones = {
                "ejemplos_suficientes": suficientes,
                "similitud_aceptable":
                    similitud_aceptable,
                "margen_aceptable":
                    margen_aceptable,
                "consenso_aceptable":
                    consenso_aceptable,
                "multiples_categorias":
                    hay_competencia,
            }

            consistente = all(
                condiciones.values()
            )

            estado = (
                "SUGERENCIA_CON_REFERENCIAS"
                if consistente
                else "EVIDENCIA_INSUFICIENTE"
            )

            motivos = []

            if not suficientes:
                motivos.append(
                    "Pocas referencias en la categoria"
                )

            if not similitud_aceptable:
                motivos.append(
                    "Similitud visual inferior al umbral"
                )

            if not margen_aceptable:
                motivos.append(
                    "Diferencia pequena entre categorias"
                )

            if not consenso_aceptable:
                motivos.append(
                    "Los vecinos cercanos no coinciden"
                )

            if not hay_competencia:
                motivos.append(
                    "No hay otras categorias comparables"
                )

            if consistente:
                motivos.append(
                    "Referencias visuales consistentes"
                )

            # ======================================
            # RESULTADO
            # ======================================

            return {

                "categoria": categoria,

                "estado": estado,

                "similitud_coseno": round(
                    similitud, 4
                ),

                "similitud_pct": round(
                    similitud * 100, 2
                ),

                "margen_coseno": round(
                    margen, 4
                ),

                "ejemplos_clase": ganador[
                    "ejemplos"
                ],

                "referencias_usadas": ganador[
                    "referencias_usadas"
                ],

                "consenso_vecinos_pct": round(
                    consenso * 100, 2
                ),

                "segunda_categoria": (
                    segunda["categoria"]
                    if segunda else None
                ),

                "ranking": [
                    {
                        "categoria": r["categoria"],
                        "similitud": round(
                            r["similitud"], 4
                        ),
                        "ejemplos": r["ejemplos"],
                    }
                    for r in ranking
                ],

                "condiciones": condiciones,

                "motivos": motivos,

                "nota": (
                    "Similitud visual mediante CLIP. "
                    "No representa confianza calibrada "
                    "ni verifica composicion quimica."
                ),
            }

    # ==============================================
    # CLASIFICAR IMAGEN DIRECTAMENTE
    # ==============================================

    def sugerir(self, imagen):

        vector = self.modelo.vectorizar(
            imagen
        )

        return self.sugerir_vector(
            vector
        )
