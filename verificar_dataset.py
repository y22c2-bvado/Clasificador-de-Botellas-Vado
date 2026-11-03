
"""
VERIFICADOR INTELIGENTE DEL DATASET
Clasificador de Botellas Vado

Categorias:
PET, HDPE, PVC, LDPE, PP, PS, PC, PUR, ABS, OTRO.

Funciones:
- Detectar imagenes danadas.
- Detectar imagenes demasiado pequenas.
- Detectar duplicados exactos.
- Detectar imagenes visualmente similares.
- Detectar etiquetas sospechosas con CLIP.
- Detectar cambios en las cantidades.
- Generar informes CSV y JSON.
- Apartar imagenes sin eliminarlas.
- Restaurar imagenes apartadas.
- Compatible con modelo.py y config.py.

IMPORTANTE:
Las predicciones de CLIP son sugerencias.
No verifican la composicion quimica.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError


# =====================================================
# CONFIGURACION
# =====================================================

CATEGORIAS = [
    "PET",
    "HDPE",
    "PVC",
    "LDPE",
    "PP",
    "PS",
    "PC",
    "PUR",
    "ABS",
    "OTRO",
]

EXTENSIONES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

COLUMNAS = [
    "archivo",
    "categoria_actual",
    "problema",
    "detalle",
    "prediccion_clip",
    "puntuacion_clip_pct",
    "margen_clip_pct",
    "archivo_similar",
    "distancia_visual",
    "accion",
]


# =====================================================
# HASH EXACTO
# =====================================================

def hash_archivo(ruta):

    sha = hashlib.sha256()

    with ruta.open("rb") as archivo:

        for bloque in iter(
            lambda: archivo.read(1024 * 1024),
            b""
        ):
            sha.update(bloque)

    return sha.hexdigest()


# =====================================================
# HASH VISUAL
# =====================================================

def huella_visual(ruta):

    with Image.open(ruta) as imagen:

        imagen = ImageOps.exif_transpose(
            imagen
        )

        imagen = imagen.convert("L")

        imagen = imagen.resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )

        # Evita getdata(), obsoleto en Pillow.
        pixeles = list(imagen.tobytes())

    resultado = 0

    for y in range(8):

        for x in range(8):

            izquierda = pixeles[y * 9 + x]
            derecha = pixeles[y * 9 + x + 1]

            resultado <<= 1

            if izquierda > derecha:
                resultado |= 1

    return resultado


def distancia_visual(a, b):

    return (a ^ b).bit_count()


# =====================================================
# LOCALIZAR ARCHIVOS
# =====================================================

def obtener_imagenes(raiz):

    archivos = []

    for categoria in CATEGORIAS:

        carpeta = raiz / categoria

        if not carpeta.is_dir():

            print(
                f"AVISO: Falta la carpeta {categoria}"
            )

            continue

        for ruta in sorted(
            carpeta.rglob("*")
        ):

            if (
                ruta.is_file()
                and ruta.suffix.lower() in EXTENSIONES
            ):

                archivos.append(
                    (categoria, ruta)
                )

    return archivos


# =====================================================
# CARGAR CLIP
# =====================================================

def cargar_modelo():

    try:

        try:
            from .config import ruta_modelo
            from .modelo import ClasificadorPlasticos

        except ImportError:
            from config import ruta_modelo
            from modelo import ClasificadorPlasticos

        modelo = ClasificadorPlasticos(
            ruta_modelo()
        )

        print(
            "Modelo CLIP cargado correctamente."
        )

        return modelo

    except Exception as error:

        print(
            f"AVISO: No se pudo cargar CLIP: {error}"
        )

        print(
            "La auditoria continuara sin el modelo."
        )

        return None


# =====================================================
# GUARDAR CSV
# =====================================================

def guardar_csv(ruta, filas):

    ruta.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with ruta.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=COLUMNAS,
            extrasaction="ignore"
        )

        escritor.writeheader()
        escritor.writerows(filas)


# =====================================================
# MOVER IMAGEN A REVISION
# =====================================================

def mover_a_revision(
    ruta,
    raiz,
    salida,
    motivo,
    manifiesto
):

    destino = (
        salida
        / "apartados"
        / motivo
        / ruta.relative_to(raiz)
    )

    destino.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if destino.exists():

        numero = 1

        while True:

            candidato = destino.with_name(
                f"{destino.stem}_{numero}"
                f"{destino.suffix}"
            )

            if not candidato.exists():
                destino = candidato
                break

            numero += 1

    # Guardar el movimiento para poder restaurarlo.
    registro = {
        "origen": str(ruta),
        "destino": str(destino),
        "motivo": motivo,
    }

    shutil.move(
        str(ruta),
        str(destino)
    )

    with manifiesto.open(
        "a",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            json.dumps(
                registro,
                ensure_ascii=False
            ) + "\n"
        )

    return str(destino)


# =====================================================
# RESTAURAR IMAGENES
# =====================================================

def restaurar_imagenes(salida):

    manifiesto = (
        salida / "movimientos.jsonl"
    )

    if not manifiesto.exists():

        print(
            "No hay movimientos registrados."
        )

        return

    registros = []

    with manifiesto.open(
        "r",
        encoding="utf-8"
    ) as archivo:

        for linea in archivo:

            if linea.strip():

                try:
                    registros.append(
                        json.loads(linea)
                    )

                except json.JSONDecodeError:
                    print(
                        "AVISO: Registro de movimiento invalido."
                    )

    pendientes = []
    restauradas = 0

    for registro in reversed(registros):

        origen = Path(
            registro["origen"]
        )

        destino = Path(
            registro["destino"]
        )

        if not destino.exists():

            if origen.exists():
                continue

            pendientes.append(registro)
            continue

        if origen.exists():

            print(
                f"No se sobrescribe: {origen}"
            )

            pendientes.append(registro)
            continue

        try:

            origen.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.move(
                str(destino),
                str(origen)
            )

            restauradas += 1

            print(
                f"Restaurada: {origen.name}"
            )

        except OSError as error:

            print(
                f"Error restaurando {origen}: {error}"
            )

            pendientes.append(registro)

    with manifiesto.open(
        "w",
        encoding="utf-8"
    ) as archivo:

        for registro in reversed(pendientes):

            archivo.write(
                json.dumps(
                    registro,
                    ensure_ascii=False
                ) + "\n"
            )

    print(
        f"\nImagenes restauradas: {restauradas}"
    )


# =====================================================
# BUSCAR IMAGENES PARECIDAS
# =====================================================

def buscar_similares(
    huella,
    categoria,
    indice_visual,
    distancia_maxima
):

    candidatos = []

    if distancia_maxima == 0:

        candidatos = indice_visual.get(
            huella,
            []
        )

    else:

        # Para datasets de miles de fotografias.
        # Compara tambien huellas cercanas.
        for huella_anterior, elementos in (
            indice_visual.items()
        ):

            distancia = distancia_visual(
                huella,
                huella_anterior
            )

            if distancia <= distancia_maxima:

                for cat, ruta in elementos:

                    candidatos.append(
                        (cat, ruta, distancia)
                    )

        candidatos.sort(
            key=lambda x: x[2]
        )

        return candidatos

    return [
        (cat, ruta, 0)
        for cat, ruta in candidatos
    ]


# =====================================================
# AUDITORIA COMPLETA
# =====================================================

def analizar(args):

    raiz = Path(
        args.datos
    ).resolve()

    salida = Path(
        args.salida
    ).resolve()

    if not raiz.is_dir():

        raise SystemExit(
            f"No existe Dataset: {raiz}"
        )

    if salida == raiz or raiz in salida.parents:

        raise SystemExit(
            "La salida debe estar fuera del Dataset."
        )

    salida.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.restaurar:

        restaurar_imagenes(salida)
        return

    sello = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    manifiesto = (
        salida / "movimientos.jsonl"
    )

    archivos = obtener_imagenes(
        raiz
    )

    modelo = (
        None
        if args.sin_modelo
        else cargar_modelo()
    )

    vistos_sha = {}

    indice_visual = defaultdict(list)

    registros = []

    conteos_antes = Counter(
        categoria
        for categoria, _ in archivos
    )

    conteos_despues = Counter()

    print("\n" + "=" * 60)
    print("    VERIFICADOR INTELIGENTE DEL DATASET")
    print("=" * 60)

    print(
        f"Imagenes encontradas: {len(archivos)}"
    )

    print(
        f"CLIP disponible: {modelo is not None}"
    )

    print(
        f"Umbral CLIP: {args.umbral}%"
    )

    print(
        f"Distancia visual maxima: "
        f"{args.distancia_visual}"
    )

    print(
        f"Correccion automatica: {args.corregir}"
    )

    print(
        "Apartar sospechosas: "
        f"{args.apartar_sospechosas}"
    )

    print("=" * 60)

    # =================================================
    # PROCESAR FOTOGRAFIAS
    # =================================================

    for numero, (categoria, ruta) in enumerate(
        archivos,
        start=1
    ):

        fila = {
            "archivo": str(
                ruta.relative_to(raiz)
            ),
            "categoria_actual": categoria,
            "problema": "",
            "detalle": "",
            "prediccion_clip": "",
            "puntuacion_clip_pct": "",
            "margen_clip_pct": "",
            "archivo_similar": "",
            "distancia_visual": "",
            "accion": "conservado",
        }

        problemas = []

        archivo_danado = False
        duplicado_misma_categoria = False
        sospechosa = False

        try:

            # =====================================
            # VALIDAR IMAGEN
            # =====================================

            with Image.open(ruta) as imagen:
                imagen.verify()

            with Image.open(ruta) as imagen:

                imagen.load()

                ancho, alto = imagen.size

                if ancho < 32 or alto < 32:

                    problemas.append(
                        "imagen_muy_pequena"
                    )

            # =====================================
            # HASH EXACTO
            # =====================================

            sha = hash_archivo(
                ruta
            )

            anterior = vistos_sha.get(
                sha
            )

            if anterior:

                cat_anterior, ruta_anterior = (
                    anterior
                )

                fila["archivo_similar"] = str(
                    ruta_anterior.relative_to(raiz)
                )

                if cat_anterior == categoria:

                    problemas.append(
                        "duplicado_exacto_misma_categoria"
                    )

                    duplicado_misma_categoria = True

                else:

                    problemas.append(
                        "duplicado_exacto_entre_categorias"
                    )

                    fila["detalle"] = (
                        "Misma imagen con etiquetas "
                        "diferentes."
                    )

            else:

                vistos_sha[sha] = (
                    categoria,
                    ruta
                )

            # =====================================
            # HASH VISUAL
            # =====================================

            if anterior is None:

                huella = huella_visual(
                    ruta
                )

                similares = buscar_similares(
                    huella,
                    categoria,
                    indice_visual,
                    args.distancia_visual
                )

                for (
                    cat_anterior,
                    ruta_anterior,
                    distancia
                ) in similares:

                    if cat_anterior != categoria:

                        problemas.append(
                            "apariencia_similar_entre_categorias"
                        )

                        fila["archivo_similar"] = str(
                            ruta_anterior.relative_to(raiz)
                        )

                        fila["distancia_visual"] = (
                            distancia
                        )

                        break

                    elif distancia > 0:

                        problemas.append(
                            "posible_duplicado_visual"
                        )

                        fila["archivo_similar"] = str(
                            ruta_anterior.relative_to(raiz)
                        )

                        fila["distancia_visual"] = (
                            distancia
                        )

                        break

                indice_visual[huella].append(
                    (categoria, ruta)
                )

            # =====================================
            # VERIFICAR CON CLIP
            # =====================================

            if modelo is not None:

                try:

                    vector = modelo.vectorizar(
                        ruta
                    )

                    pred = modelo.predecir_vector(
                        vector
                    )

                    sugerida = (
                        pred.get("categoria_sugerida")
                        or pred.get("material")
                    )

                    puntuacion = float(
                        pred.get("puntuacion_pct")
                        or 0
                    )

                    margen = float(
                        pred.get("margen_pct")
                        or 0
                    )

                    fila["prediccion_clip"] = (
                        sugerida or ""
                    )

                    fila["puntuacion_clip_pct"] = (
                        round(puntuacion, 2)
                    )

                    fila["margen_clip_pct"] = (
                        round(margen, 2)
                    )

                    if (
                        sugerida
                        and sugerida != categoria
                        and puntuacion >= args.umbral
                        and margen >= args.margen
                    ):

                        problemas.append(
                            "etiqueta_sospechosa_revisar"
                        )

                        sospechosa = True

                except Exception as error:

                    problemas.append(
                        "error_modelo"
                    )

                    fila["detalle"] = (
                        str(error)[:180]
                    )

        except (
            OSError,
            ValueError,
            UnidentifiedImageError
        ) as error:

            archivo_danado = True

            problemas.append(
                "imagen_danada"
            )

            fila["detalle"] = (
                str(error)[:180]
            )

        # =========================================
        # DECISIONES DE CORRECCION
        # =========================================

        motivo_apartado = None

        if args.corregir:

            if archivo_danado:

                motivo_apartado = (
                    "imagenes_danadas"
                )

            elif duplicado_misma_categoria:

                motivo_apartado = (
                    "duplicados_exactos"
                )

        # Apartar sospechosas solamente cuando
        # el usuario lo solicite expresamente.
        if (
            motivo_apartado is None
            and args.apartar_sospechosas
            and sospechosa
        ):

            motivo_apartado = (
                "etiquetas_sospechosas"
            )

        if motivo_apartado:

            try:

                destino = mover_a_revision(
                    ruta,
                    raiz,
                    salida,
                    motivo_apartado,
                    manifiesto
                )

                fila["accion"] = (
                    f"apartado: {destino}"
                )

            except OSError as error:

                fila["accion"] = (
                    "error_al_apartar"
                )

                fila["detalle"] += (
                    f" | {error}"
                )

        fila["problema"] = ";".join(
            dict.fromkeys(problemas)
        )

        registros.append(
            fila
        )

        if ruta.exists():

            conteos_despues[categoria] += 1

        if (
            numero % 100 == 0
            or numero == len(archivos)
        ):

            print(
                f"Revisadas: {numero}/{len(archivos)}"
            )

    # =================================================
    # INFORMES
    # =================================================

    auditoria_csv = (
        salida / f"auditoria_{sello}.csv"
    )

    sospechosas_csv = (
        salida / f"sospechosas_{sello}.csv"
    )

    guardar_csv(
        auditoria_csv,
        registros
    )

    sospechosas = [
        fila
        for fila in registros
        if fila["problema"]
    ]

    guardar_csv(
        sospechosas_csv,
        sospechosas
    )

    resumen_problemas = Counter()

    for fila in registros:

        for problema in fila["problema"].split(";"):

            if problema:

                resumen_problemas[
                    problema
                ] += 1

    apartadas = sum(
        1
        for fila in registros
        if fila["accion"].startswith(
            "apartado:"
        )
    )

    resumen = {
        "fecha": sello,
        "total_revisadas": len(registros),
        "total_sospechosas": len(sospechosas),
        "total_apartadas": apartadas,
        "categorias_antes": dict(
            conteos_antes
        ),
        "categorias_despues": dict(
            conteos_despues
        ),
        "problemas": dict(
            resumen_problemas
        ),
        "modelo_utilizado": (
            modelo is not None
        ),
        "umbral_clip": args.umbral,
        "margen_clip": args.margen,
        "distancia_visual": (
            args.distancia_visual
        ),
        "auditoria_csv": str(
            auditoria_csv
        ),
        "sospechosas_csv": str(
            sospechosas_csv
        ),
    }

    resumen_json = (
        salida / f"resumen_{sello}.json"
    )

    resumen_json.write_text(
        json.dumps(
            resumen,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # =================================================
    # RESULTADOS EN TERMINAL
    # =================================================

    print("\n" + "=" * 60)
    print("             RESULTADO DE AUDITORIA")
    print("=" * 60)

    print("\nCATEGORIAS")

    for categoria in CATEGORIAS:

        antes = conteos_antes[
            categoria
        ]

        despues = conteos_despues[
            categoria
        ]

        diferencia = (
            despues - antes
        )

        print(
            f"{categoria:6} "
            f"Antes: {antes:5} "
            f"Despues: {despues:5} "
            f"Cambio: {diferencia:+}"
        )

    print("-" * 60)

    print(
        f"TOTAL ANTES: {len(archivos)}"
    )

    print(
        "TOTAL DESPUES: "
        f"{sum(conteos_despues.values())}"
    )

    print(
        f"IMAGENES CON ALERTAS: {len(sospechosas)}"
    )

    print(
        f"IMAGENES APARTADAS: {apartadas}"
    )

    print("\nPROBLEMAS DETECTADOS")

    if resumen_problemas:

        for problema, cantidad in (
            resumen_problemas.most_common()
        ):

            print(
                f"{problema}: {cantidad}"
            )

    else:

        print(
            "No se detectaron problemas "
            "con estas comprobaciones."
        )

    print(
        f"\nInforme general: {auditoria_csv}"
    )

    print(
        f"Imagenes sospechosas: {sospechosas_csv}"
    )

    print(
        f"Resumen JSON: {resumen_json}"
    )

    print(
        "\nLas etiquetas nunca se cambian "
        "automaticamente."
    )

    print(
        "CLIP no certifica la composicion "
        "quimica de los materiales."
    )

    print("=" * 60)


# =====================================================
# ARGUMENTOS
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Verificador inteligente del Dataset."
        )
    )

    parser.add_argument(
        "--datos",
        default="Dataset",
        help="Carpeta de fotografias."
    )

    parser.add_argument(
        "--salida",
        default="revision_dataset",
        help="Carpeta para informes."
    )

    parser.add_argument(
        "--sin-modelo",
        action="store_true",
        help="No cargar CLIP."
    )

    parser.add_argument(
        "--umbral",
        type=float,
        default=85.0,
        help=(
            "Puntuacion minima de CLIP para "
            "marcar una discrepancia."
        )
    )

    parser.add_argument(
        "--margen",
        type=float,
        default=15.0,
        help=(
            "Margen minimo de CLIP para "
            "marcar una discrepancia."
        )
    )

    parser.add_argument(
        "--distancia-visual",
        type=int,
        default=0,
        help=(
            "Distancia maxima entre huellas "
            "visuales. Usar 0 por defecto."
        )
    )

    parser.add_argument(
        "--corregir",
        action="store_true",
        help=(
            "Apartar imagenes danadas y "
            "duplicados exactos de la misma clase."
        )
    )

    parser.add_argument(
        "--apartar-sospechosas",
        action="store_true",
        help=(
            "Apartar imagenes cuya etiqueta "
            "contradiga una prediccion fuerte."
        )
    )

    parser.add_argument(
        "--restaurar",
        action="store_true",
        help=(
            "Restaurar imagenes apartadas "
            "registradas en movimientos.jsonl."
        )
    )

    args = parser.parse_args()

    if not 0 <= args.umbral <= 100:

        parser.error(
            "--umbral debe estar entre 0 y 100."
        )

    if not 0 <= args.margen <= 100:

        parser.error(
            "--margen debe estar entre 0 y 100."
        )

    if not 0 <= args.distancia_visual <= 64:

        parser.error(
            "--distancia-visual debe estar "
            "entre 0 y 64."
        )

    analizar(args)


if __name__ == "__main__":
    main()
