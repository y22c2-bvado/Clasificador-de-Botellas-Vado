
"""
CLASIFICADOR DE PLASTICOS VADO
YOLO + SAM 2 + CLIP + EASY OCR

Deteccion automatica de objetos candidatos.
Recorte original y segmentado.
Seleccion de la mayor puntuacion CLIP.
Filtro de objetos pequenos y detecciones repetidas.

NOTA:
YOLO preentrenado detecta objetos, no materiales.
La puntuacion CLIP no certifica el polimero.
"""

import argparse
import io
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
from PIL import Image


BASE = Path(__file__).resolve().parent
SALIDAS = BASE / "resultados" / "yolo_clip_avanzado"

CATEGORIAS = {
    "PET", "HDPE", "PVC", "LDPE", "PP",
    "PS", "PC", "PUR", "ABS", "OTRO"
}

CANDIDATOS = {
    "bottle", "cup", "bowl", "fork", "knife",
    "spoon", "toothbrush", "vase", "chair",
    "bench", "backpack", "handbag", "suitcase",
    "umbrella", "sports ball", "frisbee",
    "keyboard", "mouse", "remote", "cell phone",
    "laptop", "tv", "scissors", "clock",
    "refrigerator", "microwave", "toaster",
    "potted plant", "dining table", "surfboard",
    "skateboard", "traffic light"
}

LECTOR_OCR = None


def log(mensaje):
    print(mensaje, flush=True)


# ==========================================
# SELECCIONAR Y ABRIR IMAGEN
# ==========================================

def seleccionar_imagen():
    raiz = Tk()
    raiz.withdraw()
    raiz.attributes("-topmost", True)
    raiz.update()

    try:
        return filedialog.askopenfilename(
            parent=raiz,
            title="Seleccionar fotografia",
            filetypes=[
                (
                    "Imagenes",
                    "*.jpg *.jpeg *.png *.webp *.bmp"
                ),
                ("Todos los archivos", "*.*")
            ]
        )
    finally:
        raiz.destroy()


def leer_imagen(ruta):
    datos = np.fromfile(
        str(ruta),
        dtype=np.uint8
    )

    imagen = cv2.imdecode(
        datos,
        cv2.IMREAD_COLOR
    )

    if imagen is None:
        raise ValueError(
            f"No se pudo abrir: {ruta}"
        )

    return imagen


# ==========================================
# MASCARAS
# ==========================================

def mascara_poligono(puntos, alto, ancho):
    puntos = np.asarray(
        puntos,
        dtype=np.int32
    )

    if puntos.ndim != 2 or len(puntos) < 3:
        return None

    mascara = np.zeros(
        (alto, ancho),
        dtype=np.uint8
    )

    cv2.fillPoly(
        mascara,
        [puntos],
        255
    )

    return mascara


def mascara_desde_sam(resultado, alto, ancho):
    if resultado.masks is None:
        return None

    if len(resultado.masks.data) == 0:
        return None

    datos = (
        resultado.masks.data[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    mascara = cv2.resize(
        datos,
        (ancho, alto),
        interpolation=cv2.INTER_NEAREST
    )

    mascara = (
        (mascara > 0.5).astype(np.uint8) * 255
    )

    if cv2.countNonZero(mascara) == 0:
        return None

    return mascara


# ==========================================
# CALCULAR SUPERPOSICION ENTRE CAJAS
# ==========================================

def calcular_iou(caja_a, caja_b):
    x1, y1, x2, y2 = caja_a
    a1, b1, a2, b2 = caja_b

    ancho_inter = max(
        0,
        min(x2, a2) - max(x1, a1)
    )

    alto_inter = max(
        0,
        min(y2, b2) - max(y1, b1)
    )

    inter = ancho_inter * alto_inter

    area_a = max(0, x2 - x1) * max(0, y2 - y1)
    area_b = max(0, a2 - a1) * max(0, b2 - b1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


# ==========================================
# DETECCION AUTOMATICA YOLO
# ==========================================

def detectar_objetos(
    imagen,
    pesos,
    confianza,
    max_objetos,
    todos=False,
    area_minima=0.025
):
    from ultralytics import YOLO

    log("[YOLO] Cargando detector...")

    modelo = YOLO(pesos)

    log("[YOLO] Detectando objetos...")

    resultados = modelo.predict(
        source=imagen,
        conf=confianza,
        imgsz=960,
        device="cpu",
        verbose=False
    )

    if not resultados:
        return []

    resultado = resultados[0]

    if resultado.boxes is None:
        return []

    alto, ancho = imagen.shape[:2]

    area_imagen = max(1, alto * ancho)

    objetos = []

    for indice, caja in enumerate(resultado.boxes):
        clase = int(caja.cls[0])

        nombre = str(
            resultado.names[clase]
        ).lower()

        if not todos and nombre not in CANDIDATOS:
            continue

        x1, y1, x2, y2 = map(
            int,
            caja.xyxy[0].tolist()
        )

        x1 = max(0, min(x1, ancho))
        x2 = max(0, min(x2, ancho))
        y1 = max(0, min(y1, alto))
        y2 = max(0, min(y2, alto))

        if x2 <= x1 or y2 <= y1:
            continue

        area_objeto = (
            (x2 - x1) * (y2 - y1)
        )

        proporcion = area_objeto / area_imagen

        # Evitar falsos objetos muy pequenos:
        # por ejemplo un logotipo dentro de un monitor.
        if proporcion < area_minima:
            log(
                f"[YOLO] Descartado {nombre}: "
                f"area {proporcion:.2%}"
            )
            continue

        coordenadas = (x1, y1, x2, y2)

        # Ampliar ligeramente una TV o monitor
        # cuando YOLO ya lo detecto correctamente.
        if nombre == "tv":
            margen_x = int((x2 - x1) * 0.05)
            margen_y = int((y2 - y1) * 0.05)

            coordenadas = (
                max(0, x1 - margen_x),
                max(0, y1 - margen_y),
                min(ancho, x2 + margen_x),
                min(alto, y2 + margen_y)
            )

        mascara = None

        if (
            resultado.masks is not None
            and indice < len(resultado.masks.xy)
        ):
            mascara = mascara_poligono(
                resultado.masks.xy[indice],
                alto,
                ancho
            )

        objetos.append({
            "caja": coordenadas,
            "nombre": nombre,
            "confianza": float(caja.conf[0]),
            "proporcion_imagen": proporcion,
            "mascara": mascara
        })

    # Ordenar por confianza de deteccion.
    objetos.sort(
        key=lambda o: (
            o["confianza"],
            o["proporcion_imagen"]
        ),
        reverse=True
    )

    # Eliminar detecciones repetidas.
    depurados = []

    for objeto in objetos:
        repetido = any(
            calcular_iou(
                objeto["caja"],
                anterior["caja"]
            ) > 0.75
            for anterior in depurados
        )

        if repetido:
            continue

        depurados.append(objeto)

        if len(depurados) >= max_objetos:
            break

    log(
        f"[YOLO] Objetos validos: {len(depurados)}"
    )

    for indice, objeto in enumerate(depurados, 1):
        log(
            f"  {indice}. {objeto['nombre']} "
            f"{objeto['confianza'] * 100:.2f}% "
            f"(area: {objeto['proporcion_imagen']:.2%})"
        )

    return depurados


# ==========================================
# SAM 2
# ==========================================

def cargar_sam():
    try:
        from ultralytics import SAM

        pesos = BASE / "sam2_t.pt"

        archivo = (
            str(pesos)
            if pesos.exists()
            else "sam2_t.pt"
        )

        log("[SAM2] Cargando modelo...")

        modelo = SAM(archivo)

        log("[SAM2] Modelo listo.")

        return modelo

    except Exception as error:
        log(f"[SAM2] Error: {error}")
        return None


def refinar_sam(modelo, imagen, caja):
    if modelo is None:
        return None

    try:
        log("[SAM2] Refinando objeto...")

        resultados = modelo.predict(
            source=imagen,
            bboxes=list(map(int, caja)),
            device="cpu",
            verbose=False
        )

        if not resultados:
            return None

        alto, ancho = imagen.shape[:2]

        return mascara_desde_sam(
            resultados[0],
            alto,
            ancho
        )

    except Exception as error:
        log(f"[SAM2] Error: {error}")
        return None


# ==========================================
# CREAR RECORTES
# ==========================================

def preparar_recortes(imagen, caja, mascara):
    alto, ancho = imagen.shape[:2]

    x1, y1, x2, y2 = map(int, caja)

    x1 = max(0, min(x1, ancho))
    x2 = max(0, min(x2, ancho))
    y1 = max(0, min(y1, alto))
    y2 = max(0, min(y2, alto))

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    original = imagen[y1:y2, x1:x2].copy()

    segmentado = None

    if (
        mascara is not None
        and mascara.shape == (alto, ancho)
    ):
        zona = mascara[y1:y2, x1:x2]

        if cv2.countNonZero(zona) > 0:
            segmentado = original.copy()

            segmentado[zona == 0] = (
                255, 255, 255
            )

    return (
        original,
        segmentado,
        (x1, y1, x2, y2)
    )


# ==========================================
# CLASIFICAR CON CLIP
# ==========================================

def clasificar_clip(ia, imagen):
    rgb = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2RGB
    )

    buffer = io.BytesIO()

    Image.fromarray(rgb).save(
        buffer,
        format="PNG"
    )

    return ia.clasificar(
        buffer.getvalue(),
        detalles=True
    )


def porcentaje_valido(valor):
    try:
        porcentaje = float(
            str(valor)
            .replace("%", "")
            .replace(",", ".")
        )

        if 0 <= porcentaje <= 100:
            return porcentaje

    except (TypeError, ValueError):
        pass

    return None


def extraer_prediccion(resultado):
    if not isinstance(resultado, dict):
        return None

    if resultado.get("error"):
        return None

    prediccion = resultado.get(
        "prediccion_modelo", {}
    )

    if not isinstance(prediccion, dict):
        prediccion = {}

    categoria = (
        resultado.get("categoria_modelo")
        or prediccion.get("categoria")
        or prediccion.get("clase")
        or prediccion.get("resultado")
        or resultado.get("resultado")
    )

    categoria = str(
        categoria or ""
    ).strip().upper()

    if categoria not in CATEGORIAS:
        return None

    porcentaje = resultado.get(
        "puntuacion_modelo_pct"
    )

    if porcentaje is None:
        porcentaje = prediccion.get(
            "puntuacion_pct"
        )

    if porcentaje is None:
        porcentaje = prediccion.get(
            "confianza_pct"
        )

    porcentaje = porcentaje_valido(
        porcentaje
    )

    if porcentaje is None:
        return None

    return {
        "categoria": categoria,
        "porcentaje": porcentaje
    }


def elegir_material(ia, original, segmentado):
    diagnosticos = {}
    opciones = []

    for nombre, recorte in (
        ("original", original),
        ("segmentado", segmentado)
    ):
        if recorte is None:
            continue

        try:
            diagnostico = clasificar_clip(
                ia,
                recorte
            )

            diagnosticos[nombre] = diagnostico

            prediccion = extraer_prediccion(
                diagnostico
            )

            if prediccion is not None:
                opciones.append({
                    "categoria": prediccion["categoria"],
                    "porcentaje": prediccion["porcentaje"],
                    "recorte": nombre
                })

        except Exception as error:
            diagnosticos[nombre] = {
                "error": str(error)
            }

            log(
                f"[CLIP] Error en {nombre}: {error}"
            )

    if not opciones:
        return (
            "SIN_PREDICCION",
            None,
            diagnosticos
        )

    mejor = max(
        opciones,
        key=lambda o: o["porcentaje"]
    )

    etiqueta = (
        f"{mejor['categoria']} "
        f"{mejor['porcentaje']:.2f}%"
    )

    return (
        etiqueta,
        mejor,
        diagnosticos
    )


# ==========================================
# EASY OCR
# ==========================================

def analizar_ocr(imagen):
    global LECTOR_OCR

    try:
        import easyocr

        if LECTOR_OCR is None:
            log("[OCR] Cargando lector...")

            LECTOR_OCR = easyocr.Reader(
                ["en"],
                gpu=False
            )

        rgb = cv2.cvtColor(
            imagen,
            cv2.COLOR_BGR2RGB
        )

        textos = LECTOR_OCR.readtext(
            rgb,
            detail=0,
            paragraph=False
        )

        texto = " ".join(
            map(str, textos)
        ).upper()

        patrones = {
            "PET": r"\bPET(?:E)?\b",
            "HDPE": r"\b(?:HDPE|PE-HD)\b",
            "PVC": r"\bPVC\b",
            "LDPE": r"\b(?:LDPE|PE-LD)\b",
            "PP": r"\bPP\b",
            "PS": r"\bPS\b",
            "PC": r"\bPC\b",
            "PUR": r"\bPUR\b",
            "ABS": r"\bABS\b"
        }

        indicios = [
            material
            for material, patron in patrones.items()
            if re.search(patron, texto)
        ]

        return {
            "estado": "analizado",
            "texto": texto,
            "indicios": indicios
        }

    except Exception as error:
        return {
            "estado": "error",
            "detalle": str(error),
            "indicios": []
        }


# ==========================================
# DIBUJAR RESULTADO
# ==========================================

def dibujar_objeto(
    imagen,
    caja,
    mascara,
    etiqueta
):
    alto, ancho = imagen.shape[:2]

    x1, y1, x2, y2 = caja

    if (
        mascara is not None
        and mascara.shape == (alto, ancho)
    ):
        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            imagen,
            contornos,
            -1,
            (0, 255, 0),
            2
        )

    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        imagen,
        etiqueta,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 0),
        2
    )


# ==========================================
# CONVERTIR JSON
# ==========================================

def convertir_json(valor):
    if isinstance(valor, Path):
        return str(valor)

    if isinstance(valor, np.ndarray):
        return valor.tolist()

    if isinstance(valor, np.generic):
        return valor.item()

    raise TypeError(
        f"No serializable: {type(valor)}"
    )


# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deteccion automatica y clasificacion "
            "visual de materiales plasticos"
        )
    )

    parser.add_argument("--imagen")

    parser.add_argument(
        "--yolo",
        default="yolo26n-seg.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--max-objetos",
        type=int,
        default=20
    )

    parser.add_argument(
        "--area-min",
        type=float,
        default=0.025,
        help="Proporcion minima del area de la imagen"
    )

    parser.add_argument(
        "--sam",
        action="store_true"
    )

    parser.add_argument(
        "--ocr",
        action="store_true"
    )

    parser.add_argument(
        "--todos",
        action="store_true"
    )

    parser.add_argument(
        "--sin-respaldo",
        action="store_true"
    )

    args = parser.parse_args()

    if not 0 < args.conf <= 1:
        parser.error(
            "--conf debe estar entre 0 y 1."
        )

    if args.max_objetos < 1:
        parser.error(
            "--max-objetos debe ser mayor que cero."
        )

    if not 0 <= args.area_min < 1:
        parser.error(
            "--area-min debe estar entre 0 y 1."
        )

    log("=" * 55)
    log("DETECTOR AUTOMATICO DE PLASTICOS VADO")
    log("=" * 55)

    ruta = args.imagen or seleccionar_imagen()

    if not ruta:
        log("[FIN] Seleccion cancelada.")
        return

    ruta = Path(ruta).resolve()

    imagen = leer_imagen(ruta)

    alto, ancho = imagen.shape[:2]

    log(f"[IMAGEN] {ruta.name}")

    # ======================================
    # DETECCION AUTOMATICA
    # ======================================

    objetos = detectar_objetos(
        imagen=imagen,
        pesos=args.yolo,
        confianza=args.conf,
        max_objetos=args.max_objetos,
        todos=args.todos,
        area_minima=args.area_min
    )

    if not objetos:
        log("[YOLO] Sin detecciones validas.")

        if args.sin_respaldo:
            log("[FIN] No se clasifico la imagen.")
            return

        log(
            "[RESPALDO] Analizando foto completa. "
            "Esto NO significa que YOLO haya "
            "detectado un objeto."
        )

        objetos = [{
            "caja": (0, 0, ancho, alto),
            "nombre": "imagen_completa_sin_deteccion",
            "confianza": None,
            "proporcion_imagen": 1.0,
            "mascara": None
        }]

    # ======================================
    # CARGAR CLASIFICADOR
    # ======================================

    log("[CLIP] Cargando modelo y memoria...")

    from ia_plasticos import IAPlasticos

    ia = IAPlasticos()

    log("[CLIP] Listo.")

    sam = (
        cargar_sam()
        if args.sam
        else None
    )

    # ======================================
    # CREAR CARPETA
    # ======================================

    carpeta = (
        SALIDAS
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    carpeta.mkdir(
        parents=True,
        exist_ok=True
    )

    dibujo = imagen.copy()
    registros = []

    # ======================================
    # ANALIZAR CADA OBJETO
    # ======================================

    for indice, objeto in enumerate(
        objetos,
        1
    ):
        log("")
        log("-" * 55)

        log(
            f"[OBJETO {indice}] {objeto['nombre']}"
        )

        caja = objeto["caja"]
        mascara = objeto["mascara"]

        es_respaldo = (
            objeto["nombre"]
            == "imagen_completa_sin_deteccion"
        )

        if sam is not None and not es_respaldo:
            refinada = refinar_sam(
                sam,
                imagen,
                caja
            )

            if refinada is not None:
                mascara = refinada

        (
            original,
            segmentado,
            caja_final
        ) = preparar_recortes(
            imagen,
            caja,
            mascara
        )

        if original is None:
            continue

        # Guardar recorte original.
        ruta_original = (
            carpeta
            / f"objeto_{indice}_original.png"
        )

        cv2.imwrite(
            str(ruta_original),
            original
        )

        ruta_segmentado = None

        if segmentado is not None:
            ruta_segmentado = (
                carpeta
                / f"objeto_{indice}_segmentado.png"
            )

            cv2.imwrite(
                str(ruta_segmentado),
                segmentado
            )

        # Clasificar recortes.
        (
            etiqueta,
            mejor,
            diagnosticos
        ) = elegir_material(
            ia,
            original,
            segmentado
        )

        log(f"[CLIP] {etiqueta}")

        # OCR sobre recorte original.
        ocr = (
            analizar_ocr(original)
            if args.ocr
            else {"estado": "desactivado"}
        )

        # Dibujar objeto.
        if not es_respaldo:
            dibujar_objeto(
                dibujo,
                caja_final,
                mascara,
                etiqueta
            )
        else:
            cv2.putText(
                dibujo,
                etiqueta,
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        registros.append({
            "objeto": indice,
            "tipo_detectado_yolo": objeto["nombre"],
            "deteccion_real_yolo": not es_respaldo,
            "confianza_yolo": objeto["confianza"],
            "caja": list(caja_final),
            "material_predicho": (
                mejor["categoria"]
                if mejor
                else None
            ),
            "puntuacion_modelo_pct": (
                mejor["porcentaje"]
                if mejor
                else None
            ),
            "resultado": etiqueta,
            "mejor_recorte": (
                mejor["recorte"]
                if mejor
                else None
            ),
            "diagnosticos": diagnosticos,
            "ocr": ocr,
            "archivo_original": str(ruta_original),
            "archivo_segmentado": (
                str(ruta_segmentado)
                if ruta_segmentado
                else None
            )
        })

    # ======================================
    # GUARDAR IMAGEN Y JSON
    # ======================================

    resultado_imagen = (
        carpeta / "resultado.jpg"
    )

    cv2.imwrite(
        str(resultado_imagen),
        dibujo
    )

    resultado_json = (
        carpeta / "resultado.json"
    )

    resultado_json.write_text(
        json.dumps(
            {
                "imagen": str(ruta),
                "objetos": registros,
                "advertencia": (
                    "YOLO identifica objetos, no "
                    "materiales. El clasificador CLIP "
                    "reporta puntuaciones visuales que "
                    "no certifican la composicion."
                )
            },
            ensure_ascii=False,
            indent=2,
            default=convertir_json
        ),
        encoding="utf-8"
    )

    log("")
    log("[FIN] Analisis completo.")
    log(f"[FIN] Imagen: {resultado_imagen}")
    log(f"[FIN] JSON: {resultado_json}")

    # ======================================
    # MOSTRAR RESULTADO
    # ======================================

    escala = min(
        1.0,
        1100 / ancho,
        750 / alto
    )

    vista = cv2.resize(
        dibujo,
        None,
        fx=escala,
        fy=escala
    )

    cv2.imshow(
        "Detector Automatico de Plasticos Vado",
        vista
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        log("[FIN] Programa interrumpido.")
        sys.exit(130)

    except Exception:
        traceback.print_exc()
        sys.exit(1)
