
"""
Reentrena la cabeza clasificadora de CLIP sin modificar el encoder.

Mejoras:
- Amplia categorias y conserva los pesos anteriores.
- Detecta imagenes duplicadas exactas.
- Divide entrenamiento, validacion y prueba.
- Balancea categorias mediante muestreo ponderado.
- Utiliza regularizacion y detencion temprana.
- Calcula temperatura de calibracion.
- Genera reportes detallados de rendimiento.
- Conserva el checkpoint original.

IMPORTANTE:
Las imagenes muy parecidas del mismo objeto deben separarse
manualmente para evitar contaminar las pruebas.
"""

import argparse
import copy
import csv
import hashlib
import json
import random

from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import train_test_split

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
)

try:
    from .config import (
        DATOS,
        EXTENSIONES_IMAGEN,
        MODELO_MEJORADO,
        MODELO_ORIGINAL,
        RESULTADOS,
        CATEGORIAS_OBJETIVO,
    )
    from .modelo import ClasificadorPlasticos

except ImportError:
    from config import (
        DATOS,
        EXTENSIONES_IMAGEN,
        MODELO_MEJORADO,
        MODELO_ORIGINAL,
        RESULTADOS,
        CATEGORIAS_OBJETIVO,
    )
    from modelo import ClasificadorPlasticos


# ==========================================================
# CARGAR IMAGENES Y EVITAR DUPLICADOS
# ==========================================================

def listar_datos(categorias, carpeta):

    archivos = []
    etiquetas = []
    hashes = {}

    extensiones = {
        str(ext).lower()
        for ext in EXTENSIONES_IMAGEN
    }

    for i, categoria in enumerate(categorias):

        directorio = carpeta / categoria

        if not directorio.is_dir():
            continue

        imagenes = sorted(
            directorio.rglob("*")
        )

        for archivo in imagenes:

            if not archivo.is_file():
                continue

            if archivo.suffix.lower() not in extensiones:
                continue

            try:

                huella = hashlib.sha256(
                    archivo.read_bytes()
                ).hexdigest()

                if huella in hashes:

                    anterior, ruta = hashes[huella]

                    if anterior != categoria:

                        raise ValueError(
                            "Imagen identica con etiquetas distintas: "
                            f"{ruta} ({anterior}) y "
                            f"{archivo} ({categoria})"
                        )

                    continue

                # Comprobar que Python puede abrir la imagen.
                with ClasificadorPlasticos.abrir_imagen(
                    archivo
                ) as imagen:
                    imagen.load()

                hashes[huella] = (
                    categoria,
                    archivo
                )

                archivos.append(archivo)
                etiquetas.append(i)

            except ValueError:
                raise

            except Exception as error:

                print(
                    f"AVISO: imagen omitida {archivo}: {error}",
                    flush=True
                )

    return (
        archivos,
        np.asarray(etiquetas, dtype=np.int64)
    )


# ==========================================================
# AMPLIAR CLASES DEL MODELO
# ==========================================================

def extender_cabeza(
    anterior,
    antiguas,
    nuevas,
    dimension,
    dispositivo
):

    cabeza = nn.Linear(
        dimension,
        len(nuevas)
    ).to(dispositivo)

    with torch.no_grad():

        for i, categoria in enumerate(nuevas):

            if categoria in antiguas:

                anterior_id = antiguas.index(
                    categoria
                )

                cabeza.weight[i].copy_(
                    anterior.weight[anterior_id]
                )

                cabeza.bias[i].copy_(
                    anterior.bias[anterior_id]
                )

    return cabeza


# ==========================================================
# EXTRAER CARACTERISTICAS DE CLIP
# ==========================================================

def obtener_vectores(
    modelo,
    archivos,
    etiquetas
):

    vectores = []

    print(
        f"Extrayendo caracteristicas de "
        f"{len(archivos)} imagenes...",
        flush=True
    )

    with torch.inference_mode():

        for posicion, ruta in enumerate(
            archivos,
            start=1
        ):

            vector = modelo.vectorizar(
                ruta
            ).detach().cpu().float().flatten()

            if not bool(
                torch.isfinite(vector).all()
            ):

                raise ValueError(
                    f"Embedding no valido: {ruta}"
                )

            vectores.append(vector)

            if (
                posicion % 50 == 0
                or posicion == len(archivos)
            ):

                print(
                    f"  {posicion}/{len(archivos)}",
                    flush=True
                )

    return (
        torch.stack(vectores),
        torch.as_tensor(
            etiquetas,
            dtype=torch.long
        )
    )


# ==========================================================
# EVALUAR RENDIMIENTO
# ==========================================================

def evaluar(
    cabeza,
    X,
    y,
    clases,
    dispositivo,
    temperatura=1.0
):

    cabeza.eval()

    with torch.inference_mode():

        logits = cabeza(
            X.to(dispositivo)
        ).cpu()

        logits = logits / temperatura

        predicciones = logits.argmax(
            dim=1
        ).numpy()

        probabilidades = F.softmax(
            logits,
            dim=1
        )

    reales = y.numpy()

    accuracy = accuracy_score(
        reales,
        predicciones
    )

    f1 = f1_score(
        reales,
        predicciones,
        labels=list(range(len(clases))),
        average="macro",
        zero_division=0
    )

    reporte = classification_report(
        reales,
        predicciones,
        labels=list(range(len(clases))),
        target_names=clases,
        zero_division=0,
        output_dict=True
    )

    matriz = confusion_matrix(
        reales,
        predicciones,
        labels=list(range(len(clases)))
    )

    perdida = F.cross_entropy(
        logits,
        y
    ).item()

    confianza_media = (
        probabilidades
        .max(dim=1)
        .values
        .mean()
        .item()
    )

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1),
        "perdida_logaritmica": float(perdida),
        "confianza_media_maxima": float(
            confianza_media
        ),
        "reporte_por_clase": reporte,
        "matriz_confusion": matriz.tolist(),
    }


# ==========================================================
# CALIBRAR PUNTUACIONES
# ==========================================================

def ajustar_temperatura(
    cabeza,
    X_val,
    y_val,
    dispositivo
):

    cabeza.eval()

    with torch.inference_mode():

        logits = cabeza(
            X_val.to(dispositivo)
        ).cpu()

    mejor_temperatura = 1.0
    mejor_perdida = float("inf")

    temperaturas = np.geomspace(
        0.5,
        10.0,
        100
    )

    for temperatura in temperaturas:

        perdida = F.cross_entropy(
            logits / float(temperatura),
            y_val
        ).item()

        if perdida < mejor_perdida:

            mejor_perdida = perdida

            mejor_temperatura = float(
                temperatura
            )

    return mejor_temperatura


# ==========================================================
# ENTRENAMIENTO PRINCIPAL
# ==========================================================

def entrenar(args):

    random.seed(args.semilla)
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)

    fuente = (
        Path(args.base)
        if args.base
        else Path(MODELO_ORIGINAL)
    )

    if not fuente.exists():
        raise FileNotFoundError(
            f"No existe el modelo base: {fuente}"
        )

    if not args.datos.is_dir():
        raise FileNotFoundError(
            f"No existe Dataset: {args.datos}"
        )

    if args.salida.resolve() == fuente.resolve():
        raise ValueError(
            "No puedes sobrescribir el modelo base."
        )

    # ------------------------------------------------------
    # CARGAR CLASIFICADOR ORIGINAL
    # ------------------------------------------------------

    modelo = ClasificadorPlasticos(
        fuente
    )

    clases_anteriores = list(
        modelo.clases
    )

    nuevas_carpetas = sorted(
        d.name
        for d in args.datos.iterdir()
        if d.is_dir()
    )

    clases = list(dict.fromkeys(
        clases_anteriores
        + list(CATEGORIAS_OBJETIVO)
        + nuevas_carpetas
    ))

    if len(clases) < 2:
        raise ValueError(
            "Se necesitan al menos dos categorias"
        )

    # ------------------------------------------------------
    # VALIDAR DATASET
    # ------------------------------------------------------

    archivos, etiquetas = listar_datos(
        clases,
        args.datos
    )

    conteo = Counter(
        etiquetas.tolist()
    )

    print(
        "\nIMAGENES VALIDAS POR CATEGORIA",
        flush=True
    )

    for i, categoria in enumerate(clases):

        print(
            f"{categoria}: {conteo[i]}",
            flush=True
        )

    faltantes = [
        categoria
        for i, categoria in enumerate(clases)
        if conteo[i] < args.minimo
    ]

    if faltantes:

        raise ValueError(
            f"Categorias con menos de "
            f"{args.minimo} imagenes: {faltantes}"
        )

    # ------------------------------------------------------
    # DIVIDIR DATOS
    # ------------------------------------------------------

    indices = np.arange(
        len(archivos)
    )

    desarrollo_ids, prueba_ids = train_test_split(
        indices,
        test_size=args.prueba,
        stratify=etiquetas,
        random_state=args.semilla
    )

    entrenamiento_ids, validacion_ids = train_test_split(
        desarrollo_ids,
        test_size=(
            args.validacion / (1 - args.prueba)
        ),
        stratify=etiquetas[desarrollo_ids],
        random_state=args.semilla
    )

    print(
        "\nDISTRIBUCION DEL DATASET",
        flush=True
    )

    print(
        f"Entrenamiento: {len(entrenamiento_ids)}",
        flush=True
    )

    print(
        f"Validacion: {len(validacion_ids)}",
        flush=True
    )

    print(
        f"Prueba independiente: {len(prueba_ids)}",
        flush=True
    )

    # ------------------------------------------------------
    # CONGELAR CLIP
    # ------------------------------------------------------

    for parametro in modelo.clip.parameters():
        parametro.requires_grad_(False)

    modelo.clip.eval()

    dimension = modelo.cabeza.in_features

    cabeza = extender_cabeza(
        modelo.cabeza,
        clases_anteriores,
        clases,
        dimension,
        modelo.dispositivo
    )

    # ------------------------------------------------------
    # EXTRAER VECTORES
    # ------------------------------------------------------

    X_train, y_train = obtener_vectores(
        modelo,
        [
            archivos[i]
            for i in entrenamiento_ids
        ],
        etiquetas[entrenamiento_ids]
    )

    X_val, y_val = obtener_vectores(
        modelo,
        [
            archivos[i]
            for i in validacion_ids
        ],
        etiquetas[validacion_ids]
    )

    X_test, y_test = obtener_vectores(
        modelo,
        [
            archivos[i]
            for i in prueba_ids
        ],
        etiquetas[prueba_ids]
    )

    # ------------------------------------------------------
    # BALANCEO DE CATEGORIAS
    # ------------------------------------------------------

    frecuencia = torch.bincount(
        y_train,
        minlength=len(clases)
    ).float()

    pesos = (
        1.0 / frecuencia[y_train].clamp(min=1)
    ).double()

    generador = (
        torch.Generator()
        .manual_seed(args.semilla)
    )

    sampler = WeightedRandomSampler(
        pesos,
        num_samples=len(y_train),
        replacement=True,
        generator=generador
    )

    loader = DataLoader(
        TensorDataset(
            X_train,
            y_train
        ),
        batch_size=args.lote,
        sampler=sampler
    )

    # ------------------------------------------------------
    # OPTIMIZADOR
    # ------------------------------------------------------

    optimizador = torch.optim.AdamW(
        cabeza.parameters(),
        lr=args.lr,
        weight_decay=args.decaimiento
    )

    mejor_f1 = -1.0
    mejor_accuracy = -1.0
    mejor_estado = None

    sin_mejora = 0
    historial = []

    # ------------------------------------------------------
    # ENTRENAR
    # ------------------------------------------------------

    for epoca in range(
        1,
        args.epocas + 1
    ):

        cabeza.train()

        perdidas = []

        for X, y in loader:

            X = X.to(
                modelo.dispositivo
            )

            y = y.to(
                modelo.dispositivo
            )

            optimizador.zero_grad(
                set_to_none=True
            )

            logits = cabeza(X)

            perdida = F.cross_entropy(
                logits,
                y,
                label_smoothing=args.suavizado
            )

            perdida.backward()

            torch.nn.utils.clip_grad_norm_(
                cabeza.parameters(),
                max_norm=5.0
            )

            optimizador.step()

            perdidas.append(
                float(perdida.item())
            )

        # --------------------------------------------------
        # VALIDAR
        # --------------------------------------------------

        metricas = evaluar(
            cabeza,
            X_val,
            y_val,
            clases,
            modelo.dispositivo
        )

        accuracy = metricas["accuracy"]
        f1 = metricas["f1_macro"]

        historial.append({
            "epoca": epoca,
            "perdida_entrenamiento": float(
                np.mean(perdidas)
            ),
            "accuracy_validacion": accuracy,
            "f1_macro_validacion": f1,
        })

        print(
            f"Epoca {epoca:02d}/{args.epocas} "
            f"| Accuracy: {accuracy:.3f} "
            f"| F1: {f1:.3f}",
            flush=True
        )

        # --------------------------------------------------
        # CONSERVAR MEJOR EPOCA
        # --------------------------------------------------

        mejoro = (
            f1 > mejor_f1 + 1e-7
            or (
                abs(f1 - mejor_f1) <= 1e-7
                and accuracy > mejor_accuracy
            )
        )

        if mejoro:

            mejor_f1 = f1
            mejor_accuracy = accuracy

            mejor_estado = copy.deepcopy(
                cabeza.state_dict()
            )

            sin_mejora = 0

        else:

            sin_mejora += 1

            if sin_mejora >= args.paciencia:

                print(
                    "Detencion temprana",
                    flush=True
                )

                break

    # ------------------------------------------------------
    # RESTAURAR MEJOR MODELO
    # ------------------------------------------------------

    cabeza.load_state_dict(
        mejor_estado
    )

    temperatura = ajustar_temperatura(
        cabeza,
        X_val,
        y_val,
        modelo.dispositivo
    )

    metricas_validacion = evaluar(
        cabeza,
        X_val,
        y_val,
        clases,
        modelo.dispositivo
    )

    metricas_prueba = evaluar(
        cabeza,
        X_test,
        y_test,
        clases,
        modelo.dispositivo,
        temperatura=temperatura
    )

    metricas_prueba_sin_calibracion = evaluar(
        cabeza,
        X_test,
        y_test,
        clases,
        modelo.dispositivo
    )

    # ------------------------------------------------------
    # GUARDAR CHECKPOINT
    # ------------------------------------------------------

    anterior = torch.load(
        fuente,
        map_location="cpu",
        weights_only=True
    )

    checkpoint = {
        "modelo_base": anterior.get(
            "modelo_base",
            "openai/clip-vit-base-patch32"
        ),
        "clases": clases,
        "dimension": dimension,
        "pesos_clip": anterior.get(
            "pesos_clip",
            {}
        ),
        "cabeza": {
            k: v.detach().cpu().clone()
            for k, v in mejor_estado.items()
        },
        "precision_validacion": (
            metricas_validacion["accuracy"]
        ),
        "imagenes_entrenamiento": len(
            entrenamiento_ids
        ),
        "imagenes_validacion": len(
            validacion_ids
        ),
        "imagenes_prueba": len(
            prueba_ids
        ),
        "temperatura_calibracion": temperatura,
        "precision_prueba": (
            metricas_prueba["accuracy"]
        ),
        "f1_macro_prueba": (
            metricas_prueba["f1_macro"]
        ),
        "version": (
            "clasificador-clip-"
            "evaluacion-independiente-3"
        ),
        "umbral_sugerido": anterior.get(
            "umbral_sugerido",
            0.55
        ),
        "margen_sugerido": anterior.get(
            "margen_sugerido",
            0.15
        ),
    }

    args.salida.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporal = args.salida.with_suffix(
        ".tmp"
    )

    torch.save(
        checkpoint,
        temporal
    )

    temporal.replace(
        args.salida
    )

    # ------------------------------------------------------
    # GUARDAR REPORTES
    # ------------------------------------------------------

    RESULTADOS.mkdir(
        parents=True,
        exist_ok=True
    )

    with (
        RESULTADOS / "historial_entrenamiento.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(historial[0])
        )

        escritor.writeheader()
        escritor.writerows(historial)

    reporte = {
        "categorias": clases,

        "imagenes_por_categoria": {
            clases[i]: conteo[i]
            for i in range(len(clases))
        },

        "division": {
            "entrenamiento": len(entrenamiento_ids),
            "validacion": len(validacion_ids),
            "prueba": len(prueba_ids),
        },

        "accuracy": metricas_validacion["accuracy"],
        "f1_macro": metricas_validacion["f1_macro"],

        "validacion": metricas_validacion,

        "prueba_independiente": metricas_prueba,

        "prueba_sin_calibracion": (
            metricas_prueba_sin_calibracion
        ),

        "temperatura_calibracion": temperatura,

        "notas": [
            "La prueba no participa en la seleccion de epocas.",
            "La calibracion se guarda como metadato.",
            "modelo.py debe aplicar la temperatura en inferencia.",
            "Evitar imagenes casi identicas entre particiones.",
            "La fotografia no certifica composicion quimica.",
        ],
    }

    (
        RESULTADOS / "informe_validacion.json"
    ).write_text(
        json.dumps(
            reporte,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    # ------------------------------------------------------
    # RESULTADO FINAL
    # ------------------------------------------------------

    print(
        "\nREENTRENAMIENTO FINALIZADO",
        flush=True
    )

    print(
        f"Checkpoint: {args.salida}",
        flush=True
    )

    print(
        "Accuracy validacion: "
        f"{metricas_validacion['accuracy'] * 100:.2f}%",
        flush=True
    )

    print(
        "Accuracy prueba independiente: "
        f"{metricas_prueba['accuracy'] * 100:.2f}%",
        flush=True
    )

    print(
        "F1 macro prueba independiente: "
        f"{metricas_prueba['f1_macro'] * 100:.2f}%",
        flush=True
    )

    print(
        "Temperatura estimada: "
        f"{temperatura:.3f}",
        flush=True
    )

    print(
        "Modelo original conservado.",
        flush=True
    )


# ==========================================================
# ARGUMENTOS DE LA TERMINAL
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Entrena CLIP con categorias nuevas "
            "y evalua mediante prueba independiente"
        )
    )

    parser.add_argument(
        "--datos",
        type=Path,
        default=DATOS
    )

    parser.add_argument(
        "--base",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--salida",
        type=Path,
        default=MODELO_MEJORADO
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=40
    )

    parser.add_argument(
        "--lote",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--validacion",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--prueba",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--paciencia",
        type=int,
        default=10
    )

    parser.add_argument(
        "--semilla",
        type=int,
        default=42
    )

    parser.add_argument(
        "--minimo",
        type=int,
        default=10
    )

    parser.add_argument(
        "--decaimiento",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--suavizado",
        type=float,
        default=0.05
    )

    args = parser.parse_args()

    if not (
        0 < args.validacion < 0.4
        and 0 < args.prueba < 0.4
        and args.validacion + args.prueba < 0.6
    ):
        parser.error(
            "Fracciones de validacion/prueba invalidas"
        )

    if min(
        args.epocas,
        args.lote,
        args.paciencia,
        args.minimo
    ) < 1:
        parser.error(
            "Epocas, lote, paciencia y minimo "
            "deben ser positivos"
        )

    if (
        args.lr <= 0
        or args.decaimiento < 0
        or not 0 <= args.suavizado < 1
    ):
        parser.error(
            "Hiperparametros fuera de rango"
        )

    entrenar(args)


if __name__ == "__main__":
    main()
