"""Fine-tuning de la capa clasificadora de CLIP, partiendo del modelo Colab.

Se conservan los pesos CLIP originales. Se expande la cabeza para clases nuevas,
copiando sus filas antiguas según nombre; sin datos de las seis categorías
originales NO se realiza entrenamiento para evitar olvido catastrófico.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

try:
    from .config import (DATOS, EXTENSIONES_IMAGEN, MODELO_MEJORADO, MODELO_ORIGINAL, RESULTADOS, CATEGORIAS_OBJETIVO)
    from .modelo import ClasificadorPlasticos
except ImportError:
    from config import (DATOS, EXTENSIONES_IMAGEN, MODELO_MEJORADO, MODELO_ORIGINAL, RESULTADOS, CATEGORIAS_OBJETIVO)
    from modelo import ClasificadorPlasticos


def listar_datos(categorias, carpeta):
    """Evita usar exactamente la misma imagen en distintos conjuntos/clases."""
    archivos, etiquetas = [], []
    hashes = {}
    for i, categoria in enumerate(categorias):
        directorio = carpeta / categoria
        imagenes = sorted(p for p in directorio.rglob("*")
                          if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN) \
            if directorio.is_dir() else []
        for archivo in imagenes:
            try:
                # Hash del contenido para deduplicar archivos incluso renombrados.
                huella = hashlib.sha256(archivo.read_bytes()).hexdigest()
                if huella in hashes:
                    clase_previa, ruta_previa = hashes[huella]
                    if clase_previa != categoria:
                        raise ValueError(
                            f"La misma foto tiene dos etiquetas: {ruta_previa} "
                            f"({clase_previa}) y {archivo} ({categoria})."
                        )
                    continue
                # Abre y valida sin cargar permanentemente toda la imagen.
                with ClasificadorPlasticos.abrir_imagen(archivo) as _:
                    pass
                hashes[huella] = (categoria, archivo)
                archivos.append(archivo)
                etiquetas.append(i)
            except ValueError:
                raise
            except Exception as exc:
                print(f"AVISO: archivo omitido {archivo}: {exc}")
    return archivos, etiquetas


def extender_cabeza(cabeza_anterior, clases_anteriores, clases_nuevas, dimension, device):
    """Copia por nombre exactamente las filas originales y crea filas adicionales."""
    ampliada = nn.Linear(dimension, len(clases_nuevas)).to(device)
    with torch.no_grad():
        for i, categoria in enumerate(clases_nuevas):
            if categoria in clases_anteriores:
                anterior = clases_anteriores.index(categoria)
                ampliada.weight[i].copy_(cabeza_anterior.weight[anterior])
                ampliada.bias[i].copy_(cabeza_anterior.bias[anterior])
    return ampliada


def obtener_vectores(modelo, archivos, etiquetas):
    """Cachea features: CLIP no se ajusta en este entrenamiento (solo la cabeza)."""
    vectores = []
    print(f"Extrayendo características de {len(archivos)} fotografías...")
    for posicion, ruta in enumerate(archivos, start=1):
        vector = modelo.vectorizar(ruta)
        vectores.append(vector.cpu().squeeze(0))
        if posicion % 50 == 0 or posicion == len(archivos):
            print(f"  {posicion}/{len(archivos)}")
    return torch.stack(vectores), torch.tensor(etiquetas, dtype=torch.long)


def entrenar(args):
    random.seed(args.semilla)
    np.random.seed(args.semilla)
    torch.manual_seed(args.semilla)
    # Por defecto siempre conserva la fuente original intacta.
    fuente = args.base if args.base else MODELO_ORIGINAL
    modelo = ClasificadorPlasticos(fuente)
    clases_anteriores = list(modelo.clases)
    # Cada carpeta nueva es una categoría, sin cambios de código.
    agregadas = sorted(d.name.upper() for d in args.datos.iterdir() if d.is_dir()) \
        if args.datos.exists() else []
    # Incluir siempre las clases pactadas, aunque todavía no haya fotografías.
    # Sin suficientes datos de PC, PUR, ABS y OTRO NO fabricamos un checkpoint
    # aparentemente entrenado con filas aleatorias.
    clases = clases_anteriores + [c for c in CATEGORIAS_OBJETIVO if c not in clases_anteriores]
    clases += [c for c in agregadas if c not in clases]
    if len(clases) < 2:
        raise ValueError("Se necesitan al menos dos categorías.")

    # Requiere imágenes de todas las clases que seguirá prediciendo.
    archivos, etiquetas = listar_datos(clases, args.datos)
    contadores = Counter(etiquetas)
    print("Fotos por categoría:")
    for indice, nombre in enumerate(clases):
        print(f"  {nombre}: {contadores[indice]}")
    insuficientes = [clases[i] for i in range(len(clases))
                     if contadores[i] < args.minimo]
    if insuficientes:
        raise RuntimeError(
            f"Faltan fotos verificadas (mínimo {args.minimo} por clase): "
            + ", ".join(insuficientes)
            + ". Recomendado: 30 a 100+ imágenes variadas por categoría."
        )
    if args.salida.resolve() == fuente.resolve():
        raise ValueError("No sobreescribas el checkpoint de entrada.")

    indices = list(range(len(archivos)))
    train_ids, val_ids = train_test_split(
        indices, test_size=args.validacion, stratify=etiquetas,
        random_state=args.semilla
    )
    print(f"Entrenamiento: {len(train_ids)}; validación: {len(val_ids)}")

    # No hay backpropagation al encoder: se reutilizan sus pesos ya aprendidos.
    for parametro in modelo.clip.parameters():
        parametro.requires_grad_(False)
    modelo.clip.eval()
    dimension = modelo.cabeza.in_features
    cabeza = extender_cabeza(modelo.cabeza, clases_anteriores, clases,
                             dimension, modelo.dispositivo)

    # Procesamiento vectorial sobre imágenes para reutilizar CLIP sin reevaluarlo cada época.
    X_train_cpu, y_train_cpu = obtener_vectores(
        modelo, [archivos[i] for i in train_ids], [etiquetas[i] for i in train_ids]
    )
    X_val_cpu, y_val_cpu = obtener_vectores(
        modelo, [archivos[i] for i in val_ids], [etiquetas[i] for i in val_ids]
    )
    X_val = X_val_cpu.to(modelo.dispositivo)
    y_val = y_val_cpu.to(modelo.dispositivo)
    frecuencia = torch.bincount(y_train_cpu, minlength=len(clases)).float()
    peso_muestra = (1.0 / frecuencia[y_train_cpu].clamp(min=1)).double()
    sampler = WeightedRandomSampler(peso_muestra, num_samples=len(y_train_cpu), replacement=True)
    train_loader = DataLoader(TensorDataset(X_train_cpu, y_train_cpu),
                              batch_size=args.lote, sampler=sampler)

    opt = torch.optim.AdamW(cabeza.parameters(), lr=args.lr, weight_decay=0.01)
    mejor_f1 = -1.0
    mejor_accuracy = -1.0
    mejor_estado = None
    sin_mejora = 0
    historial = []
    for epoca in range(1, args.epocas + 1):
        cabeza.train()
        perdidas = []
        for X, y in train_loader:
            X = X.to(modelo.dispositivo)
            y = y.to(modelo.dispositivo)
            opt.zero_grad()
            perdida = F.cross_entropy(cabeza(X), y)
            perdida.backward()
            opt.step()
            perdidas.append(float(perdida.item()))
        cabeza.eval()
        with torch.inference_mode():
            pred = cabeza(X_val).argmax(dim=1)
        real = y_val.cpu().numpy()
        estimado = pred.cpu().numpy()
        accuracy = float(accuracy_score(real, estimado))
        macro_f1 = float(f1_score(real, estimado, labels=list(range(len(clases))),
                                  average="macro", zero_division=0))
        historial.append({"epoca": epoca, "accuracy": accuracy,
                          "f1_macro": macro_f1,
                          "perdida": round(float(np.mean(perdidas)), 6)})
        print(f"Época {epoca:02d}/{args.epocas}: accuracy={accuracy:.3f} "
              f"F1-macro={macro_f1:.3f}")
        if macro_f1 > mejor_f1 + 1e-7 or (
            abs(macro_f1 - mejor_f1) <= 1e-7 and accuracy > mejor_accuracy
        ):
            mejor_f1, mejor_accuracy = macro_f1, accuracy
            mejor_estado = copy.deepcopy(cabeza.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print("Detención temprana por falta de mejora.")
                break
    cabeza.load_state_dict(mejor_estado)
    cabeza.eval()
    with torch.inference_mode():
        pred_final = cabeza(X_val).argmax(dim=1).cpu().numpy()

    informe = classification_report(y_val_cpu.numpy(), pred_final,
                                    labels=list(range(len(clases))),
                                    target_names=clases, zero_division=0,
                                    output_dict=True)
    matriz = confusion_matrix(y_val_cpu.numpy(), pred_final,
                              labels=list(range(len(clases)))).tolist()
    # Conservar la estructura original del .pth, y la cabeza reajustada.
    anterior = torch.load(fuente, map_location="cpu", weights_only=True)
    nuevo = {
        "modelo_base": anterior.get("modelo_base", "openai/clip-vit-base-patch32"),
        "clases": clases,
        "dimension": dimension,
        "pesos_clip": anterior.get("pesos_clip", {}),
        "cabeza": {k: v.detach().cpu().clone() for k, v in mejor_estado.items()},
        "precision_validacion": mejor_accuracy,
        "imagenes_entrenamiento": len(train_ids),
        "imagenes_validacion": len(val_ids),
        "version": "api-zip-finetuning-pc-pur-abs-otro-2",
        "umbral_sugerido": 0.55,
        "margen_sugerido": 0.15,
    }
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save(nuevo, args.salida)
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    with (RESULTADOS / "historial_entrenamiento.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoca", "accuracy", "f1_macro", "perdida"])
        writer.writeheader()
        writer.writerows(historial)
    reporte = {
        "categorias": clases,
        "accuracy": mejor_accuracy,
        "f1_macro": mejor_f1,
        "reporte_por_clase": informe,
        "matriz_confusion": matriz,
        "notas": [
            "El resultado procede de una división de las fotos suministradas; no es prueba de rendimiento en campo.",
            "Los scores softmax no están calibrados como probabilidades reales.",
            "La fotografía sola no verifica composición química de un plástico.",
        ],
    }
    (RESULTADOS / "informe_validacion.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nREENTRENAMIENTO FINALIZADO")
    print(f"Checkpoint NUEVO: {args.salida}")
    print(f"Precisión validación: {mejor_accuracy*100:.2f}%")
    print(f"F1 macro: {mejor_f1*100:.2f}%")
    print("Original sin modificar.")


def main():
    parser = argparse.ArgumentParser(description="Reentrena CLIP con clases nuevas")
    parser.add_argument("--datos", type=Path, default=DATOS)
    parser.add_argument("--base", type=Path, default=None,
                        help="Usa un .pth específico; por defecto el original de Colab")
    parser.add_argument("--salida", type=Path, default=MODELO_MEJORADO)
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--lote", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--validacion", type=float, default=0.2)
    parser.add_argument("--paciencia", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--minimo", type=int, default=10)
    args = parser.parse_args()
    if not (0 < args.validacion < 0.5):
        parser.error("--validacion debe estar entre 0 y 0.5")
    if min(args.epocas, args.lote, args.paciencia, args.minimo) < 1:
        parser.error("Epocas, lote, paciencia y minimo deben ser positivos")
    entrenar(args)


if __name__ == "__main__":
    main()
