
"""
CLASIFICADOR DE BOTELLAS - PRUEBA DE IMAGEN

Permite seleccionar una fotografía desde Windows
y clasificarla utilizando CLIP + memoria visual.

Resultados posibles:
PET, HDPE, PVC, LDPE, PP, PS, PC, PUR, ABS, OTRO.

OTRO incluye materiales fuera de las categorias
y casos donde la evidencia es insuficiente.

Uso:
    python probar_imagen.py

Diagnostico:
    python probar_imagen.py --detalles
"""

import argparse
import sys
from pathlib import Path
from pprint import pprint
from tkinter import Tk, filedialog

from ia_plasticos import IAPlasticos


def seleccionar_imagen():
    """Abre el explorador de archivos de Windows."""

    ventana = Tk()
    ventana.withdraw()
    ventana.attributes("-topmost", True)

    try:
        ruta = filedialog.askopenfilename(
            parent=ventana,
            title="Selecciona una imagen para clasificar",
            filetypes=[
                (
                    "Imagenes",
                    "*.jpg *.jpeg *.png *.webp *.bmp"
                ),
                ("Todos los archivos", "*.*"),
            ],
        )
        return Path(ruta) if ruta else None

    finally:
        ventana.destroy()


def mostrar_resultado(resultado, detalles=False):
    """Muestra una unica categoria principal."""

    categoria = resultado.get("resultado", "OTRO")

    print("\n" + "=" * 55)
    print("           RESULTADO DE CLASIFICACION")
    print("=" * 55)

    print(f"\nMATERIAL: {categoria}")

    print("\n" + "=" * 55)

    if not detalles:
        return

    print("\n--- DIAGNOSTICO DEL MODELO ---")

    print(
        "Categoria sugerida:",
        resultado.get("categoria_modelo")
    )

    print(
        "Puntuacion del modelo:",
        resultado.get("puntuacion_modelo_pct")
    )

    print(
        "Estado del modelo:",
        resultado.get("estado_modelo")
    )

    print(
        "Categoria de referencias:",
        resultado.get("categoria_referencias")
    )

    print(
        "Similitud coseno:",
        resultado.get("similitud_coseno")
    )

    print(
        "Estado de referencias:",
        resultado.get("estado_referencias")
    )

    print(
        "Coincidencia:",
        resultado.get("acuerdo")
    )

    print(
        "Motivo:",
        resultado.get("motivo")
    )

    print("\n--- PREDICCION COMPLETA ---")
    pprint(
        resultado.get("prediccion_modelo"),
        sort_dicts=False
    )

    print("\n--- COMPARACION DE EJEMPLOS ---")
    pprint(
        resultado.get("comparacion_ejemplos"),
        sort_dicts=False
    )

    print(
        "\nNota: la puntuacion del modelo no es una "
        "probabilidad calibrada, salvo que el modelo "
        "de inferencia aplique la calibracion."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prueba fotografias con el clasificador "
            "de botellas."
        )
    )

    parser.add_argument(
        "--detalles",
        action="store_true",
        help="Muestra informacion tecnica de la prediccion."
    )

    parser.add_argument(
        "--imagen",
        type=Path,
        default=None,
        help="Ruta de la imagen sin abrir el explorador."
    )

    args = parser.parse_args()

    print("=" * 55)
    print("       CLASIFICADOR DE BOTELLAS VADO")
    print("=" * 55)

    if args.imagen is not None:
        ruta = args.imagen
    else:
        try:
            ruta = seleccionar_imagen()

        except Exception as error:
            print(
                f"No se pudo abrir el selector: {error}"
            )
            return 1

    if ruta is None:
        print("No seleccionaste ninguna imagen.")
        return 0

    if not ruta.is_file():
        print(f"No existe la imagen: {ruta}")
        return 1

    print(f"\nImagen: {ruta.name}")
    print("Cargando inteligencia artificial...")

    try:
        ia = IAPlasticos()

        with ruta.open("rb") as archivo:
            imagen_bytes = archivo.read()

        print("Analizando imagen...")

        resultado = ia.clasificar(
            imagen_bytes,
            detalles=args.detalles
        )

        mostrar_resultado(
            resultado,
            detalles=args.detalles
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        OSError
    ) as error:
        print(f"\nERROR: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
