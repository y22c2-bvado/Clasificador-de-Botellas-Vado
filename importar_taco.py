
"""
Prepara ejemplos de TACO para revisión manual.

Extrae objetos de vidrio, metal y carton anotados.
NO los incorpora automaticamente a Dataset/OTRO.
NO asigna categorias de polimeros.
"""

import json
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parent
TACO = RAIZ / "TACO"
ANOTACIONES = TACO / "data" / "annotations.json"
DESTINO = RAIZ / "TACO_preparado" / "OTRO"

# Categorias de objetos relativamente claras.
# Los nombres se verifican contra annotations.json.
CATEGORIAS = {
    "glass bottle",
    "scrap metal",
    "drink can",
    "corrugated carton",
}

def main():
    if not ANOTACIONES.is_file():
        raise FileNotFoundError(
            f"No se encontro {ANOTACIONES}"
        )

    with ANOTACIONES.open(
        "r", encoding="utf-8"
    ) as archivo:
        datos = json.load(archivo)

    categorias = {
        c["id"]: c["name"]
        for c in datos["categories"]
    }

    imagenes = {
        i["id"]: i
        for i in datos["images"]
    }

    print("Categorias TACO disponibles:")
    for nombre in sorted(set(categorias.values())):
        print(" -", nombre)

    DESTINO.mkdir(parents=True, exist_ok=True)

    guardadas = 0
    omitidas = 0
    nombres = defaultdict(int)

    for anotacion in datos["annotations"]:
        nombre = categorias.get(
            anotacion["category_id"], ""
        )

        if nombre.casefold() not in CATEGORIAS:
            continue

        info = imagenes.get(anotacion["image_id"])
        if not info:
            omitidas += 1
            continue

        ruta = TACO / "data" / info["file_name"]

        if not ruta.is_file():
            omitidas += 1
            continue

        try:
            with Image.open(ruta) as imagen:
                imagen = ImageOps.exif_transpose(
                    imagen
                ).convert("RGB")

                x, y, ancho, alto = anotacion["bbox"]

                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(imagen.width, int(x + ancho))
                y2 = min(imagen.height, int(y + alto))

                if x2 - x1 < 32 or y2 - y1 < 32:
                    omitidas += 1
                    continue

                recorte = imagen.crop(
                    (x1, y1, x2, y2)
                )

                etiqueta = nombre.lower().replace(
                    " ", "_"
                )
                nombres[etiqueta] += 1

                destino = DESTINO / (
                    f"taco_{etiqueta}_"
                    f"{anotacion['id']}.jpg"
                )

                recorte.save(
                    destino,
                    "JPEG",
                    quality=92
                )
                guardadas += 1

        except (OSError, ValueError) as error:
            omitidas += 1
            print(f"Omitida {ruta}: {error}")

    print("\nIMPORTACION TERMINADA")
    print("Imagenes preparadas:", guardadas)
    print("Omitidas:", omitidas)
    print("Carpeta:", DESTINO)

    print("\nRevisa las imagenes antes de")
    print("incorporarlas a Dataset/OTRO.")


if __name__ == "__main__":
    main()
