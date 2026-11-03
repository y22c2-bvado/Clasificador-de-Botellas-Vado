
"""
REVISOR VISUAL DEL DATASET
Clasificador de Botellas Vado

Permite:
- Ver imagenes sospechosas una por una.
- Consultar etiqueta actual y prediccion CLIP.
- Conservar la etiqueta correcta.
- Cambiar una imagen de categoria.
- Apartar imagenes dudosas.
- Dejar imagenes pendientes.
- Guardar decisiones para continuar despues.

No entrena el modelo automaticamente.
"""

import csv
import json
import shutil
import tkinter as tk

from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox

from PIL import Image, ImageTk, ImageOps


CATEGORIAS = [
    "PET", "HDPE", "PVC", "LDPE", "PP",
    "PS", "PC", "PUR", "ABS", "OTRO"
]

BASE = Path(__file__).resolve().parent
DATASET = BASE / "Dataset"
REVISION = BASE / "revision_dataset"
DESCARTADAS = REVISION / "descartadas_manualmente"
HISTORIAL = REVISION / "decisiones_manuales.jsonl"

REVISION.mkdir(parents=True, exist_ok=True)


def ultimo_informe():
    archivos = sorted(
        REVISION.glob("sospechosas_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not archivos:
        raise FileNotFoundError(
            "No existe un informe sospechosas_*.csv.\n"
            "Ejecuta primero:\n"
            "python verificar_dataset.py --umbral 70"
        )

    return archivos[0]


def leer_informe(ruta):
    with ruta.open(
        "r", encoding="utf-8-sig", newline=""
    ) as archivo:
        filas = list(csv.DictReader(archivo))

    return [
        fila for fila in filas
        if fila.get("archivo", "").strip()
    ]


def leer_decisiones():
    decisiones = {}

    if not HISTORIAL.exists():
        return decisiones

    with HISTORIAL.open(
        "r", encoding="utf-8"
    ) as archivo:
        for linea in archivo:
            try:
                registro = json.loads(linea)
                nombre = registro.get("archivo")

                if nombre:
                    decisiones[nombre] = registro
            except (ValueError, TypeError):
                continue

    return decisiones


def guardar_decision(archivo, accion, categoria=None):
    registro = {
        "fecha": datetime.now().isoformat(
            timespec="seconds"
        ),
        "archivo": archivo,
        "accion": accion,
        "categoria": categoria
    }

    with HISTORIAL.open(
        "a", encoding="utf-8"
    ) as destino:
        destino.write(
            json.dumps(
                registro, ensure_ascii=False
            ) + "\n"
        )


def destino_disponible(ruta):
    if not ruta.exists():
        return ruta

    numero = 1

    while True:
        candidato = ruta.with_name(
            f"{ruta.stem}_{numero}{ruta.suffix}"
        )

        if not candidato.exists():
            return candidato

        numero += 1


class RevisorDataset:

    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title(
            "Revisor de Dataset | Clasificador Vado"
        )
        self.ventana.geometry("1020x850")
        self.ventana.minsize(760, 650)

        self.informe = ultimo_informe()
        self.filas = leer_informe(self.informe)
        self.decisiones = leer_decisiones()

        self.pendientes = [
            fila for fila in self.filas
            if fila["archivo"] not in self.decisiones
        ]

        self.indice = 0
        self.imagen_tk = None

        self.categoria_elegida = tk.StringVar(
            value="OTRO"
        )

        self.crear_interfaz()
        self.mostrar_imagen()

    def crear_interfaz(self):
        encabezado = tk.Frame(
            self.ventana, bg="#172b49"
        )
        encabezado.pack(fill="x")

        tk.Label(
            encabezado,
            text="REVISOR INTELIGENTE DEL DATASET",
            font=("Segoe UI", 19, "bold"),
            fg="white",
            bg="#172b49",
            pady=15
        ).pack()

        self.progreso = tk.Label(
            self.ventana,
            font=("Segoe UI", 12, "bold")
        )
        self.progreso.pack(pady=8)

        self.nombre = tk.Label(
            self.ventana,
            font=("Segoe UI", 12),
            wraplength=850
        )
        self.nombre.pack(pady=5)

        self.panel_imagen = tk.Label(
            self.ventana,
            bg="#ececec",
            width=70,
            height=20
        )
        self.panel_imagen.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        self.informacion = tk.Label(
            self.ventana,
            font=("Segoe UI", 12),
            justify="left",
            wraplength=850
        )
        self.informacion.pack(pady=10)

        seleccion = tk.Frame(self.ventana)
        seleccion.pack(pady=8)

        tk.Label(
            seleccion,
            text="Material confirmado:",
            font=("Segoe UI", 11, "bold")
        ).pack(side="left", padx=10)

        self.combo = ttk.Combobox(
            seleccion,
            textvariable=self.categoria_elegida,
            values=CATEGORIAS,
            state="readonly",
            width=15,
            font=("Segoe UI", 11)
        )
        self.combo.pack(side="left")

        botones = tk.Frame(self.ventana)
        botones.pack(pady=12)

        self.crear_boton(
            botones, "Conservar etiqueta",
            "#238636", self.conservar
        )

        self.crear_boton(
            botones, "Cambiar categoria",
            "#146cc5", self.cambiar
        )

        self.crear_boton(
            botones, "Descartar",
            "#b42318", self.descartar
        )

        self.crear_boton(
            botones, "No se / siguiente",
            "#777777", self.siguiente
        )

        self.estado = tk.Label(
            self.ventana,
            text="",
            font=("Segoe UI", 10),
            fg="#444444",
            wraplength=850
        )
        self.estado.pack(pady=7)

        tk.Label(
            self.ventana,
            text=(
                "Verifica el material con informacion "
                "confiable. CLIP solo sugiere etiquetas."
            ),
            fg="#666666",
            font=("Segoe UI", 9)
        ).pack(pady=8)

    def crear_boton(
        self, padre, texto, color, funcion
    ):
        tk.Button(
            padre,
            text=texto,
            command=funcion,
            bg=color,
            fg="white",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            cursor="hand2"
        ).pack(
            side="left",
            padx=5
        )

    def actual(self):
        if self.indice >= len(self.pendientes):
            return None

        return self.pendientes[self.indice]

    def mostrar_imagen(self):
        fila = self.actual()

        if fila is None:
            self.progreso.config(
                text="REVISION FINALIZADA"
            )
            self.nombre.config(
                text="No quedan imagenes pendientes."
            )
            self.panel_imagen.config(
                image="",
                text="Revision completa"
            )
            self.imagen_tk = None
            self.informacion.config(
                text="Todas las decisiones fueron guardadas."
            )
            self.estado.config(
                text=f"Historial: {HISTORIAL}"
            )
            return

        archivo = fila["archivo"]
        ruta = DATASET / archivo

        self.progreso.config(
            text=(
                f"Imagen {self.indice + 1} "
                f"de {len(self.pendientes)}"
            )
        )

        self.nombre.config(text=archivo)

        actual = fila.get(
            "categoria_actual", ""
        )

        prediccion = fila.get(
            "prediccion_clip", ""
        )

        puntuacion = fila.get(
            "puntuacion_clip_pct", ""
        )

        margen = fila.get(
            "margen_clip_pct", ""
        )

        problema = fila.get(
            "problema", ""
        )

        self.informacion.config(
            text=(
                f"Categoria actual: {actual}\n"
                f"Prediccion CLIP: {prediccion}\n"
                f"Puntuacion CLIP: {puntuacion}%\n"
                f"Margen: {margen}%\n"
                f"Problema: {problema}"
            )
        )

        self.categoria_elegida.set(
            actual if actual in CATEGORIAS else "OTRO"
        )

        self.imagen_tk = None
        self.panel_imagen.config(
            image="", text=""
        )

        if not ruta.is_file():
            self.panel_imagen.config(
                text="Archivo no encontrado"
            )
            self.estado.config(
                text=str(ruta)
            )
            return

        try:
            with Image.open(ruta) as imagen:
                imagen = ImageOps.exif_transpose(
                    imagen
                ).convert("RGB")
                imagen.thumbnail(
                    (850, 460),
                    Image.Resampling.LANCZOS
                )
                copia = imagen.copy()

            self.imagen_tk = ImageTk.PhotoImage(
                copia
            )

            self.panel_imagen.config(
                image=self.imagen_tk,
                text=""
            )

            self.estado.config(
                text="Selecciona una decision."
            )

        except Exception as error:
            self.panel_imagen.config(
                text="No se pudo abrir la imagen"
            )
            self.estado.config(
                text=str(error)
            )

    def avanzar(self):
        self.indice += 1
        self.mostrar_imagen()

    def conservar(self):
        fila = self.actual()

        if fila is None:
            return

        guardar_decision(
            fila["archivo"],
            "conservar",
            fila.get("categoria_actual")
        )

        self.avanzar()

    def cambiar(self):
        fila = self.actual()

        if fila is None:
            return

        categoria = self.categoria_elegida.get()

        if categoria not in CATEGORIAS:
            messagebox.showerror(
                "Error",
                "Selecciona una categoria valida."
            )
            return

        ruta_origen = DATASET / fila["archivo"]

        if not ruta_origen.is_file():
            messagebox.showerror(
                "Error",
                "No se encuentra la imagen original."
            )
            return

        actual = fila.get(
            "categoria_actual", ""
        )

        if categoria == actual:
            self.conservar()
            return

        confirmar = messagebox.askyesno(
            "Confirmar categoria",
            (
                f"¿Confirmaste que el material es "
                f"{categoria}?\n\n"
                "La imagen sera movida a esa carpeta."
            )
        )

        if not confirmar:
            return

        destino = destino_disponible(
            DATASET / categoria / ruta_origen.name
        )

        try:
            destino.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.move(
                str(ruta_origen),
                str(destino)
            )

            guardar_decision(
                fila["archivo"],
                "cambiar",
                categoria
            )

            self.avanzar()

        except OSError as error:
            messagebox.showerror(
                "Error al mover imagen",
                str(error)
            )

    def descartar(self):
        fila = self.actual()

        if fila is None:
            return

        confirmar = messagebox.askyesno(
            "Descartar imagen",
            (
                "¿Queres apartar esta fotografia "
                "del Dataset?\n\n"
                "No se eliminara."
            )
        )

        if not confirmar:
            return

        origen = DATASET / fila["archivo"]

        if not origen.is_file():
            messagebox.showerror(
                "Error",
                "La imagen original no existe."
            )
            return

        destino = destino_disponible(
            DESCARTADAS / fila["archivo"]
        )

        try:
            destino.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.move(
                str(origen),
                str(destino)
            )

            guardar_decision(
                fila["archivo"],
                "descartar"
            )

            self.avanzar()

        except OSError as error:
            messagebox.showerror(
                "Error",
                str(error)
            )

    def siguiente(self):
        # No registra decision; queda pendiente
        # para una proxima revision.
        self.avanzar()


def main():
    ventana = tk.Tk()

    try:
        RevisorDataset(ventana)

    except Exception as error:
        messagebox.showerror(
            "No se puede iniciar",
            str(error)
        )
        ventana.destroy()
        return

    ventana.mainloop()


if __name__ == "__main__":
    main()
