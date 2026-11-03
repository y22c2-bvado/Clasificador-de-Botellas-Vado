
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
SALIDA = BASE / "resultados" / "detecciones_yolo"
SALIDA.mkdir(parents=True, exist_ok=True)

# Usa un detector propio si existe.
# De lo contrario, usa YOLO preentrenado.
PERSONALIZADO = BASE / "models" / "best.pt"
PESOS = (
    str(PERSONALIZADO)
    if PERSONALIZADO.exists()
    else "yolo26n.pt"
)

modelo = YOLO(PESOS)

ventana = Tk()
ventana.withdraw()

archivo = filedialog.askopenfilename(
    title="Seleccionar imagen para YOLO",
    filetypes=[
        ("Imagenes", "*.jpg *.jpeg *.png *.webp")
    ]
)

ventana.destroy()

if not archivo:
    print("No seleccionaste ninguna imagen.")
    raise SystemExit

resultados = modelo.predict(
    source=archivo,
    conf=0.25,
    device="cpu",
    verbose=False
)

imagen = cv2.imread(archivo)

if imagen is None:
    raise SystemExit("No se pudo abrir la imagen.")

detecciones = 0

for resultado in resultados:
    for caja in resultado.boxes:
        x1, y1, x2, y2 = map(
            int, caja.xyxy[0].tolist()
        )

        clase = int(caja.cls[0])
        confianza = float(caja.conf[0])
        nombre = modelo.names[clase]

        cv2.rectangle(
            imagen,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            imagen,
            f"{nombre} {confianza:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        detecciones += 1

destino = SALIDA / (
    Path(archivo).stem + "_yolo.jpg"
)

cv2.imwrite(str(destino), imagen)

print(f"Objetos detectados: {detecciones}")
print(f"Resultado guardado en: {destino}")

cv2.imshow("YOLO - Deteccion de objetos", imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
