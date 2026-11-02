# IA de plásticos para integrar en tu API existente

**NO incluye ni reemplaza un servidor**, ngrok, web, SQLite, ni cámara. Puedes integrarla en tu API Python actual usando `IAPlasticos`.

## ¿Qué contiene?

- `models/Clasificador_Botellas_Aprendido.pth`: tu checkpoint ORIGINAL de Colab, conserva seis clases **PET, HDPE, PVC, LDPE, PP, PS**.
- `ia_plasticos.py`: adaptador para tu API que recibe bytes de imágenes, ZIPs de ejemplos, y devuelve diccionarios para convertir a JSON.
- `importador.py`: importa categorías automáticamente por nombre de carpeta, sin editarlas manualmente en el código.
- `referencias.py`: compara con fotos etiquetadas (aprendizaje inmediato por ejemplos, con similitudes sin calibrar).
- `entrenamiento.py`: entrena la capa final con todas las clases presentes en `dataset/` y guarda un nuevo checkpoint en `models/Clasificador_Ampliado.pth`.
- `modelo.py`: reproduce el CLIP del notebook original; requiere descargar `openai/clip-vit-base-patch32` la primera vez.

## Instalación (Windows, dentro de la carpeta del módulo)

```powershell
python -m pip install -r requirements.txt
```

## Usarlo desde tu API (Python)

```python
from IA_Plasticos_Para_Tu_API import IAPlasticos

ia = IAPlasticos()  # crear UNA sola vez cuando arranque tu API

# La API recibe los bytes de una foto JPEG o PNG
respuesta = ia.clasificar(foto_bytes)  # dict; retorna JSON desde tu API

# La API recibe los bytes de un ZIP con fotos etiquetadas
resultado = ia.aprender_zip(zip_bytes) # importa imágenes + actualiza referencias

# Reentrenamiento REAL (trabajo lento; hacerlo en tarea externa, NO en la ruta HTTP)
resultado_entrenamiento = ia.reentrenar(minimo=10, epocas=40)
# Esto creará models/Clasificador_Ampliado.pth y recargará la IA.
```

Deja la carpeta `IA_Plasticos_Para_Tu_API` dentro de la carpeta de tu API, junto a `app.py` o `main.py`. El código usa importaciones relativas y encuentra el `.pth` por su propia ruta, sin depender del directorio actual. Si tu API está en otro proyecto, añade la carpeta **padre** al Python path.

## Ejemplo de ZIP

```text
Fotos.zip
├── PET/
│   └── botella_01.jpg
├── HDPE/
│   └── envase_01.jpg
├── PVC/
│   └── tubo_01.jpg
├── LDPE/
│   └── bolsa_01.jpg
├── PP/
│   └── envase_01.jpg
├── PS/
│   └── vaso_01.jpg
├── PC/
│   └── cubierta_01.jpg
├── PUR/
│   └── cateter_01.jpg
├── ABS/
│   └── teclado_01.jpg
└── OTRO/
    └── mano_01.jpg
```

La categoría **OTRO** se debe llenar exclusivamente con fotos de objetos **NO plásticos** (manos, metal, madera, papel, etc.). Un plástico de tipo desconocido NO es OTRO: su respuesta correcta es NO DETERMINADO.

No es obligatorio que todas las clases estén presentes en cada ZIP. Sin embargo, **para reentrenar** sí se requieren suficientes fotos *de cada clase antigua y nueva*, no solo de PC, PUR y ABS; el script bloqueará entrenamientos incompletos para evitar el olvido de las seis clases anteriores. Por defecto mínimo 10 imágenes válidas por clase, preferible >=30-100 variadas y un conjunto de prueba independiente.

Un ZIP de solo PC, PUR, ABS y OTRO sirve para comenzar a comparar fotos, **no** para generar automáticamente un `.pth` que las reconozca con precisión.

## Integración opcional con FastAPI YA EXISTENTE

Dentro de tu servidor actual, agrega tus rutas según tus necesidades; ejemplo de fragmentos, no crea otro servidor:

```python
from fastapi import UploadFile, File
from IA_Plasticos_Para_Tu_API import IAPlasticos
ia = IAPlasticos()

# Dentro de una ruta async def existente:
# datos = await archivo.read()
# return ia.clasificar(datos)

# Y para importar un ZIP:
# datos = await archivo_zip.read()
# return ia.aprender_zip(datos)
```

La ruta de entrenamiento debería ejecutarse en un **worker de fondo controlado** o proceso separado para evitar tiempos de espera y sobrecarga. No expongas carga de ZIPs y entrenamiento sin autenticación o límites.

## Advertencias importantes

- El `.pth` incluido **no está reentrenado** para PC, PUR, ABS ni OTRO. Solo dispone de seis clases. El `.pth` ampliado **se genera al reentrenar con fotografías etiquetadas reales**.
- La memoria por imágenes ofrece similitudes visuales, nunca composición química certificada ni aptitud médica.
- Clasificar una foto de un material flexible o transparente es incierto. Idealmente verifica clase con marca/ficha técnica del artículo.
- La puntuación de CLIP/softmax no es una probabilidad calibrada de seguridad.
- No envíes fotos de pacientes identificables a un servicio público.
- Usa versiones compatibles de transformers con CLIP, y comprueba las rutas de Python del entorno virtual de VS Code.

## Actualización: PC, PUR, ABS y rechazo NO PLÁSTICO

Las **categorías objetivo** están definidas: PET, HDPE, PVC, LDPE, PP, PS, PC, PUR, ABS y OTRO.
El `.pth` original incluido conserva **solo las seis** categorías entrenadas y no debe etiquetarse como ya entrenado para ABS u otras clases nuevas.

```python
from IA_Plasticos_PC_PUR_ABS_OTRO_API import IAPlasticos
ia = IAPlasticos()
print(ia.catalogo()) # categorías objetivo versus realmente entrenadas
r = ia.clasificar(imagen_bytes) # r['resultado'] = PVC, NO ES PLÁSTICO o NO DETERMINADO
```

- `NO ES PLÁSTICO`: cuando el clasificador **ya entrenado en OTRO** o las referencias etiquetadas OTRO ofrecen evidencia visual suficiente.
- `NO DETERMINADO`: cuando no hay evidencia suficiente para asignar una categoría, por ejemplo una mano que se parezca engañosamente a PVC. No hay garantía de detectar todos los objetos no plásticos sin un dataset de OTRO adecuado.
- `ABS`: añade a las referencias automáticamente después de importar imágenes, y se incorpora al nuevo `.pth` después del reentrenamiento.

**IMPORTANTE**: Los umbrales de semejanza son heurísticos sin calibración. Entrenar con ejemplos verificados de OTRO, y evaluar con fotos nunca vistas, es necesario antes de confiar en el rechazo. Evitá intentar deducir inocuidad o grado médico solo por una imagen.

Tu API sigue siendo responsable de recibir archivos y llamar a `clasificar`, `aprender_zip` o `reentrenar`; este paquete **no contiene servidor**.

## Verificar qué está realmente entrenado

Desde la carpeta del módulo:

```powershell
python verificar_checkpoint.py
```

Mientras no completes el reentrenamiento, es normal que indique «pendientes PC, PUR, ABS, OTRO». El adaptador **ya admite** cargar ejemplos de estas clases para comparación y las añade al próximo entrenamiento. El checkpoint `.pth` original no se reetiqueta falsamente.

## Integración a tu API existente

El archivo `ia_plasticos.py` NO abre puertos. Tu propia API debe administrar autenticación, límite de tamaño y autenticidad de etiquetas en las imágenes que reciba.

```python
# Llamadas desde tu código API en Python (no es un servidor nuevo):
from IA_Plasticos_PC_PUR_ABS_OTRO_API import IAPlasticos
ia = IAPlasticos()  # cargar una sola vez al iniciar la API

resultado = ia.clasificar(foto_bytes)            # => diccionario JSON
resultado_zip = ia.aprender_zip(zip_bytes)       # => guarda ejemplos + referencias
estados = ia.catalogo()                          # => clases objetivo/entrenadas
# ia.reentrenar()  # solo al disponer de fotos en todas las clases; tarea separada
```
