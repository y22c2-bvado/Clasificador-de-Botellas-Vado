"""Importa ZIPs con carpetas de clases, sin lista de clases programada.
Cada imagen validada se guarda como JPG, nombrada con hash. Solo acepta un
nombre de categoría inmediatamente encima de la imagen.
"""
from __future__ import annotations
import hashlib
import io
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, BadZipFile
from PIL import Image, ImageOps, UnidentifiedImageError
try:
    from .config import DATOS, EXTENSIONES_IMAGEN
except ImportError:
    from config import DATOS, EXTENSIONES_IMAGEN

MAX_ZIP = 80*1024*1024
MAX_DESCOMPRIMIDO = 240*1024*1024
MAX_IMG = 10*1024*1024
MAX_CANTIDAD = 700
# Nombres válidos como PET, PUR, POLICARBONATO, OTRO. Cada nueva carpeta
# se convierte automáticamente en categoría, previa verificación humana.
NOMBRE_CLASE = re.compile(r'^[A-Za-z][A-Za-z0-9_-]{0,39}$')

class ErrorDataset(ValueError):
    pass


def importar_zip(contenido: bytes, destino: Path = DATOS) -> dict:
    if not isinstance(contenido, bytes) or len(contenido) > MAX_ZIP:
        raise ErrorDataset('ZIP inválido o mayor de 80 MB')
    pendientes = []
    try:
        with ZipFile(io.BytesIO(contenido)) as z:
            entradas = [e for e in z.infolist() if not e.is_dir()]
            if len(entradas) > MAX_CANTIDAD+60:
                raise ErrorDataset('ZIP con demasiados archivos')
            if sum(e.file_size for e in entradas) > MAX_DESCOMPRIMIDO:
                raise ErrorDataset('ZIP descomprimido demasiado grande')
            for info in entradas:
                n = info.filename
                if '\\' in n or ':' in n or n.startswith('/') or any(
                    x in ('', '.', '..') for x in n.split('/')
                ):
                    raise ErrorDataset('Ruta insegura en el ZIP')
                # Symlinks no admitidos.
                if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ErrorDataset('ZIP contiene enlaces simbólicos')
                p = PurePosixPath(n)
                if p.name.startswith('.') or '__MACOSX' in p.parts or p.name.lower()=='thumbs.db':
                    continue
                if p.suffix.lower() not in EXTENSIONES_IMAGEN:
                    if p.suffix.lower() in ('.txt', '.md', '.csv'):
                        continue
                    raise ErrorDataset(f'Archivo no admitido: {p.name}')
                if len(p.parts) < 2:
                    raise ErrorDataset(f'Imagen sin carpeta de categoría: {n}')
                clase = p.parts[-2].upper()
                if not NOMBRE_CLASE.fullmatch(clase):
                    raise ErrorDataset(f'Nombre de categoría no válido: {clase}')
                if info.file_size > MAX_IMG:
                    raise ErrorDataset(f'Imagen demasiado grande: {p.name}')
                if len(pendientes) >= MAX_CANTIDAD:
                    raise ErrorDataset('Demasiadas imágenes; máximo 700')
                with z.open(info) as f:
                    datos = f.read(MAX_IMG+1)
                if len(datos) > MAX_IMG:
                    raise ErrorDataset('Imagen mayor de 10 MB')
                try:
                    with Image.open(io.BytesIO(datos)) as im:
                        if im.width*im.height > 24_000_000:
                            raise ErrorDataset('Imagen con demasiados píxeles')
                        imagen = ImageOps.exif_transpose(im).convert('RGB')
                        salida = io.BytesIO()
                        imagen.save(salida, format='JPEG', quality=92, optimize=True)
                        jpg = salida.getvalue()
                except (UnidentifiedImageError, OSError, ValueError) as e:
                    raise ErrorDataset(f'Imagen ilegible {n}: {e}') from e
                pendientes.append((clase, jpg))
    except (BadZipFile, EOFError) as e:
        raise ErrorDataset('ZIP corrupto') from e
    if not pendientes:
        raise ErrorDataset('No se encontraron imágenes válidas')
    destino.mkdir(parents=True,exist_ok=True)
    agregados=Counter()
    repetidos=Counter()
    # Comprobación de duplicados entre clases en este lote y en dataset existente.
    huellas={}
    for p in destino.iterdir():
        if p.is_dir():
            for f in p.glob('*.jpg'):
                huellas[f.stem] = p.name
    for clase,jpg in pendientes:
        hashfoto=hashlib.sha256(jpg).hexdigest()
        if hashfoto in huellas and huellas[hashfoto] != clase:
            raise ErrorDataset(f'La misma imagen tiene dos clases: {huellas[hashfoto]}, {clase}')
        huellas[hashfoto]=clase
    for clase,jpg in pendientes:
        hashfoto=hashlib.sha256(jpg).hexdigest()
        carpeta=destino/clase
        carpeta.mkdir(exist_ok=True)
        ruta=carpeta/f'{hashfoto}.jpg'
        if ruta.exists():
            repetidos[clase]+=1
        else:
            ruta.write_bytes(jpg)
            agregados[clase]+=1
    return {'imagenes_nuevas':sum(agregados.values()), 'por_categoria':dict(agregados),
            'duplicadas':sum(repetidos.values()), 'categorias_importadas':sorted({x for x,_ in pendientes})}
