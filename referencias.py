"""CLIP k-NN por similitud coseno. No modifica la red neuronal."""
from collections import defaultdict
import hashlib
from pathlib import Path
import threading
import torch
import torch.nn.functional as F
try:
    from .config import DATOS, RESULTADOS, EXTENSIONES_IMAGEN
except ImportError:
    from config import DATOS, RESULTADOS, EXTENSIONES_IMAGEN

class MemoriaVisual:
    def __init__(self, modelo, carpeta=DATOS):
        self.modelo = modelo
        self.carpeta = Path(carpeta)
        self._index = []
        self._firma = None
        self._lock = threading.RLock()
        RESULTADOS.mkdir(parents=True,exist_ok=True)
        self._cache = RESULTADOS/'referencias.pt'

    def fotos(self):
        if not self.carpeta.exists():
            return []
        fotos=[]
        for d in sorted(self.carpeta.iterdir()):
            if d.is_dir():
                fotos.extend((d.name, p) for p in sorted(d.iterdir())
                             if p.is_file() and p.suffix.lower() in EXTENSIONES_IMAGEN)
        return fotos

    def _fingerprint(self, fotos):
        archivo = self.modelo.archivo
        st = archivo.stat()
        partes=[f'{archivo.resolve()}:{st.st_size}:{st.st_mtime_ns}']
        partes.extend(f'{c}/{p.name}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for c,p in fotos)
        return hashlib.sha256('\n'.join(partes).encode('utf-8')).hexdigest()

    def indexar(self, forzar=False):
        with self._lock:
            fotos=self.fotos()
            firma=self._fingerprint(fotos)
            if not forzar and firma==self._firma:
                return len(self._index)
            if not forzar and self._cache.exists():
                try:
                    datos=torch.load(self._cache,map_location='cpu',weights_only=True)
                    if datos['firma']==firma:
                        self._index=datos['referencias']; self._firma=firma
                        return len(self._index)
                except (KeyError, OSError, RuntimeError, ValueError):
                    pass
            ref=[]
            for clase,ruta in fotos:
                try:
                    v=self.modelo.vectorizar(ruta).detach().cpu().flatten()
                    ref.append((clase,F.normalize(v,dim=0)))
                except (OSError, ValueError, RuntimeError) as e:
                    print(f'Imagen omitida {ruta}: {e}')
            self._index=ref; self._firma=firma
            torch.save({'firma':firma,'referencias':ref},self._cache)
            return len(ref)

    def sugerir_vector(self, vector):
        self.indexar()
        with self._lock:
            if not self._index:
                return {'categoria':None,'estado':'SIN_EJEMPLOS'}
            v=F.normalize(vector.detach().cpu().flatten(),dim=0)
            puntajes=defaultdict(list)
            for clase,ref in self._index:
                puntajes[clase].append(float(torch.dot(v,ref)))
            # Media de 3 mejores por clase, evita un único falso parecido.
            orden=sorted(((clase,sum(sorted(scores,reverse=True)[:3])/min(3,len(scores)),len(scores))
                         for clase,scores in puntajes.items()),key=lambda x:x[1],reverse=True)
            clase,sim,n=orden[0]
            margen=sim-orden[1][1] if len(orden)>1 else 0
            # Heurísticas que deberán calibrarse con fotos reales. No son probabilidades.
            consistente=(n>=5 and len(orden)>=2 and sim>=0.83 and margen>=0.04)
            return {'categoria':clase,'similitud_coseno':round(sim,4),
                    'ejemplos_clase':n,'margen_coseno':round(margen,4),
                    'estado':'SUGERENCIA_CON_REFERENCIAS' if consistente else 'EVIDENCIA_INSUFICIENTE',
                    'nota':'Similitud visual; NO verifica composición química ni aptitud médica.'}

    def sugerir(self, imagen):
        return self.sugerir_vector(self.modelo.vectorizar(imagen))
