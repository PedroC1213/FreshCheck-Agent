"""Importa FruitVision (carpeta descomprimida) a data/raw/fruitvision/<clase>/<fruta>__<id>.<ext>.

Reglas (Hito 1): fresh -> fresca, rotten -> danada, formalin -> se EXCLUYE (fuera del diseno).
La clase 'maduracion' no existe en FruitVision. Las imagenes con aumentos se excluyen con --excluir-regex.

IMPORTANTE: la clase se deduce del NOMBRE de las carpetas/archivos. Corra primero con --simular y revise
el resumen; si muestra muchas imagenes 'sin clase detectada' (p. ej. si el dataset trae etiquetas en
archivos .txt tipo YOLO), la estructura es distinta y hay que adaptar este script.

Uso:  python -m src.import_fruitvision --origen RUTA/FruitVision --simular
      python -m src.import_fruitvision --origen RUTA/FruitVision --excluir-regex "aug"
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path

from . import config
from .config import Rutas

FRUTAS = {"apple": "manzana", "banana": "banano", "mango": "mango", "orange": "naranja", "grape": "uva"}


def clasificar(partes: list[str]):
    txt = "/".join(partes).lower()
    if "formalin" in txt:
        return "excluida_formalina", None
    if "rotten" in txt or "podrid" in txt:
        clase = "danada"
    elif "fresh" in txt:
        clase = "fresca"
    else:
        return "sin_clase", None
    fruta = next((v for k, v in FRUTAS.items() if k in txt), "desconocida")
    return clase, fruta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", type=Path, required=True)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--excluir-regex", default=None, help="regex (sin distinguir mayusculas) de rutas a excluir, p. ej. 'aug'")
    ap.add_argument("--simular", action="store_true", help="no copia; solo muestra el resumen")
    a = ap.parse_args(argv)
    destino = Rutas(a.raiz).raw / "fruitvision"
    excl = re.compile(a.excluir_regex, re.I) if a.excluir_regex else None

    cuentas, por_carpeta = Counter(), Counter()
    for p in sorted(a.origen.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in config.EXTENSIONES:
            continue
        rel = p.relative_to(a.origen)
        if excl and excl.search(rel.as_posix()):
            cuentas["excluida_aumento"] += 1
            continue
        clase, fruta = clasificar(list(rel.parts))
        top = rel.parts[0] if len(rel.parts) > 1 else "."
        por_carpeta[(top, clase)] += 1
        if clase in ("excluida_formalina", "sin_clase"):
            cuentas[clase] += 1
            continue
        cuentas[(clase, fruta)] += 1
        if not a.simular:
            d = destino / clase
            d.mkdir(parents=True, exist_ok=True)
            nuevo = d / f"{fruta}__{hashlib.sha1(rel.as_posix().encode()).hexdigest()[:8]}_{p.stem}{p.suffix.lower()}"
            shutil.copy2(p, nuevo)

    print("Resumen por (carpeta de primer nivel, clase detectada):")
    for (top, clase), n in sorted(por_carpeta.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(f"  {top:<30} {str(clase):<20} {n}")
    print("\nResumen por clase y fruta:")
    for k, n in sorted(cuentas.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(k):<40} {n}")
    print("\n(simulacion: no se copio nada)" if a.simular else f"\n-> copiadas a {destino}")


if __name__ == "__main__":
    main()
