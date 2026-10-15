"""Constantes compartidas por todos los módulos."""

CLASES = ("fresca", "maduracion", "danada")
ACCIONES = ("CONTINUAR", "DESVIAR", "REVISION_MANUAL", "NINGUNA")
IMG_SIZE = 160                     # lado de la imagen simulada (píxeles)

# Geometría y temporización de la línea piloto simulada
DISTANCIA_CAMARA_DESVIADOR_M = 0.60   # metros entre la cámara y el desviador
MARGEN_SEGURIDAD = 0.80               # fracción del tiempo disponible que el agente se permite usar
COOLDOWN_DESVIADOR_S = 0.060          # tiempo mínimo entre dos accionamientos del desviador
