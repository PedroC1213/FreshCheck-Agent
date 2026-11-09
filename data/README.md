# Datos (proyecto final, FreshCheck-Agent v2)

## Fuentes

| Fuente | Qué aporta | Licencia / permiso | Cómo se obtiene |
|---|---|---|---|
| **FruitVision** (Mendeley Data, doi 10.17632/xkbjx8959c.2) | 10 154 imágenes originales de manzana, banano, mango, naranja y uva (81 232 con aumentos). Categorías: fresca, podrida, con formalina | MIT según el repositorio de los autores; **confirmar en la página de Mendeley** y anotar aquí el resultado: `[completar]` | Descargar y descomprimir; luego `python -m src.import_fruitvision --origen RUTA --excluir-regex "aug" --simular` y, si el resumen es correcto, sin `--simular` |
| **Imágenes de internet** | Las tres clases, incluida «en maduración» (que FruitVision no tiene) | Solo imágenes con licencia abierta (p. ej., Wikimedia Commons, Openverse con filtro de licencia). **No** se usa descarga masiva desde buscadores | A mano, a `data/raw/internet/<clase>/<fruta>__<id>.jpg`, y una fila por imagen en `sources_internet.csv` |

Mapeo de FruitVision: fresh → `fresca`, rotten → `danada`, formalin → **excluida** (fuera del diseño). Las imágenes
con aumentos no se importan: los aumentos se aplican solo sobre entrenamiento, en el código.

## `sources_internet.csv`

Una fila por imagen de internet. Columnas: `archivo` (nombre del archivo, sin carpeta), `clase`, `fruta`, `url`,
`autor`, `licencia`, `fecha_descarga`, `notas`. `src.make_manifest` **se detiene** si alguna imagen de
`data/raw/internet/` no tiene licencia en este archivo. Borre la fila de ejemplo antes de empezar.

## Criterio de etiquetado (ejemplo de partida: ajústelo y déjelo fijo antes de etiquetar)

| Clase | Criterio visual |
|---|---|
| `fresca` | Piel íntegra, color y firmeza propios de fruta lista para empacar; sin manchas ni golpes visibles |
| `maduracion` | Cambio de color o manchas leves propios del avance de la maduración, sin pudrición ni daño profundo |
| `danada` | Pudrición, moho, golpes o heridas profundas, arrugas marcadas |

Reglas: una sola fruta principal por imagen, sin personas ni datos personales; todas las clases se reúnen de las mismas
fuentes (para que el modelo no aprenda la fuente en lugar del estado); mínimo 200 imágenes por clase de internet;
etiquetado propio y único: la subjetividad de «maduración» es una limitación que se reporta en el informe.

## Archivos generados (se versionan)

- `manifest.csv`: una fila por imagen (fuente, clase, fruta, tamaño, dHash, grupo de duplicados).
- `splits.csv`: partición 70/15/15 (semilla 42), estratificada por clase y fuente y por grupo de duplicados.
