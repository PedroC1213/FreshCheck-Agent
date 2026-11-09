# Datos (proyecto final, FreshCheck-Agent v2)

## Fuente única: FruitVision

| Atributo | Detalle |
|---|---|
| Dataset | FruitVision (Bijoy, Tasnim, Awsaf y Hasan, 2025), Mendeley Data V2, doi 10.17632/xkbjx8959c.2 |
| Contenido | 10 154 imágenes originales de manzana, banano, mango, naranja y uva (81 232 con aumentos). Categorías: fresca, podrida y con formalina |
| Licencia | **CC BY-NC-ND 4.0** según la página de Mendeley. El repositorio de GitHub de uno de los autores declara MIT; se aplica la más restrictiva |
| Uso aquí | Académico y no comercial. **No se redistribuyen** las imágenes ni versiones modificadas: `data/raw/` está en `.gitignore` |
| Descarga | https://data.mendeley.com/datasets/xkbjx8959c/2 (botón *Download All*; puede pedir iniciar sesión) |

Mapeo: fresh → `fresca`, rotten → `danada`, formalin → **excluida**. Las imágenes con aumentos no se importan: los
aumentos se aplican solo sobre entrenamiento, en el código. La clase «en maduración» del Hito 1 no existe en FruitVision
y se eliminó (ver `docs/cambios_respecto_hito1.md`).

## Importar

```
python -m src.import_fruitvision --origen "C:\RUTA\FruitVision" --excluir-regex "aug" --simular
python -m src.import_fruitvision --origen "C:\RUTA\FruitVision" --excluir-regex "aug"
```

Con `--simular` no se copia nada. Revise el resumen: `fresca` y `danada` deben sumar menos de 10 154 imágenes, con
`excluida_formalina` y `excluida_aumento` aparte y `sin_clase` cerca de 0. Si suman cerca de 90 000, las aumentadas no
se están excluyendo (habría fuga de datos); si hay muchas `sin_clase`, la estructura es distinta y hay que adaptar el script.

Resultado: `data/raw/fruitvision/{fresca,danada}/<fruta>__<id>.jpg`, donde `<fruta>` ∈ manzana, banano, mango, naranja, uva.

## Archivos generados (se versionan)

- `manifest.csv`: una fila por imagen (clase, fruta, tamaño, dHash, grupo de duplicados).
- `splits.csv`: partición 70/15/15 (semilla 42), estratificada por clase y fruta y por grupo de duplicados.

## Limitaciones de los datos

- Según la descripción del dataset, las imágenes se tomaron con tres celulares (iPhone 15 Pro Max, Redmi POCO M2 Reloaded y Redmi Note 9 Pro) con luz del día, luz interior y **fondos simples**: no representan una línea de empaque real. El README de los autores menciona 512×512, pero las 6 978 imágenes importadas tienen 455 resoluciones distintas (633–4160 px de ancho y 718–4472 px de alto; sobre todo 960×1280 y 1280×960): el modelo las redimensiona.
- El control de duplicados (dHash, Hamming ≤ 2) produjo 269 grupos con más de una imagen, y 93 de ellos mezclan clases: parte son falsos positivos del hash (fondos lisos). Agrupar de más es conservador, pero el control de duplicados reales es grueso.
- Recolectado por una sola institución (Daffodil International University, Bangladesh): posible sesgo hacia ese tipo de fruta, iluminación y fondo.
- Sin identificador del fruto físico: puede haber fotos de la misma fruta en particiones distintas.
- Los aumentos oficiales (rotación, volteo, zoom, brillo, corte y ruido) se excluyen de la importación; verifique en `--simular` que el filtro los detecta.
