# Bitácora humano-IA — proyecto final (FreshCheck-Agent v2)

Registro de los usos de IA generativa que cambiaron decisiones o produjeron código y texto del proyecto. La bitácora de la
actividad anterior está en `docs/Bitacora_uso_IA.pdf`. Herramienta en todos los casos: Claude (Sonnet 5.5, claude.ai).
Integrante: Pedro Castro.

| Fecha | Para qué se usó | Qué se aceptó, modificó o descartó | Cómo se verificó |
|---|---|---|---|
| 2026-11-07 | Elegir ruta y alcance y redactar la propuesta del Hito 1 (PEAS, datos, línea base, métricas, umbrales) | Aceptó continuar FreshCheck-Agent en la ruta F, la línea base y las métricas. Modificó: pidió imágenes de internet en lugar de fotos propias. Descartó: empezar un problema nuevo y el enfoque mixto con fotos propias | Claude revisó el repositorio anterior y la descripción de FruitVision, y detectó que el dataset no tiene la clase «en maduración» |
| 2026-11-07 | Generar el código del pipeline v2 (partición sin fuga, línea base, métricas, umbrales, pruebas) | Aceptó el código sin cambiar su contenido; solo corrigió los nombres de archivo al subirlo a GitHub | Claude lo probó con datos sintéticos y comparó los archivos subidos con los originales. Pedro ejecutó la prueba en su equipo (pasó) y la línea base con FruitVision |
| 2026-11-08 | Verificar la licencia y la estructura de FruitVision y rediseñar el alcance para usar solo FruitVision | Pedro pidió quitar las imágenes de internet. Claude advirtió que se pierde la clase «en maduración» y propuso dos clases, medir la fruta no vista y ajustar los umbrales. Aceptó y subió los cambios (commit 8366c2a) | Claude leyó la página de Mendeley (licencia CC BY-NC-ND 4.0, no MIT) y el README de los autores; después, el manifiesto mostró 455 resoluciones y no 512×512 |
| 2026-11-09 | Interpretar los resultados de la línea base y redactar el avance del Hito 2 | Aceptó los números tal como salieron de `experiments/baseline/`; el texto lo redactó Claude | Claude leyó los archivos de resultados del repositorio y contrastó cada cifra con la salida de consola que Pedro envió |
| 2026-11-12 | Diseñar y generar el código del Hito 3: ajuste fino de ResNet-50 y EfficientNet-B0, búsqueda de hiperparámetros, evaluación final, prototipo Streamlit y cuaderno de Colab | El código lo generó Claude a pedido de Pedro. Se descartó incluir en el informe un mosaico con fotos de errores, por la licencia CC BY-NC-ND de los datos; el mosaico queda solo para uso local | Claude lo probó con datos sintéticos y con torch real sin pesos preentrenados: el entrenamiento aprende, guarda y carga el modelo y recorre todo el flujo. **No** se ejecutó con FruitVision ni con pesos de ImageNet; Pedro debe ejecutarlo y registrar aquí qué cambió |
| 2026-11-12 | Redactar el informe IEEE con un generador LaTeX que toma las cifras de los resultados | Claude redactó el texto y las referencias; Pedro debe revisarlo y poder defenderlo | Claude compiló un borrador con las cifras reales del Hito 2 y marcas «pendiente» donde faltan los resultados finales; las cifras finales salen de `experiments/final/` |

## Reflexión: casos en que la IA propuso algo inadecuado

**Licencia.** La IA anotó la licencia de FruitVision como MIT, porque así figura en el repositorio de GitHub de uno de los
autores, y dijo que se confirmaría. Al revisar la página oficial en Mendeley, la licencia era **CC BY-NC-ND 4.0**, más
restrictiva (sin uso comercial ni obras derivadas). Se corrigió la documentación, se aplicó la licencia más estricta y no se
publican las imágenes. Lección: verificar en la fuente oficial del dataset y no en un repositorio secundario.

**Resolución y duplicados.** La IA documentó imágenes de 512×512 según el README de los autores y propuso detectar duplicados
con un hash de imagen. Con las fotos reales se vio que tienen 455 resoluciones distintas y que 93 de los 269 grupos de
«duplicados» mezclan clases, es decir, el hash agrupa imágenes distintas con fondos lisos. Se corrigió la documentación y el
control de duplicados se reporta como limitación. Ambos casos se detectaron contrastando lo que dijo la IA con la fuente
primaria o con los datos reales.

**Cuadrícula estrecha.** La IA propuso una cuadrícula de regularización cuyo mejor valor quedó en el extremo; se detectó al revisar los resultados del Hito 2 (macro-F1 irregular entre valores) y se amplió en la línea base final. **Licencia de las figuras.** La IA incluyó primero un mosaico con fotos de errores en el informe; al revisar la licencia CC BY-NC-ND (sin obras derivadas) se retiró y se dejó solo para uso local.

> Agregue una fila por cada nuevo uso importante de IA (fecha, para qué, qué aceptó/modificó/descartó, cómo lo verificó) y cambie «Pedro debe…» por lo que realmente hizo.
