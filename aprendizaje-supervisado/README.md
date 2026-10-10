# Aprendizaje supervisado — Predicción de ocupación del MIO Cali

Actividad 3 de la asignatura **Inteligencia Artificial** (Corporación Universitaria Iberoamericana). Continúa el proyecto del sistema de transporte masivo **MIO** aplicando **aprendizaje supervisado**: un **árbol de decisión** que predice el **nivel de ocupación** de una estación (`baja`, `media`, `alta`) según el día, la franja horaria, el tipo de estación, el clima y la presencia de eventos.

Fundamentado en Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*. McGraw-Hill, capítulo 17 (Aprendizaje de árboles y reglas de decisión).

## Fuente de datos

No existe un conjunto de datos abierto con la ocupación del MIO por estación y franja horaria, por lo que se construyó un **dataset de muestra** de 800 registros (`datos/ocupacion_mio.csv`) con el script `generar_dataset.py`. Los registros siguen patrones reales del transporte masivo (horas pico, días laborales, terminales y transbordos, lluvia y eventos) e incluyen ruido aleatorio. La descripción completa está en `docs/descripcion_datos.pdf`.

## Estructura

| Archivo | Descripción | Responsable |
|---|---|---|
| `generar_dataset.py` | Genera el dataset de muestra | Francisco Ramírez |
| `datos/ocupacion_mio.csv` | Fuente de datos (800 registros) | Francisco Ramírez |
| `docs/descripcion_datos.pdf` | Descripción y diccionario de los datos | Francisco Ramírez |
| `modelo_arbol.py` | Entrenamiento del árbol de decisión y extracción de reglas | Juliana Campos |
| `evaluacion.py` | Evaluación del modelo y predicciones de prueba | Juan Esteban Fierro |
| `docs/pruebas.pdf` | Documento de pruebas del modelo | Juan Esteban Fierro |

## Requisitos

- Python 3.8 o superior.
- Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python generar_dataset.py   # (opcional) regenera datos/ocupacion_mio.csv
python modelo_arbol.py      # entrena el modelo y muestra las reglas
python evaluacion.py        # evalúa el modelo y realiza predicciones
```

El dataset se genera con semilla fija, por lo que siempre produce los mismos datos.

## Autores

1. Francisco Luis Ramírez Galvis
2. Juliana Campos Florez
3. Juan Esteban Fierro Cortes
