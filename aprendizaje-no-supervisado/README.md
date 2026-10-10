# Aprendizaje no supervisado — Segmentación de usuarios del MIO Cali

Actividad 4 de la asignatura **Inteligencia Artificial** (Corporación Universitaria Iberoamericana). Continúa el proyecto del sistema de transporte masivo **MIO** aplicando **aprendizaje no supervisado**: técnicas de **agrupamiento (clustering)** que descubren, sin etiquetas previas, **perfiles de usuarios** según su comportamiento de viaje.

Fundamentado en Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*. McGraw-Hill, capítulo 16 (Técnicas de agrupamiento).

## Fuente de datos

Las transacciones de la tarjeta MIO son datos personales y no se publican de forma abierta, por lo que se construyó un **dataset de muestra** de 600 usuarios (`datos/usuarios_mio.csv`) con el script `generar_dataset.py`. Cada usuario se describe con 6 variables numéricas de su comportamiento mensual. La descripción completa está en `docs/descripcion_datos.pdf`.

El archivo `datos/perfiles_referencia.csv` contiene el perfil con el que se generó cada usuario. **No se usa para entrenar el modelo**: solo se utiliza en las pruebas para comprobar si los grupos descubiertos coinciden con la realidad.

## Estructura

| Archivo | Descripción | Responsable |
|---|---|---|
| `generar_dataset.py` | Genera el dataset de muestra | Francisco Ramírez |
| `datos/usuarios_mio.csv` | Fuente de datos (600 usuarios) | Francisco Ramírez |
| `datos/perfiles_referencia.csv` | Perfiles reales, solo para pruebas | Francisco Ramírez |
| `docs/descripcion_datos.pdf` | Descripción y diccionario de los datos | Francisco Ramírez |
| `modelo_clustering.py` | Escalado, método del codo, K-Means y perfiles de los grupos | Juliana Campos |
| `evaluacion.py` | Evaluación del agrupamiento y gráficas | Juan Esteban Fierro |
| `docs/pruebas.pdf` | Documento de pruebas del modelo | Juan Esteban Fierro |

## Requisitos

- Python 3.8 o superior.
- Instalar dependencias:

```bash
python3 -m pip install -r requirements.txt
```

## Ejecución

```bash
python3 generar_dataset.py     # (opcional) regenera los datos
python3 modelo_clustering.py   # agrupa a los usuarios y describe cada grupo
python3 evaluacion.py          # evalúa el agrupamiento y genera las gráficas
```

El dataset se genera con semilla fija, por lo que siempre produce los mismos datos.

## Autores

1. Francisco Luis Ramírez Galvis
2. Juliana Campos Florez
3. Juan Esteban Fierro Cortes
