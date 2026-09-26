# Sistema Experto de Rutas — MIO Cali

Sistema inteligente basado en conocimiento que, a partir de una **base de conocimiento escrita en reglas lógicas**, calcula la **mejor ruta** entre dos estaciones del sistema de transporte masivo **MIO (Cali)** mediante **búsqueda heurística (algoritmo A\*)**.

Proyecto académico de la asignatura **Inteligencia Artificial**, Corporación Universitaria Iberoamericana. Fundamentado en Benítez, R. (2014). *Inteligencia artificial avanzada*. Editorial UOC (capítulos 2, 3 y 9).

## Enfoque

El proyecto integra los tres conceptos centrales de la unidad:

1. **Representación del conocimiento (cap. 2):** el sistema de transporte se modela como *hechos* (estaciones y conexiones) y coordenadas geográficas.
2. **Sistema basado en reglas (cap. 3):** *reglas* (predicados) que consultan los hechos, por ejemplo `vecinos(x)`, `es_transbordo(x)` y `lineas_de(x)`.
3. **Búsqueda heurística (cap. 9):** el algoritmo A\* explora la base de conocimiento usando como heurística la distancia en línea recta al destino, garantizando la ruta óptima en tiempo.

## Estructura del proyecto

| Archivo | Descripción | Responsable |
|---|---|---|
| `mio_kb.py` | Base de conocimiento: hechos (estaciones, conexiones) y reglas (predicados) | Francisco Ramírez |
| `busqueda.py` | Motor de búsqueda heurística A\* (costo + heurística) | Juliana Campos |
| `main.py` | Aplicación de consola: solicita origen/destino y muestra la ruta | Juan Esteban Fierro |
| `pruebas.py` | Casos de prueba automatizados | Juan Esteban Fierro |
| `docs/pruebas.pdf` | Documento con las pruebas realizadas | Juan Esteban Fierro |

## Requisitos

- **Python 3.8** o superior.
- No requiere librerías externas; utiliza únicamente la librería estándar.

## Ejecución

Ejecutar la aplicación principal:

```bash
python main.py
```

El programa solicita la estación de **origen (A)** y la de **destino (B)** y muestra la mejor ruta: secuencia de estaciones, líneas utilizadas, transbordos y tiempo total estimado en minutos.

Ejecutar los casos de prueba:

```bash
python pruebas.py
```

## Interfaz de la base de conocimiento (`mio_kb.py`)

Módulos que consumen `busqueda.py` y `main.py`:

1. `ESTACIONES` — diccionario de estaciones con sus líneas y coordenadas.
2. `CONEXIONES` — lista de conexiones directas `(origen, destino, tiempo, linea)`.
3. `PENALIZACION_TRANSBORDO` — minutos que se suman al cambiar de línea.
4. `existe_estacion(x)`, `lineas_de(x)`, `es_transbordo(x)`.
5. `vecinos(x)` — devuelve `[(destino, tiempo, linea), ...]`.
6. `distancia_linea_recta(x, y)` — heurística en kilómetros (Haversine).
7. `listar_estaciones()` — lista ordenada de estaciones.

## Enlaces de la entrega

- **Repositorio Git:** (este repositorio)
- **Video explicativo:** *(pendiente)*

## Autores

1. Francisco Luis Ramírez Galvis
2. Juliana Campos Florez
3. Juan Esteban Fierro Cortes