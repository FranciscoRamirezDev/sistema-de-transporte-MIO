"""
mio_kb.py
Base de conocimiento del Sistema Experto de Rutas del MIO (Cali).

Representa el conocimiento del sistema de transporte masivo mediante
HECHOS (estaciones y conexiones) y REGLAS (predicados que consultan esos
hechos), siguiendo el enfoque de representación del conocimiento y de los
sistemas basados en reglas (Benítez, 2014, capítulos 2 y 3).

NOTA: El modelo es una representación REPRESENTATIVA del MIO, no el mapa
oficial completo. Las coordenadas son aproximadas y se emplean únicamente
como heurística para la búsqueda (distancia en línea recta, capítulo 9).
"""

from math import radians, sin, cos, asin, sqrt

# ---------------------------------------------------------------------------
# HECHOS 1 — Estaciones.
# Cada estación pertenece a uno o más corredores (líneas) y tiene una
# coordenada geográfica aproximada (latitud, longitud). Una estación que
# pertenece a más de una línea es, por regla, un punto de transbordo.
# ---------------------------------------------------------------------------
ESTACIONES = {
    "Terminal Menga":         {"lineas": ["L1"],       "coord": (3.4795, -76.5165)},
    "Chiminangos":            {"lineas": ["L1"],       "coord": (3.4700, -76.5150)},
    "Flora Industrial":       {"lineas": ["L1"],       "coord": (3.4640, -76.5140)},
    "Salomia":                {"lineas": ["L1"],       "coord": (3.4590, -76.5150)},
    "Popular":                {"lineas": ["L1"],       "coord": (3.4540, -76.5170)},
    "Manzana del Saber":      {"lineas": ["L1"],       "coord": (3.4490, -76.5250)},
    "Torre de Cali":          {"lineas": ["L1"],       "coord": (3.4450, -76.5290)},
    "Plaza de Caicedo":       {"lineas": ["L1", "L2"], "coord": (3.4516, -76.5320)},
    "San Bosco":              {"lineas": ["L2"],       "coord": (3.4470, -76.5330)},
    "San Pedro":              {"lineas": ["L2"],       "coord": (3.4430, -76.5350)},
    "Unidad Deportiva":       {"lineas": ["L2", "L3"], "coord": (3.4270, -76.5410)},
    "Terminal Guadalupe":     {"lineas": ["L2"],       "coord": (3.4120, -76.5460)},
    "Estadio":                {"lineas": ["L3"],       "coord": (3.4280, -76.5320)},
    "Las Américas":           {"lineas": ["L3"],       "coord": (3.4300, -76.5200)},
    "Terminal Andrés Sanín":  {"lineas": ["L3"],       "coord": (3.4340, -76.5040)},
    "Nuevo Latir":            {"lineas": ["L3"],       "coord": (3.4270, -76.4950)},
    "Terminal Aguablanca":    {"lineas": ["L3"],       "coord": (3.4200, -76.4880)},
}

# ---------------------------------------------------------------------------
# HECHOS 2 — Conexiones directas entre estaciones contiguas.
# (estacion_origen, estacion_destino, tiempo_en_minutos, linea)
# Se declaran en un solo sentido; las reglas las tratan como bidireccionales
# (los buses circulan en ambos sentidos del corredor).
# ---------------------------------------------------------------------------
CONEXIONES = [
    # Corredor L1 (Norte - Centro)
    ("Terminal Menga",        "Chiminangos",           3, "L1"),
    ("Chiminangos",           "Flora Industrial",      2, "L1"),
    ("Flora Industrial",      "Salomia",               2, "L1"),
    ("Salomia",               "Popular",               2, "L1"),
    ("Popular",               "Manzana del Saber",     3, "L1"),
    ("Manzana del Saber",     "Torre de Cali",         2, "L1"),
    ("Torre de Cali",         "Plaza de Caicedo",      2, "L1"),
    # Corredor L2 (Centro - Sur)
    ("Plaza de Caicedo",      "San Bosco",             2, "L2"),
    ("San Bosco",             "San Pedro",             2, "L2"),
    ("San Pedro",             "Unidad Deportiva",      4, "L2"),
    ("Unidad Deportiva",      "Terminal Guadalupe",    5, "L2"),
    # Corredor L3 (Oriente)
    ("Unidad Deportiva",      "Estadio",               3, "L3"),
    ("Estadio",               "Las Américas",          3, "L3"),
    ("Las Américas",          "Terminal Andrés Sanín", 4, "L3"),
    ("Terminal Andrés Sanín", "Nuevo Latir",           3, "L3"),
    ("Nuevo Latir",           "Terminal Aguablanca",   3, "L3"),
]

# Penalización (en minutos) que se suma al realizar un transbordo de línea.
PENALIZACION_TRANSBORDO = 6

# ---------------------------------------------------------------------------
# REGLAS — predicados que operan sobre los hechos anteriores.
# ---------------------------------------------------------------------------

def existe_estacion(x):
    """Regla: x es una estación válida de la base de conocimiento."""
    return x in ESTACIONES

def lineas_de(x):
    """Regla: corredores (líneas) a los que pertenece la estación x."""
    return ESTACIONES[x]["lineas"] if existe_estacion(x) else []

def es_transbordo(x):
    """Regla: x es estación de transbordo si pertenece a más de una línea."""
    return len(lineas_de(x)) > 1

def vecinos(x):
    """
    Regla: estaciones directamente conectadas con x.
    Devuelve una lista de tuplas (destino, tiempo, linea).
    Trata las conexiones como bidireccionales.
    """
    resultado = []
    for a, b, t, linea in CONEXIONES:
        if a == x:
            resultado.append((b, t, linea))
        elif b == x:
            resultado.append((a, t, linea))
    return resultado

def distancia_linea_recta(x, y):
    """
    Heurística (capítulo 9): distancia en línea recta en kilómetros entre
    dos estaciones (fórmula de Haversine), calculada a partir de sus
    coordenadas. Es una estimación optimista del costo restante hacia el
    destino, por lo que nunca lo sobreestima.
    """
    if not (existe_estacion(x) and existe_estacion(y)):
        return 0.0
    lat1, lon1 = ESTACIONES[x]["coord"]
    lat2, lon2 = ESTACIONES[y]["coord"]
    radio_tierra_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radio_tierra_km * asin(sqrt(a))

def listar_estaciones():
    """Utilidad: lista ordenada alfabéticamente de las estaciones disponibles."""
    return sorted(ESTACIONES.keys())
