"""
busqueda.py
Motor de búsqueda heurística del Sistema Experto de Rutas del MIO (Cali).

Implementa el algoritmo A* (A estrella) para encontrar la mejor ruta entre dos
estaciones, aplicando búsqueda heurística sobre la base de conocimiento
(Benítez, 2014, capítulo 9). A* combina:

  - el costo real acumulado g(n): suma del tiempo de viaje de los tramos
    recorridos más una penalización cada vez que se cambia de línea
    (transbordo); y
  - una heurística h(n): estimación optimista del tiempo restante hasta el
    destino, obtenida de la distancia en línea recta (Haversine) que provee
    la base de conocimiento.

La función principal, a_estrella(origen, destino), devuelve un diccionario con:
    exito        -> bool
    mensaje      -> str  (texto informativo cuando no hay ruta)
    camino       -> lista de estaciones en orden
    tiempo_total -> int  (minutos; incluye la penalización por transbordo)
    transbordos  -> lista de estaciones donde se cambia de línea
    tramos       -> lista de dicts {desde, hasta, linea, tiempo}
"""

import heapq
import mio_kb as kb

# Velocidad optimista (km por minuto) usada para convertir la heurística, que
# viene en kilómetros, a una estimación de tiempo en minutos. Al ser optimista
# (más rápida que el bus real), la heurística nunca sobreestima el costo, lo que
# mantiene a A* admisible y garantiza la ruta óptima.
VELOCIDAD_KM_POR_MIN = 0.5


def _heuristica(estacion, destino):
    """h(n): estima el tiempo restante (min) hasta el destino en línea recta."""
    return kb.distancia_linea_recta(estacion, destino) / VELOCIDAD_KM_POR_MIN


def _reconstruir(padre, est_final, linea_final, tiempo_total):
    """Reconstruye la ruta desde el destino hacia el origen usando el mapa de padres."""
    tramos = []
    est, linea = est_final, linea_final
    while (est, linea) in padre:
        prev_est, prev_linea, t_tramo, l_tramo = padre[(est, linea)]
        tramos.append({"desde": prev_est, "hasta": est, "linea": l_tramo, "tiempo": t_tramo})
        est, linea = prev_est, prev_linea
    tramos.reverse()

    camino = [tramos[0]["desde"]] + [tr["hasta"] for tr in tramos]
    transbordos = [tramos[i]["desde"] for i in range(1, len(tramos))
                   if tramos[i]["linea"] != tramos[i - 1]["linea"]]

    return {"exito": True, "mensaje": "", "camino": camino,
            "tiempo_total": tiempo_total, "transbordos": transbordos, "tramos": tramos}


def a_estrella(origen, destino):
    """
    Encuentra la mejor ruta (menor tiempo) entre 'origen' y 'destino' aplicando
    el algoritmo A* sobre la base de conocimiento del MIO.
    """
    # Validación de entradas contra la base de conocimiento (reglas).
    if not kb.existe_estacion(origen) or not kb.existe_estacion(destino):
        return {"exito": False, "mensaje": "Estación de origen o destino no válida.",
                "camino": [], "tiempo_total": 0, "transbordos": [], "tramos": []}

    if origen == destino:
        return {"exito": True, "mensaje": "El origen y el destino son la misma estación.",
                "camino": [origen], "tiempo_total": 0, "transbordos": [], "tramos": []}

    # El estado del nodo incluye la línea en la que se llega a la estación, para
    # poder contabilizar los transbordos correctamente.
    # Frontera (cola de prioridad): (f = g + h, g, estacion, linea_actual)
    frontera = [(_heuristica(origen, destino), 0, origen, None)]
    mejor_g = {(origen, None): 0}
    padre = {}  # (estacion, linea) -> (est_prev, linea_prev, tiempo_tramo, linea_tramo)

    while frontera:
        f, g, est, linea = heapq.heappop(frontera)

        # Al extraer el destino por primera vez, A* garantiza que es la ruta óptima.
        if est == destino:
            return _reconstruir(padre, est, linea, g)

        # Expandir vecinos aplicando la regla vecinos(x) de la base de conocimiento.
        for dest, t, l in kb.vecinos(est):
            hay_transbordo = linea is not None and l != linea
            nuevo_g = g + t + (kb.PENALIZACION_TRANSBORDO if hay_transbordo else 0)
            clave = (dest, l)
            if nuevo_g < mejor_g.get(clave, float("inf")):
                mejor_g[clave] = nuevo_g
                padre[clave] = (est, linea, t, l)
                heapq.heappush(frontera, (nuevo_g + _heuristica(dest, destino), nuevo_g, dest, l))

    # Frontera agotada sin llegar al destino: no hay ruta.
    return {"exito": False, "mensaje": "No existe una ruta entre las estaciones indicadas.",
            "camino": [], "tiempo_total": 0, "transbordos": [], "tramos": []}
