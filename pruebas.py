"""
pruebas.py
Casos de prueba del Sistema Experto de Rutas del MIO (Cali).

Ejecuta un conjunto de consultas de ruta contra el motor de búsqueda y valida
que los resultados sean coherentes (la ruta inicia en el origen, termina en el
destino y el tiempo total es válido). Sirve como evidencia para el documento
PDF de pruebas.

Ejecutar con:  python pruebas.py
"""

import mio_kb as kb
from busqueda import a_estrella

# Casos: (origen, destino)
CASOS = [
    ("Terminal Menga", "Terminal Aguablanca"),   # ruta larga con 2 transbordos
    ("Terminal Guadalupe", "Terminal Menga"),     # sur -> norte
    ("Flora Industrial", "Estadio"),              # con transbordo
    ("Popular", "San Pedro"),                     # con un transbordo
    ("Plaza de Caicedo", "Plaza de Caicedo"),     # mismo origen y destino
    ("Salomia", "Chiminangos"),                   # misma línea, sin transbordo
]


def validar(origen, destino, res):
    """Devuelve una lista de problemas encontrados (vacía si el caso pasa)."""
    problemas = []
    if not res["exito"]:
        return ["la búsqueda no tuvo éxito: " + res["mensaje"]]
    if res["camino"]:
        if res["camino"][0] != origen:
            problemas.append("la ruta no inicia en el origen")
        if res["camino"][-1] != destino:
            problemas.append("la ruta no termina en el destino")
    if res["tiempo_total"] < 0:
        problemas.append("el tiempo total es negativo")
    return problemas


def main():
    print("=" * 62)
    print("  PRUEBAS - SISTEMA EXPERTO DE RUTAS MIO CALI")
    print("=" * 62)
    superados = 0
    for i, (origen, destino) in enumerate(CASOS, 1):
        res = a_estrella(origen, destino)
        problemas = validar(origen, destino, res)
        estado = "PASA" if not problemas else "FALLA"
        if not problemas:
            superados += 1
        print(f"\nCaso {i}: {origen}  ->  {destino}   [{estado}]")
        if res["exito"] and res["camino"]:
            print("   Ruta        :", "  ->  ".join(res["camino"]))
            print("   Transbordos :", len(res["transbordos"]))
            print("   Tiempo total:", res["tiempo_total"], "min")
        for p in problemas:
            print("   -", p)
    print("\n" + "=" * 62)
    print(f"  RESULTADO: {superados}/{len(CASOS)} casos superados")
    print("=" * 62)


if __name__ == "__main__":
    main()
