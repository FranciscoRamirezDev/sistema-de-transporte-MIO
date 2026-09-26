"""
main.py
Aplicación de consola del Sistema Experto de Rutas del MIO (Cali).

Solicita al usuario una estación de origen (A) y una de destino (B), consulta
el motor de búsqueda heurística (busqueda.a_estrella) sobre la base de
conocimiento (mio_kb) y muestra la mejor ruta encontrada.

Contrato que devuelve busqueda.a_estrella(origen, destino):
    diccionario con las claves:
      exito        -> bool
      mensaje      -> str (texto informativo cuando no hay ruta)
      camino       -> lista de estaciones en orden
      tiempo_total -> int (minutos, incluye penalización por transbordo)
      transbordos  -> lista de estaciones donde se cambia de línea
      tramos       -> lista de dicts {desde, hasta, linea, tiempo}
"""

import mio_kb as kb
from busqueda import a_estrella


def mostrar_estaciones():
    """Imprime la lista numerada de estaciones disponibles."""
    print("\nEstaciones disponibles:")
    for i, e in enumerate(kb.listar_estaciones(), 1):
        marca = "  (transbordo)" if kb.es_transbordo(e) else ""
        print(f"  {i:2}. {e}{marca}")


def pedir_estacion(rol):
    """Pide una estación al usuario; acepta el número de la lista o el nombre."""
    estaciones = kb.listar_estaciones()
    while True:
        dato = input(f"\nEstación de {rol} (número o nombre): ").strip()
        if dato.isdigit():
            idx = int(dato)
            if 1 <= idx <= len(estaciones):
                return estaciones[idx - 1]
        else:
            for e in estaciones:
                if e.lower() == dato.lower():
                    return e
        print("  Entrada no válida. Intente de nuevo.")


def mostrar_resultado(res, origen, destino):
    """Imprime la ruta encontrada de forma legible."""
    print("\n" + "=" * 60)
    if not res["exito"] or not res["camino"]:
        print(f"  {res['mensaje']}")
        print("=" * 60)
        return
    if len(res["camino"]) == 1:
        print(f"  {res['mensaje']}")
        print("=" * 60)
        return
    print(f"  MEJOR RUTA:  {origen}  ->  {destino}")
    print("=" * 60)
    for tr in res["tramos"]:
        print(f"  [{tr['linea']}]  {tr['desde']}  ->  {tr['hasta']}  ({tr['tiempo']} min)")
    print("-" * 60)
    detalle = f"  ({', '.join(res['transbordos'])})" if res["transbordos"] else ""
    print(f"  Estaciones en la ruta : {len(res['camino'])}")
    print(f"  Transbordos           : {len(res['transbordos'])}{detalle}")
    print(f"  Tiempo total estimado : {res['tiempo_total']} min")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  SISTEMA EXPERTO DE RUTAS - MIO CALI")
    print("  Mejor ruta de A a B mediante búsqueda heurística (A*)")
    print("=" * 60)
    while True:
        mostrar_estaciones()
        origen = pedir_estacion("ORIGEN (A)")
        destino = pedir_estacion("DESTINO (B)")
        resultado = a_estrella(origen, destino)
        mostrar_resultado(resultado, origen, destino)
        otra = input("\n¿Consultar otra ruta? (s/n): ").strip().lower()
        if otra != "s":
            print("\nHasta pronto.")
            break


if __name__ == "__main__":
    main()
