"""
generar_dataset.py
Generador del dataset de usuarios del MIO (Cali) para aprendizaje NO supervisado.

Cada registro representa a un usuario de la tarjeta MIO y describe su
comportamiento de viaje durante un mes típico. No existe un conjunto de datos
abierto con esta información (las transacciones de la tarjeta son datos
personales protegidos), por lo que, tal como lo permite la actividad, se
construye un dataset de MUESTRA.

Para que los datos tengan una estructura realista, cada usuario se genera a
partir de un perfil de comportamiento (trabajador de rutina, estudiante,
usuario ocasional de fin de semana o usuario de largo recorrido). Ese perfil
NO forma parte de los datos de entrada del modelo: se guarda en un archivo
aparte (perfiles_referencia.csv) y solo se usa al final, en las pruebas, para
comprobar si los grupos que descubre el modelo coinciden con la realidad.

Ejecutar con:  python generar_dataset.py
Genera:        datos/usuarios_mio.csv          (datos de entrada del modelo)
               datos/perfiles_referencia.csv   (solo para las pruebas)
"""

import csv
import os
import random

SEMILLA = 42
NUM_USUARIOS = 600
CARPETA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_USUARIOS = os.path.join(CARPETA, "datos", "usuarios_mio.csv")
ARCHIVO_PERFILES = os.path.join(CARPETA, "datos", "perfiles_referencia.csv")

# Perfiles de comportamiento: proporción de usuarios y (media, desviación) de
# cada variable.
PERFILES = {
    "trabajador_rutina": {
        "proporcion": 0.35,
        "viajes_semana": (11, 1.5),           # ida y vuelta de lunes a viernes
        "hora_primer_viaje": (6.5, 0.6),      # salida temprano
        "pct_viajes_fin_semana": (5, 4),
        "transbordos_promedio": (1.0, 0.35),
        "distancia_promedio_km": (9, 2.5),
        "recargas_mes": (4, 1),
    },
    "estudiante": {
        "proporcion": 0.25,
        "viajes_semana": (9, 1.5),
        "hora_primer_viaje": (9.5, 1.3),      # horarios de clase variados
        "pct_viajes_fin_semana": (10, 5),
        "transbordos_promedio": (0.6, 0.3),
        "distancia_promedio_km": (5, 1.5),
        "recargas_mes": (7, 1.5),             # recargas pequeñas y frecuentes
    },
    "ocasional_fin_semana": {
        "proporcion": 0.25,
        "viajes_semana": (3, 1),
        "hora_primer_viaje": (11, 1.5),
        "pct_viajes_fin_semana": (65, 12),
        "transbordos_promedio": (0.4, 0.3),
        "distancia_promedio_km": (6, 2),
        "recargas_mes": (1.5, 0.7),
    },
    "largo_recorrido": {
        "proporcion": 0.15,
        "viajes_semana": (10, 1.5),
        "hora_primer_viaje": (5.5, 0.5),      # vive lejos: sale muy temprano
        "pct_viajes_fin_semana": (12, 6),
        "transbordos_promedio": (2.2, 0.4),
        "distancia_promedio_km": (17, 3),
        "recargas_mes": (5, 1),
    },
}

# Límites válidos de cada variable (los valores se recortan a este rango).
LIMITES = {
    "viajes_semana": (1, 20),
    "hora_primer_viaje": (4.5, 21),
    "pct_viajes_fin_semana": (0, 100),
    "transbordos_promedio": (0, 4),
    "distancia_promedio_km": (1, 30),
    "recargas_mes": (1, 15),
}
DECIMALES = {"viajes_semana": 0, "hora_primer_viaje": 2, "pct_viajes_fin_semana": 1,
             "transbordos_promedio": 2, "distancia_promedio_km": 1, "recargas_mes": 0}


def generar_usuario(rng, perfil):
    datos = {}
    for var, (minimo, maximo) in LIMITES.items():
        media, desv = PERFILES[perfil][var]
        valor = min(max(rng.gauss(media, desv), minimo), maximo)
        valor = round(valor, DECIMALES[var])
        datos[var] = int(valor) if DECIMALES[var] == 0 else valor
    return datos


def main():
    rng = random.Random(SEMILLA)
    perfiles = list(PERFILES)
    pesos = [PERFILES[p]["proporcion"] for p in perfiles]

    usuarios, referencia = [], []
    for i in range(1, NUM_USUARIOS + 1):
        id_usuario = f"U{i:04d}"
        perfil = rng.choices(perfiles, pesos)[0]
        usuarios.append({"id_usuario": id_usuario, **generar_usuario(rng, perfil)})
        referencia.append({"id_usuario": id_usuario, "perfil_real": perfil})

    os.makedirs(os.path.dirname(ARCHIVO_USUARIOS), exist_ok=True)
    for ruta, filas in [(ARCHIVO_USUARIOS, usuarios), (ARCHIVO_PERFILES, referencia)]:
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            escritor.writeheader()
            escritor.writerows(filas)

    print(f"Dataset generado: datos/usuarios_mio.csv ({len(usuarios)} usuarios)")
    print("Perfiles de referencia: datos/perfiles_referencia.csv (solo para pruebas)")


if __name__ == "__main__":
    main()
