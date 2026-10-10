"""
generar_dataset.py
Generador del dataset de ocupación del MIO (Cali) para aprendizaje supervisado.

No existe un conjunto de datos público y abierto con la ocupación por estación y
franja horaria del MIO, por lo que, tal como lo permite la actividad, se construye
un dataset de MUESTRA. Los registros se generan a partir de reglas que reflejan
patrones conocidos del transporte masivo (horas pico, días laborales, estaciones
terminales y de transbordo, lluvia y eventos) y se les agrega ruido aleatorio para
que el problema sea realista y no trivial.

Las estaciones y líneas son las mismas de la base de conocimiento de la
Actividad 2 (mio_kb.py), para dar continuidad al proyecto.

Ejecutar con:  python generar_dataset.py
Genera:        datos/ocupacion_mio.csv
"""

import csv
import os
import random

SEMILLA = 42            # semilla fija: el dataset es reproducible
NUM_REGISTROS = 800
ARCHIVO_SALIDA = os.path.join("datos", "ocupacion_mio.csv")

# Estaciones del MIO: (línea, tipo de estación).
# tipo: terminal | transbordo | intermedia
ESTACIONES = {
    "Terminal Menga":         ("L1", "terminal"),
    "Chiminangos":            ("L1", "intermedia"),
    "Flora Industrial":       ("L1", "intermedia"),
    "Salomia":                ("L1", "intermedia"),
    "Popular":                ("L1", "intermedia"),
    "Manzana del Saber":      ("L1", "intermedia"),
    "Torre de Cali":          ("L1", "intermedia"),
    "Plaza de Caicedo":       ("L1-L2", "transbordo"),
    "San Bosco":              ("L2", "intermedia"),
    "San Pedro":              ("L2", "intermedia"),
    "Unidad Deportiva":       ("L2-L3", "transbordo"),
    "Terminal Guadalupe":     ("L2", "terminal"),
    "Estadio":                ("L3", "intermedia"),
    "Las Américas":           ("L3", "intermedia"),
    "Terminal Andrés Sanín":  ("L3", "terminal"),
    "Nuevo Latir":            ("L3", "intermedia"),
    "Terminal Aguablanca":    ("L3", "terminal"),
}

DIAS = ["laboral", "sabado", "domingo_festivo"]
PESO_DIAS = [0.65, 0.18, 0.17]          # más registros en días laborales

FRANJAS = ["madrugada", "manana", "mediodia", "tarde", "noche"]
CLIMAS = ["despejado", "lluvioso"]
PESO_CLIMA = [0.70, 0.30]

# Aporte de cada factor a la demanda (puntaje).
APORTE_FRANJA = {"madrugada": 0.0, "manana": 2.0, "mediodia": 1.0, "tarde": 2.0, "noche": 0.5}
APORTE_DIA = {"laboral": 1.0, "sabado": 0.0, "domingo_festivo": -1.0}
APORTE_TIPO = {"terminal": 1.0, "transbordo": 1.5, "intermedia": 0.0}
APORTE_LLUVIA = 0.7
APORTE_EVENTO = 1.5
RUIDO = 0.3             # desviación del ruido aleatorio


def es_hora_pico(dia, franja):
    """Hora pico: mañana o tarde en día laboral."""
    return dia == "laboral" and franja in ("manana", "tarde")


def nivel_ocupacion(puntaje):
    """Convierte el puntaje de demanda en la etiqueta de clase."""
    if puntaje < 1.5:
        return "baja"
    if puntaje < 3.0:
        return "media"
    return "alta"


def generar_registro(rng):
    estacion = rng.choice(list(ESTACIONES))
    linea, tipo = ESTACIONES[estacion]
    dia = rng.choices(DIAS, PESO_DIAS)[0]
    franja = rng.choice(FRANJAS)
    clima = rng.choices(CLIMAS, PESO_CLIMA)[0]
    evento = "si" if rng.random() < 0.08 else "no"
    pico = es_hora_pico(dia, franja)

    puntaje = (APORTE_FRANJA[franja] + APORTE_DIA[dia] + APORTE_TIPO[tipo]
               + (APORTE_LLUVIA if clima == "lluvioso" else 0)
               + (APORTE_EVENTO if evento == "si" else 0)
               + rng.gauss(0, RUIDO))

    return {
        "estacion": estacion,
        "linea": linea,
        "tipo_estacion": tipo,
        "dia_tipo": dia,
        "franja_horaria": franja,
        "hora_pico": "si" if pico else "no",
        "clima": clima,
        "evento_especial": evento,
        "nivel_ocupacion": nivel_ocupacion(puntaje),
    }


def main():
    rng = random.Random(SEMILLA)
    registros = [generar_registro(rng) for _ in range(NUM_REGISTROS)]
    os.makedirs("datos", exist_ok=True)
    with open(ARCHIVO_SALIDA, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
        escritor.writeheader()
        escritor.writerows(registros)
    print(f"Dataset generado: {ARCHIVO_SALIDA} ({len(registros)} registros)")


if __name__ == "__main__":
    main()
