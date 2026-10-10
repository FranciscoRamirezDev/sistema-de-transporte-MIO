"""
evaluacion.py
Evaluación y pruebas del árbol de decisión que predice el nivel de ocupación
de las estaciones del MIO (Cali).

Utiliza el modelo entrenado en modelo_arbol.py y realiza cuatro pruebas:
  1. Métricas generales sobre el conjunto de prueba (datos que el modelo no vio).
  2. Matriz de confusión: aciertos y errores por clase.
  3. Validación cruzada: estabilidad del modelo con distintas particiones.
  4. Escenarios de prueba: predicciones para situaciones concretas del MIO,
     comparadas con el resultado esperado. El resultado esperado se calcula con
     la misma regla que generó los datos (generar_dataset.py), sin el ruido
     aleatorio, para tener una referencia objetiva.

Ejecutar con:  python evaluacion.py
"""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

import generar_dataset as gen
import generar_dataset as gen
from modelo_arbol import (CRITERIO, PROFUNDIDAD_MAXIMA, SEMILLA,
                          entrenar_modelo, predecir, preparar_datos)

CLASES = ["baja", "media", "alta"]

# Escenarios de prueba: (descripción, estación, tipo de día, franja, clima, evento)
ESCENARIOS = [
    ("Transbordo en hora pico de la mañana con lluvia",
     "Plaza de Caicedo", "laboral", "manana", "lluvioso", "no"),
    ("Terminal en hora pico de la tarde",
     "Terminal Menga", "laboral", "tarde", "despejado", "no"),
    ("Transbordo un sábado al mediodía con lluvia",
     "Unidad Deportiva", "sabado", "mediodia", "lluvioso", "no"),
    ("Estación intermedia con evento un sábado en la mañana con lluvia",
     "Las Américas", "sabado", "manana", "lluvioso", "si"),
    ("Estación intermedia en día laboral al mediodía",
     "Popular", "laboral", "mediodia", "despejado", "no"),
    ("Terminal un domingo en la tarde",
     "Terminal Aguablanca", "domingo_festivo", "tarde", "despejado", "no"),
    ("Estación intermedia en día laboral de noche",
     "San Bosco", "laboral", "noche", "despejado", "no"),
    ("Estación intermedia un sábado al mediodía",
     "Salomia", "sabado", "mediodia", "despejado", "no"),
    ("Estación intermedia un domingo de madrugada",
     "Nuevo Latir", "domingo_festivo", "madrugada", "despejado", "no"),
    ("Estadio con evento especial un domingo en la noche",
     "Estadio", "domingo_festivo", "noche", "despejado", "si"),
    ("Estación intermedia en día laboral de madrugada",
     "Chiminangos", "laboral", "madrugada", "despejado", "no"),
    ("Estación intermedia un sábado en la noche",
     "Torre de Cali", "sabado", "noche", "despejado", "no"),
]


def construir_registro(estacion, dia, franja, clima, evento):
    """Arma el registro completo; la línea, el tipo y la hora pico se derivan."""
    linea, tipo = gen.ESTACIONES[estacion]
    return {"estacion": estacion, "linea": linea, "tipo_estacion": tipo,
            "dia_tipo": dia, "franja_horaria": franja,
            "hora_pico": "si" if gen.es_hora_pico(dia, franja) else "no",
            "clima": clima, "evento_especial": evento}


def resultado_esperado(registro):
    """Aplica la regla de generación de los datos sin ruido aleatorio."""
    puntaje = (gen.APORTE_FRANJA[registro["franja_horaria"]]
               + gen.APORTE_DIA[registro["dia_tipo"]]
               + gen.APORTE_TIPO[registro["tipo_estacion"]]
               + (gen.APORTE_LLUVIA if registro["clima"] == "lluvioso" else 0)
               + (gen.APORTE_EVENTO if registro["evento_especial"] == "si" else 0))
    return puntaje, gen.nivel_ocupacion(puntaje)


def titulo(texto):
    print("\n" + "=" * 64)
    print("  " + texto)
    print("=" * 64)


def prueba_metricas(r):
    titulo("PRUEBA 1. MÉTRICAS SOBRE EL CONJUNTO DE PRUEBA")
    y_pred = r["modelo"].predict(r["X_test"])
    print(f"Registros evaluados : {len(r['y_test'])} (no usados en el entrenamiento)")
    print(f"Precisión (accuracy): {accuracy_score(r['y_test'], y_pred):.2%}\n")
    print(classification_report(r["y_test"], y_pred, labels=CLASES, digits=3,
                                zero_division=0))
    return y_pred


def prueba_matriz(r, y_pred):
    titulo("PRUEBA 2. MATRIZ DE CONFUSIÓN")
    matriz = confusion_matrix(r["y_test"], y_pred, labels=CLASES)
    print("Filas = clase real   |   Columnas = clase predicha\n")
    print(f"{'':>10}" + "".join(f"{c:>10}" for c in CLASES))
    for clase, fila in zip(CLASES, matriz):
        print(f"{clase:>10}" + "".join(f"{v:>10}" for v in fila))
    aciertos = matriz.trace()
    print(f"\nAciertos: {aciertos} de {matriz.sum()}  |  Errores: {matriz.sum() - aciertos}")
    graves = matriz[0][2] + matriz[2][0]
    print(f"Errores entre extremos (baja <-> alta): {graves}")


def prueba_validacion_cruzada(r):
    titulo("PRUEBA 3. VALIDACIÓN CRUZADA (5 PARTICIONES)")
    X, y = preparar_datos(r["datos"])
    arbol = DecisionTreeClassifier(criterion=CRITERIO, max_depth=PROFUNDIDAD_MAXIMA,
                                   random_state=SEMILLA)
    pliegues = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    puntajes = cross_val_score(arbol, X, y, cv=pliegues)
    for i, p in enumerate(puntajes, 1):
        print(f"  Partición {i}: {p:.2%}")
    print(f"\nPrecisión promedio : {puntajes.mean():.2%}")
    print(f"Desviación estándar: {puntajes.std():.2%}")


def prueba_escenarios(r):
    titulo("PRUEBA 4. ESCENARIOS DE PRUEBA")
    correctos = 0
    for i, (desc, *datos) in enumerate(ESCENARIOS, 1):
        registro = construir_registro(*datos)
        puntaje, esperado = resultado_esperado(registro)
        obtenido = predecir(r["modelo"], r["columnas"], registro)
        ok = obtenido == esperado
        correctos += ok
        print(f"\nEscenario {i}: {desc}")
        print(f"   {registro['estacion']} | {registro['dia_tipo']} | {registro['franja_horaria']}"
              f" | clima {registro['clima']} | evento {registro['evento_especial']}")
        print(f"   Puntaje de demanda: {puntaje:.1f}   Esperado: {esperado:6}"
              f"  Predicho: {obtenido:6}  [{'PASA' if ok else 'FALLA'}]")
    print(f"\nResultado: {correctos}/{len(ESCENARIOS)} escenarios correctos")


def main():
    print("=" * 64)
    print("  EVALUACIÓN DEL MODELO - OCUPACIÓN DE ESTACIONES DEL MIO")
    print("=" * 64)
    r = entrenar_modelo()
    y_pred = prueba_metricas(r)
    prueba_matriz(r, y_pred)
    prueba_validacion_cruzada(r)
    prueba_escenarios(r)


if __name__ == "__main__":
    main()
