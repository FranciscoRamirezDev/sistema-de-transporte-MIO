"""
modelo_arbol.py
Modelo de aprendizaje supervisado: árbol de decisión que predice el nivel de
ocupación (baja, media, alta) de una estación del MIO (Cali).

Aplica el aprendizaje de árboles y reglas de decisión (Palma Méndez, 2008,
capítulo 17). El árbol se construye con el criterio de ENTROPÍA, es decir,
eligiendo en cada nodo la variable que aporta mayor ganancia de información,
el mismo principio de los algoritmos ID3 y C4.5. Del árbol entrenado se
extraen las REGLAS de decisión del tipo SI <condiciones> ENTONCES <clase>.

Ejecutar con:  python modelo_arbol.py
Genera:        docs/reglas_arbol.txt  (reglas extraídas del árbol)
"""

import os
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(CARPETA, "datos", "ocupacion_mio.csv")
RUTA_REGLAS = os.path.join(CARPETA, "docs", "reglas_arbol.txt")

OBJETIVO = "nivel_ocupacion"
PROPORCION_PRUEBA = 0.30    # 70 % entrenamiento, 30 % prueba
SEMILLA = 42                # resultados reproducibles

# Hiperparámetros del árbol.
CRITERIO = "entropy"        # ganancia de información (ID3 / C4.5)
PROFUNDIDAD_MAXIMA = 7      # limita el tamaño del árbol y evita el sobreajuste


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el dataset de ocupación desde el archivo CSV."""
    return pd.read_csv(ruta)


def preparar_datos(df):
    """
    Separa las variables de entrada (X) de la variable objetivo (y) y aplica
    codificación one-hot: cada valor de una variable categórica se convierte en
    una columna binaria, formato numérico que requiere scikit-learn.
    """
    X = pd.get_dummies(df.drop(columns=[OBJETIVO]), dtype=int)
    y = df[OBJETIVO]
    return X, y


def entrenar_modelo(ruta=RUTA_DATOS):
    """
    Carga los datos, los divide en entrenamiento y prueba (conservando la
    proporción de cada clase) y entrena el árbol de decisión.
    Devuelve un diccionario con el modelo y los conjuntos de datos.
    """
    df = cargar_datos(ruta)
    X, y = preparar_datos(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, random_state=SEMILLA, stratify=y)

    modelo = DecisionTreeClassifier(criterion=CRITERIO,
                                    max_depth=PROFUNDIDAD_MAXIMA,
                                    random_state=SEMILLA)
    modelo.fit(X_train, y_train)

    return {"modelo": modelo, "columnas": list(X.columns), "datos": df,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test}


def predecir(modelo, columnas, registro):
    """
    Predice el nivel de ocupación para un registro nuevo, expresado como
    diccionario con las variables originales, por ejemplo:
    {"estacion": "Popular", "linea": "L1", "tipo_estacion": "intermedia", ...}
    """
    fila = pd.get_dummies(pd.DataFrame([registro]), dtype=int)
    fila = fila.reindex(columns=columnas, fill_value=0)
    return modelo.predict(fila)[0]


def importancia_variables(modelo, columnas):
    """
    Suma la importancia de las columnas one-hot por variable original, para
    saber qué variables pesan más en las decisiones del árbol.
    """
    variables_originales = ["estacion", "linea", "tipo_estacion", "dia_tipo",
                            "franja_horaria", "hora_pico", "clima", "evento_especial"]
    total = {}
    for col, imp in zip(columnas, modelo.feature_importances_):
        base = next(v for v in variables_originales if col.startswith(v + "_"))
        total[base] = total.get(base, 0) + imp
    return sorted(total.items(), key=lambda par: par[1], reverse=True)


def reglas_legibles(modelo, columnas, max_depth=10):
    """
    Extrae las reglas del árbol (export_text) y las reescribe en lenguaje claro.
    El árbol pregunta por columnas one-hot, por ejemplo
    'franja_horaria_manana <= 0.50'; aquí se traducen a 'franja_horaria != manana'
    o 'franja_horaria = manana'. En las variables de dos valores (por ejemplo
    clima) la negación se expresa con el otro valor: 'clima != despejado'
    se escribe 'clima = lluvioso'.
    """
    texto = export_text(modelo, feature_names=columnas, max_depth=max_depth)

    variables = ["estacion", "linea", "tipo_estacion", "dia_tipo",
                 "franja_horaria", "hora_pico", "clima", "evento_especial"]
    valores = {v: [c[len(v) + 1:] for c in columnas if c.startswith(v + "_")]
               for v in variables}
    patron = re.compile(r"(" + "|".join(variables) + r")_(.+?)\s+(<=|>)\s+0\.50")

    def reemplazo(m):
        var, valor, op = m.group(1), m.group(2), m.group(3)
        if op == ">":
            return f"{var} = {valor}"
        if len(valores[var]) == 2:
            otro = [x for x in valores[var] if x != valor][0]
            return f"{var} = {otro}"
        return f"{var} != {valor}"

    texto = patron.sub(reemplazo, texto).replace("class:", "CLASE:")
    return re.sub(r"truncated branch of depth (\d+)", r"(rama resumida: \1 niveles más)", texto)


def main():
    print("=" * 64)
    print("  APRENDIZAJE SUPERVISADO - ÁRBOL DE DECISIÓN")
    print("  Predicción del nivel de ocupación de estaciones del MIO")
    print("=" * 64)

    r = entrenar_modelo()
    modelo = r["modelo"]

    print(f"\nRegistros totales      : {len(r['datos'])}")
    print(f"Variables de entrada   : {r['datos'].shape[1] - 1} "
          f"({len(r['columnas'])} columnas tras la codificación one-hot)")
    print(f"Entrenamiento / prueba : {len(r['X_train'])} / {len(r['X_test'])} registros")
    print(f"Criterio               : {CRITERIO} (ganancia de información)")
    print(f"Profundidad del árbol  : {modelo.get_depth()}")
    print(f"Número de hojas (reglas): {modelo.get_n_leaves()}")

    prec_train = modelo.score(r["X_train"], r["y_train"])
    prec_test = modelo.score(r["X_test"], r["y_test"])
    print(f"\nPrecisión en entrenamiento : {prec_train:.2%}")
    print(f"Precisión en prueba        : {prec_test:.2%}")

    print("\nImportancia de las variables:")
    for var, imp in importancia_variables(modelo, r["columnas"]):
        barra = "#" * round(imp * 40)
        print(f"  {var:16} {imp:6.1%}  {barra}")

    reglas = reglas_legibles(modelo, r["columnas"])
    os.makedirs(os.path.dirname(RUTA_REGLAS), exist_ok=True)
    with open(RUTA_REGLAS, "w", encoding="utf-8") as f:
        f.write("REGLAS DE DECISIÓN DEL ÁRBOL - OCUPACIÓN DEL MIO\n")
        f.write("Cada rama, leída desde la raíz hasta una hoja, es una regla\n")
        f.write("SI <condiciones> ENTONCES <CLASE>.\n\n")
        f.write(reglas)

    print("\nPrimeros niveles del árbol (reglas principales):")
    for linea in reglas_legibles(modelo, r["columnas"], max_depth=2).splitlines():
        print("  " + linea)
    print(f"\nReglas completas guardadas en: docs/reglas_arbol.txt")


if __name__ == "__main__":
    main()
