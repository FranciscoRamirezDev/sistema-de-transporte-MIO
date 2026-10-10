"""
modelo_clustering.py
Modelo de aprendizaje NO supervisado: agrupamiento de usuarios del MIO (Cali)
según su comportamiento de viaje.

Aplica las técnicas de agrupamiento (Palma Méndez, 2008, capítulo 16). Los
datos no tienen etiqueta: el algoritmo K-Means descubre por sí solo grupos de
usuarios parecidos. El proceso es:

  1. Estandarizar las variables para que todas pesen igual en las distancias.
  2. Elegir el número de grupos (k) con el método del codo.
  3. Entrenar K-Means con el k elegido.
  4. Describir cada grupo con el promedio de sus variables (centroide) y
     asignarle un nombre interpretable.

Ejecutar con:  python modelo_clustering.py
Genera:        docs/metodo_codo.png            (gráfica del método del codo)
               datos/usuarios_agrupados.csv    (usuarios con su grupo asignado)
"""

import os

import matplotlib
matplotlib.use("Agg")       # genera las gráficas como archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

CARPETA = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(CARPETA, "datos", "usuarios_mio.csv")
RUTA_AGRUPADOS = os.path.join(CARPETA, "datos", "usuarios_agrupados.csv")
RUTA_CODO = os.path.join(CARPETA, "docs", "metodo_codo.png")

VARIABLES = ["viajes_semana", "hora_primer_viaje", "pct_viajes_fin_semana",
             "transbordos_promedio", "distancia_promedio_km", "recargas_mes"]
K_GRUPOS = 4                # número de grupos elegido con el método del codo
RANGO_K = range(2, 9)       # valores de k evaluados en el método del codo
SEMILLA = 42


def cargar_datos(ruta=RUTA_DATOS):
    """Lee el dataset de usuarios. El perfil real NO se carga: no es una entrada."""
    return pd.read_csv(ruta)


def escalar(df):
    """
    Estandariza las variables (media 0, desviación 1). Sin este paso, las
    variables con valores grandes, como el porcentaje de fin de semana (0 a 100),
    dominarían la distancia frente a otras, como los transbordos (0 a 4).
    """
    escalador = StandardScaler()
    X = escalador.fit_transform(df[VARIABLES])
    return X, escalador


def metodo_codo(X):
    """
    Entrena K-Means para cada k del rango y registra la inercia: la suma de las
    distancias al cuadrado de cada usuario al centro de su grupo. El 'codo' es
    el punto a partir del cual agregar más grupos casi no reduce la inercia.
    """
    inercias = {}
    for k in RANGO_K:
        modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X)
        inercias[k] = modelo.inertia_
    return inercias


def graficar_codo(inercias, ruta=RUTA_CODO):
    ks, valores = list(inercias), list(inercias.values())
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ks, valores, marker="o", color="#0F4761", linewidth=2)
    ax.axvline(K_GRUPOS, color="#C55A11", linestyle="--", label=f"k elegido = {K_GRUPOS}")
    ax.set_title("Método del codo")
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Inercia")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def entrenar_kmeans(X, k=K_GRUPOS):
    """Entrena el modelo K-Means definitivo con k grupos."""
    return KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(X)


def perfiles_grupos(df, etiquetas):
    """Promedio de cada variable por grupo (centroides en unidades originales)."""
    datos = df[VARIABLES].copy()
    datos["grupo"] = etiquetas
    perfiles = datos.groupby("grupo").mean()
    perfiles["usuarios"] = datos.groupby("grupo").size()
    return perfiles


def nombrar_grupos(perfiles):
    """
    Asigna un nombre interpretable a cada grupo según sus características:
      - el grupo con mayor porcentaje de viajes en fin de semana -> ocasional;
      - de los restantes, el de mayor distancia -> largo recorrido;
      - de los restantes, el que empieza a viajar más tarde -> estudiante;
      - el grupo que queda -> trabajador de rutina.
    """
    nombres = {}
    restantes = perfiles.copy()
    for variable, nombre in [("pct_viajes_fin_semana", "Ocasional de fin de semana"),
                             ("distancia_promedio_km", "Largo recorrido"),
                             ("hora_primer_viaje", "Estudiante")]:
        grupo = restantes[variable].idxmax()
        nombres[grupo] = nombre
        restantes = restantes.drop(index=grupo)
    nombres[restantes.index[0]] = "Trabajador de rutina"
    return nombres


def asignar_usuario(modelo, escalador, usuario):
    """Asigna un usuario nuevo (diccionario con las 6 variables) a un grupo."""
    fila = pd.DataFrame([usuario])[VARIABLES]
    return int(modelo.predict(escalador.transform(fila))[0])


def construir_modelo(ruta=RUTA_DATOS):
    """Ejecuta el proceso completo y devuelve todo lo necesario para evaluarlo."""
    df = cargar_datos(ruta)
    X, escalador = escalar(df)
    modelo = entrenar_kmeans(X)
    perfiles = perfiles_grupos(df, modelo.labels_)
    nombres = nombrar_grupos(perfiles)
    return {"datos": df, "X": X, "escalador": escalador, "modelo": modelo,
            "perfiles": perfiles, "nombres": nombres}


def main():
    print("=" * 66)
    print("  APRENDIZAJE NO SUPERVISADO - AGRUPAMIENTO K-MEANS")
    print("  Segmentación de usuarios del MIO según su comportamiento")
    print("=" * 66)

    df = cargar_datos()
    X, _ = escalar(df)
    print(f"\nUsuarios: {len(df)}   |   Variables: {len(VARIABLES)} (estandarizadas)")

    print("\nMÉTODO DEL CODO")
    print(f"  {'k':>3} {'Inercia':>10} {'Reducción frente a k-1':>24}")
    inercias = metodo_codo(X)
    anterior = None
    for k, inercia in inercias.items():
        reduccion = "" if anterior is None else f"{(anterior - inercia) / anterior:.1%}"
        print(f"  {k:>3} {inercia:>10.1f} {reduccion:>24}")
        anterior = inercia
    graficar_codo(inercias)
    print(f"  -> Se eligen k = {K_GRUPOS} grupos: a partir de ahí agregar grupos")
    print("     casi no reduce la inercia (gráfica en docs/metodo_codo.png).")

    r = construir_modelo()
    perfiles, nombres = r["perfiles"], r["nombres"]

    print(f"\nPERFIL DE LOS {K_GRUPOS} GRUPOS (promedio de cada variable)")
    encabezado = (f"  {'Grupo':<28}{'Usuarios':>9}{'Viajes':>8}{'Hora':>7}"
                  f"{'%FinS':>7}{'Transb':>8}{'Km':>6}{'Recar':>7}")
    print(encabezado)
    print("  " + "-" * (len(encabezado) - 2))
    for grupo, fila in perfiles.iterrows():
        print(f"  {grupo}. {nombres[grupo]:<25}{int(fila['usuarios']):>9}"
              f"{fila['viajes_semana']:>8.1f}{fila['hora_primer_viaje']:>7.1f}"
              f"{fila['pct_viajes_fin_semana']:>7.1f}{fila['transbordos_promedio']:>8.2f}"
              f"{fila['distancia_promedio_km']:>6.1f}{fila['recargas_mes']:>7.1f}")

    agrupados = r["datos"].copy()
    agrupados["grupo"] = r["modelo"].labels_
    agrupados["nombre_grupo"] = agrupados["grupo"].map(nombres)
    agrupados.to_csv(RUTA_AGRUPADOS, index=False)
    print("\nUsuarios con su grupo asignado guardados en: datos/usuarios_agrupados.csv")


if __name__ == "__main__":
    main()
