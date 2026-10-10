"""
evaluacion.py
Evaluación y pruebas del agrupamiento K-Means de usuarios del MIO (Cali).

En el aprendizaje no supervisado no hay una etiqueta con la cual medir aciertos
durante el entrenamiento, por lo que el modelo se evalúa desde varios ángulos:

  1. Coeficiente de silueta: qué tan compactos y separados están los grupos.
  2. Comparación con agrupamiento jerárquico: si otro método encuentra los
     mismos grupos.
  3. Estabilidad: si K-Means encuentra los mismos grupos con distintas
     inicializaciones aleatorias.
  4. Coincidencia con los perfiles reales: los perfiles de referencia, que el
     modelo nunca vio, se usan SOLO aquí para comprobar los grupos.
  5. Asignación de usuarios nuevos a un grupo.

Ejecutar con:  python evaluacion.py
Genera en docs/: silueta_por_k.png, grupos_pca.png, dendrograma.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from modelo_clustering import (CARPETA, K_GRUPOS, RANGO_K, SEMILLA, VARIABLES,
                               asignar_usuario, construir_modelo)

RUTA_PERFILES = os.path.join(CARPETA, "datos", "perfiles_referencia.csv")
CARPETA_DOCS = os.path.join(CARPETA, "docs")
COLORES = ["#0F4761", "#C55A11", "#548235", "#7F6000"]

# Equivalencia entre el nombre de cada grupo y el perfil con que se generaron los datos.
NOMBRE_A_PERFIL = {"Trabajador de rutina": "trabajador_rutina",
                   "Estudiante": "estudiante",
                   "Ocasional de fin de semana": "ocasional_fin_semana",
                   "Largo recorrido": "largo_recorrido"}

# Usuarios nuevos de prueba: (descripción, variables, perfil esperado).
# El perfil esperado es aquel cuyos valores promedio, definidos en
# generar_dataset.py, están más cerca del usuario (distancia con las variables
# estandarizadas). Los casos 5 y 6 están cerca de la frontera entre dos perfiles.
USUARIOS_NUEVOS = [
    ("Sale a las 6 a. m. y viaja de lunes a viernes",
     dict(viajes_semana=10, hora_primer_viaje=6.2, pct_viajes_fin_semana=4,
          transbordos_promedio=1.1, distancia_promedio_km=8.5, recargas_mes=4),
     "Trabajador de rutina"),
    ("Viaja entre semana desde media mañana, trayectos cortos",
     dict(viajes_semana=8, hora_primer_viaje=10.0, pct_viajes_fin_semana=8,
          transbordos_promedio=0.5, distancia_promedio_km=4.5, recargas_mes=8),
     "Estudiante"),
    ("Viaja casi solo los fines de semana",
     dict(viajes_semana=2, hora_primer_viaje=12.0, pct_viajes_fin_semana=80,
          transbordos_promedio=0.2, distancia_promedio_km=5.0, recargas_mes=1),
     "Ocasional de fin de semana"),
    ("Sale muy temprano, viaje largo con varios transbordos",
     dict(viajes_semana=10, hora_primer_viaje=5.2, pct_viajes_fin_semana=10,
          transbordos_promedio=2.5, distancia_promedio_km=19.0, recargas_mes=5),
     "Largo recorrido"),
    ("Caso frontera: trabajador que vive algo lejos",
     dict(viajes_semana=11, hora_primer_viaje=6.0, pct_viajes_fin_semana=6,
          transbordos_promedio=1.6, distancia_promedio_km=13.0, recargas_mes=4),
     "Trabajador de rutina"),
    ("Caso frontera: estudiante que también viaja el sábado",
     dict(viajes_semana=7, hora_primer_viaje=9.0, pct_viajes_fin_semana=25,
          transbordos_promedio=0.6, distancia_promedio_km=5.5, recargas_mes=6),
     "Estudiante"),
]


def titulo(texto):
    print("\n" + "=" * 66)
    print("  " + texto)
    print("=" * 66)


def guardar(fig, nombre):
    os.makedirs(CARPETA_DOCS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(CARPETA_DOCS, nombre), dpi=150)
    plt.close(fig)


def prueba_silueta(r):
    titulo("PRUEBA 1. COEFICIENTE DE SILUETA")
    print("Mide de -1 a 1 qué tan bien ubicado está cada usuario en su grupo:")
    print("valores altos indican grupos compactos y bien separados.\n")
    X = r["X"]
    siluetas = {}
    for k in RANGO_K:
        etiquetas = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(X)
        siluetas[k] = silhouette_score(X, etiquetas)
        marca = "  <- k elegido" if k == K_GRUPOS else ""
        print(f"  k = {k}:  silueta = {siluetas[k]:.3f}{marca}")

    valores = silhouette_samples(X, r["modelo"].labels_)
    print(f"\nSilueta por grupo (k = {K_GRUPOS}):")
    for grupo, nombre in sorted(r["nombres"].items()):
        media = valores[r["modelo"].labels_ == grupo].mean()
        print(f"  {nombre:<28} {media:.3f}")
    print(f"\nUsuarios con silueta negativa (posiblemente mal ubicados): "
          f"{(valores < 0).sum()} de {len(valores)}")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(list(siluetas), list(siluetas.values()), marker="o", color="#0F4761", linewidth=2)
    ax.axvline(K_GRUPOS, color="#C55A11", linestyle="--", label=f"k elegido = {K_GRUPOS}")
    ax.set_title("Coeficiente de silueta según el número de grupos")
    ax.set_xlabel("Número de grupos (k)")
    ax.set_ylabel("Silueta promedio")
    ax.legend(); ax.grid(alpha=0.3)
    guardar(fig, "silueta_por_k.png")


def prueba_jerarquico(r):
    titulo("PRUEBA 2. COMPARACIÓN CON AGRUPAMIENTO JERÁRQUICO")
    X = r["X"]
    jerarquico = AgglomerativeClustering(n_clusters=K_GRUPOS, linkage="ward").fit_predict(X)
    ari = adjusted_rand_score(r["modelo"].labels_, jerarquico)
    print("Se agrupan los mismos usuarios con el método jerárquico aglomerativo")
    print("(enlace de Ward) y se compara con K-Means usando el índice de Rand")
    print("ajustado (1 = agrupamientos idénticos, 0 = coincidencia al azar).\n")
    print(f"  Índice de Rand ajustado K-Means vs. jerárquico: {ari:.3f}")

    enlaces = linkage(X, method="ward")
    # Altura de corte que deja exactamente K_GRUPOS grupos: punto medio entre
    # la unión que forma K_GRUPOS grupos y la que forma K_GRUPOS - 1.
    corte = (enlaces[-K_GRUPOS, 2] + enlaces[-(K_GRUPOS - 1), 2]) / 2
    set_link_color_palette(COLORES)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    dendrogram(enlaces, truncate_mode="lastp", p=30, ax=ax, color_threshold=corte,
               above_threshold_color="#808080", leaf_font_size=7)
    ax.axhline(corte, color="black", linestyle="--", linewidth=1,
               label=f"Corte en {K_GRUPOS} grupos")
    ax.set_title("Dendrograma del agrupamiento jerárquico (últimas 30 uniones)")
    ax.set_xlabel("Grupos de usuarios (entre paréntesis, cantidad de usuarios)")
    ax.set_ylabel("Distancia de unión")
    ax.legend(fontsize=8)
    guardar(fig, "dendrograma.png")


def prueba_estabilidad(r):
    titulo("PRUEBA 3. ESTABILIDAD DEL MODELO")
    print("Se entrena K-Means con 5 semillas aleatorias distintas y se compara")
    print("cada resultado con el modelo principal.\n")
    for semilla in [1, 7, 21, 99, 2026]:
        etiquetas = KMeans(n_clusters=K_GRUPOS, n_init=10, random_state=semilla).fit_predict(r["X"])
        ari = adjusted_rand_score(r["modelo"].labels_, etiquetas)
        print(f"  Semilla {semilla:>4}: índice de Rand ajustado = {ari:.3f}")


def prueba_perfiles_reales(r):
    titulo("PRUEBA 4. COINCIDENCIA CON LOS PERFILES REALES")
    referencia = pd.read_csv(RUTA_PERFILES)
    datos = r["datos"][["id_usuario"]].copy()
    datos["grupo"] = [r["nombres"][g] for g in r["modelo"].labels_]
    datos = datos.merge(referencia, on="id_usuario")

    ari = adjusted_rand_score(datos["perfil_real"], datos["grupo"])
    coinciden = (datos["grupo"].map(NOMBRE_A_PERFIL) == datos["perfil_real"]).sum()
    print("Los perfiles reales NO se usaron para entrenar; solo se usan aquí.\n")
    print(f"  Índice de Rand ajustado          : {ari:.3f}")
    print(f"  Usuarios en el grupo correcto    : {coinciden} de {len(datos)}"
          f" ({coinciden / len(datos):.1%})\n")
    tabla = pd.crosstab(datos["perfil_real"], datos["grupo"])
    tabla = tabla[[n for n in NOMBRE_A_PERFIL if n in tabla.columns]]
    tabla.columns = ["Trabajador", "Estudiante", "Ocasional", "Largo rec."]
    print("Filas = perfil real   |   Columnas = grupo asignado por K-Means\n")
    print(tabla.to_string())

    pca = PCA(n_components=2, random_state=SEMILLA)
    puntos = pca.fit_transform(r["X"])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, (grupo, nombre) in enumerate(sorted(r["nombres"].items())):
        mascara = r["modelo"].labels_ == grupo
        ax.scatter(puntos[mascara, 0], puntos[mascara, 1], s=14, alpha=0.7,
                   color=COLORES[i], label=nombre)
    centros = pca.transform(r["modelo"].cluster_centers_)
    ax.scatter(centros[:, 0], centros[:, 1], s=180, marker="X", color="black", label="Centroides")
    var = pca.explained_variance_ratio_.sum()
    ax.set_title(f"Grupos de usuarios en 2 dimensiones (PCA, {var:.0%} de la variación)")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    guardar(fig, "grupos_pca.png")


def prueba_usuarios_nuevos(r):
    titulo("PRUEBA 5. ASIGNACIÓN DE USUARIOS NUEVOS")
    correctos = 0
    for i, (desc, usuario, esperado) in enumerate(USUARIOS_NUEVOS, 1):
        grupo = asignar_usuario(r["modelo"], r["escalador"], usuario)
        obtenido = r["nombres"][grupo]
        ok = obtenido == esperado
        correctos += ok
        print(f"\nUsuario {i}: {desc}")
        print(f"   viajes {usuario['viajes_semana']}/sem | hora {usuario['hora_primer_viaje']}"
              f" | fin de semana {usuario['pct_viajes_fin_semana']}%"
              f" | transbordos {usuario['transbordos_promedio']}"
              f" | {usuario['distancia_promedio_km']} km | recargas {usuario['recargas_mes']}")
        print(f"   Esperado: {esperado:<27} Asignado: {obtenido:<27} [{'PASA' if ok else 'FALLA'}]")
    print(f"\nResultado: {correctos}/{len(USUARIOS_NUEVOS)} usuarios asignados correctamente")


def main():
    print("=" * 66)
    print("  EVALUACIÓN DEL AGRUPAMIENTO - USUARIOS DEL MIO")
    print("=" * 66)
    r = construir_modelo()
    prueba_silueta(r)
    prueba_jerarquico(r)
    prueba_estabilidad(r)
    prueba_perfiles_reales(r)
    prueba_usuarios_nuevos(r)
    print("\nGráficas generadas en docs/: silueta_por_k.png, dendrograma.png, grupos_pca.png")


if __name__ == "__main__":
    main()
