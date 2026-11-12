import os
import sys
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# 1. Definir rutas de trabajo
RUTA_BASE = r"C:\Users\cadsi\Avance_Mate"
RUTA_ENV = os.path.join(RUTA_BASE, ".env")
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "03_resultados")

ARCHIVO_TABLA_02 = os.path.join(RUTA_RESULTADOS, "tabla_02_estadistica_descriptiva_general.csv")
ARCHIVO_TABLA_03 = os.path.join(RUTA_RESULTADOS, "tabla_03_resumen_valores_extremos_ric.csv")
ARCHIVO_TABLA_04 = os.path.join(RUTA_RESULTADOS, "tabla_04_detalle_municipios_extremos.csv")
ARCHIVO_TABLA_05 = os.path.join(RUTA_RESULTADOS, "tabla_05_descriptiva_por_tipo_municipio.csv")
ARCHIVO_TABLA_06 = os.path.join(RUTA_RESULTADOS, "tabla_06_descriptiva_por_intensidad_carencias.csv")

ARCHIVO_FIGURA_01 = os.path.join(RUTA_RESULTADOS, "figura_01_histogramas_variables.png")
ARCHIVO_FIGURA_02 = os.path.join(RUTA_RESULTADOS, "figura_02_diagramas_caja_variables.png")

# 2. Cargar credenciales desde .env y leer tabla datos_limpios desde MySQL
load_dotenv(RUTA_ENV)

try:
    conexion = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE", "avance_bd"),
        port=int(os.getenv("MYSQL_PORT", "3307"))
    )
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM datos_limpios ORDER BY clave_municipio ASC;")
    filas_mysql = cursor.fetchall()
    df = pd.DataFrame(filas_mysql)
    cursor.close()
    conexion.close()
except Exception as e:
    print("ERROR al leer datos_limpios desde MySQL:", e)
    sys.exit(1)

# 3. Asegurar que las 12 variables numericas tengan formato decimal (float)
VARIABLES_NUMERICAS = [
    "ingreso_inferior_lpi_porcentaje",
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje",
    "pobreza_porcentaje",
    "pobreza_extrema_porcentaje",
    "carencias_promedio",
    "poblacion_2020",
    "pobreza_personas"
]

for col in VARIABLES_NUMERICAS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 95)
print(f"DATOS LEIDOS DESDE MYSQL (avance_bd.datos_limpios): {df.shape[0]} municipios, {df.shape[1]} columnas")
print("=" * 95)

# 4. Calcular Tabla 2: Estadistica descriptiva completa de las 12 variables numericas
lista_descriptiva = []
resumen_extremos = []
detalle_extremos = []

for col in VARIABLES_NUMERICAS:
    serie = df[col].dropna()
    n_datos = int(serie.count())
    media = float(serie.mean())
    mediana = float(serie.median())
    desv_est = float(serie.std(ddof=1))
    varianza = float(serie.var(ddof=1))
    cv_pct = float((desv_est / media) * 100.0) if media != 0 else np.nan
    minimo = float(serie.min())
    q1 = float(serie.quantile(0.25))
    q3 = float(serie.quantile(0.75))
    ric = float(q3 - q1)
    maximo = float(serie.max())
    asimetria = float(serie.skew())
    curtosis = float(serie.kurt())

    lista_descriptiva.append({
        "variable": col,
        "n": n_datos,
        "media": round(media, 4),
        "mediana": round(mediana, 4),
        "desv_est": round(desv_est, 4),
        "varianza": round(varianza, 4),
        "coef_variacion_pct": round(cv_pct, 4),
        "minimo": round(minimo, 4),
        "Q1": round(q1, 4),
        "Q3": round(q3, 4),
        "rango_intercuartilico": round(ric, 4),
        "maximo": round(maximo, 4),
        "asimetria": round(asimetria, 4),
        "curtosis": round(curtosis, 4)
    })

    # Regla del rango intercuartilico (RIC) para valores extremos
    limite_inferior = q1 - 1.5 * ric
    limite_superior = q3 + 1.5 * ric

    df_bajos = df[df[col] < limite_inferior].sort_values(by=col, ascending=True)
    df_altos = df[df[col] > limite_superior].sort_values(by=col, ascending=False)

    nombres_bajos = [f"{r['municipio']} ({r[col]:.2f})" for _, r in df_bajos.iterrows()]
    nombres_altos = [f"{r['municipio']} ({r[col]:.2f})" for _, r in df_altos.iterrows()]

    resumen_extremos.append({
        "variable": col,
        "limite_inferior_ric": round(limite_inferior, 4),
        "limite_superior_ric": round(limite_superior, 4),
        "num_extremos_bajos": len(nombres_bajos),
        "num_extremos_altos": len(nombres_altos),
        "total_extremos": len(nombres_bajos) + len(nombres_altos),
        "municipios_extremos_bajos": "; ".join(nombres_bajos) if nombres_bajos else "Ninguno",
        "municipios_extremos_altos": "; ".join(nombres_altos) if nombres_altos else "Ninguno"
    })

    for _, fila in df_bajos.iterrows():
        detalle_extremos.append({
            "variable": col,
            "tipo_extremo": "Bajo (< Q1 - 1.5*RIC)",
            "clave_municipio": fila["clave_municipio"],
            "municipio": fila["municipio"],
            "valor": round(float(fila[col]), 4),
            "umbral_ric": round(limite_inferior, 4)
        })
    for _, fila in df_altos.iterrows():
        detalle_extremos.append({
            "variable": col,
            "tipo_extremo": "Alto (> Q3 + 1.5*RIC)",
            "clave_municipio": fila["clave_municipio"],
            "municipio": fila["municipio"],
            "valor": round(float(fila[col]), 4),
            "umbral_ric": round(limite_superior, 4)
        })

df_tabla_02 = pd.DataFrame(lista_descriptiva)
df_tabla_02.to_csv(ARCHIVO_TABLA_02, index=False, encoding="utf-8-sig")

df_tabla_03 = pd.DataFrame(resumen_extremos)
df_tabla_03.to_csv(ARCHIVO_TABLA_03, index=False, encoding="utf-8-sig")

df_tabla_04 = pd.DataFrame(detalle_extremos)
df_tabla_04.to_csv(ARCHIVO_TABLA_04, index=False, encoding="utf-8-sig")

print("\nTABLA 2: ESTADISTICA DESCRIPTIVA GENERAL (PARTE A: TENDENCIA CENTRAL Y DISPERSION)")
cols_parte_a = ["variable", "n", "media", "mediana", "desv_est", "varianza", "coef_variacion_pct"]
print(df_tabla_02[cols_parte_a].to_string(index=False))

print("\nTABLA 2: ESTADISTICA DESCRIPTIVA GENERAL (PARTE B: POSICION Y FORMA)")
cols_parte_b = ["variable", "minimo", "Q1", "Q3", "rango_intercuartilico", "maximo", "asimetria", "curtosis"]
print(df_tabla_02[cols_parte_b].to_string(index=False))

print("\n" + "=" * 95)
print("TABLA 3: DETECCION DE VALORES EXTREMOS CON LA REGLA DEL RANGO INTERCUARTILICO (RIC)")
print("=" * 95)
for _, fila in df_tabla_03.iterrows():
    print(f"- {fila['variable']}: Total={fila['total_extremos']} "
          f"(Bajos={fila['num_extremos_bajos']}, Altos={fila['num_extremos_altos']}) | "
          f"Limites RIC: [{fila['limite_inferior_ric']}, {fila['limite_superior_ric']}]")
    if fila["municipios_extremos_bajos"] != "Ninguno":
        print(f"    Extremos bajos: {fila['municipios_extremos_bajos']}")
    if fila["municipios_extremos_altos"] != "Ninguno":
        print(f"    Extremos altos: {fila['municipios_extremos_altos']}")

# 5. Estadisticas agrupadas por tipo_municipio_poblacion y por nivel_intensidad_carencias
VARIABLES_MODELO = [
    "ingreso_inferior_lpi_porcentaje",
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje",
    "pobreza_porcentaje"
]

def calcular_resumen_agrupado(dataframe, col_grupo, lista_vars):
    filas_grupo = []
    conteo_grupos = dataframe[col_grupo].value_counts()
    for grupo in conteo_grupos.index:
        sub = dataframe[dataframe[col_grupo] == grupo]
        for var in lista_vars:
            s = sub[var].dropna()
            filas_grupo.append({
                col_grupo: grupo,
                "n_municipios": int(s.count()),
                "variable": var,
                "media": round(float(s.mean()), 4),
                "mediana": round(float(s.median()), 4),
                "desv_est": round(float(s.std(ddof=1)), 4),
                "minimo": round(float(s.min()), 4),
                "maximo": round(float(s.max()), 4)
            })
    return pd.DataFrame(filas_grupo)

df_tabla_05 = calcular_resumen_agrupado(df, "tipo_municipio_poblacion", VARIABLES_MODELO)
df_tabla_05.to_csv(ARCHIVO_TABLA_05, index=False, encoding="utf-8-sig")

df_tabla_06 = calcular_resumen_agrupado(df, "nivel_intensidad_carencias", VARIABLES_MODELO)
df_tabla_06.to_csv(ARCHIVO_TABLA_06, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 5: ESTADISTICAS DESCRIPTIVAS POR TIPO DE MUNICIPIO (POBLACION)")
print("=" * 95)
print(df_tabla_05.to_string(index=False))

print("\n" + "=" * 95)
print("TABLA 6: ESTADISTICAS DESCRIPTIVAS POR NIVEL DE INTENSIDAD DE CARENCIAS (SOLO DESCRIPCION)")
print("=" * 95)
print(df_tabla_06.to_string(index=False))

# 6. Generar Figura 1: Histogramas de las 9 variables porcentuales principales
VARIABLES_GRAFICAS = [
    ("ingreso_inferior_lpi_porcentaje", "Ingreso inferior a LPI (%) [Variable y]"),
    ("carencia_vivienda_porcentaje", "Carencia por vivienda (%) [x1]"),
    ("carencia_servicios_basicos_porcentaje", "Carencia servicios basicos (%) [x2]"),
    ("carencia_alimentacion_porcentaje", "Carencia por alimentacion (%) [x3]"),
    ("rezago_educativo_porcentaje", "Rezago educativo (%) [x4]"),
    ("carencia_salud_porcentaje", "Carencia servicios de salud (%) [x5]"),
    ("carencia_seguridad_social_porcentaje", "Carencia seguridad social (%) [x6]"),
    ("pobreza_porcentaje", "Pobreza multidimensional (%)"),
    ("pobreza_extrema_porcentaje", "Pobreza extrema (%)")
]

fig1, axes1 = plt.subplots(3, 3, figsize=(15, 11))
axes1 = axes1.flatten()

for idx, (col, titulo) in enumerate(VARIABLES_GRAFICAS):
    ax = axes1[idx]
    datos = df[col].dropna()
    media_val = datos.mean()
    mediana_val = datos.median()

    ax.hist(datos, bins=14, color="#3b6998", edgecolor="black", alpha=0.82)
    ax.axvline(media_val, color="#d62728", linestyle="--", linewidth=1.8, label=f"Media: {media_val:.2f}%")
    ax.axvline(mediana_val, color="#2ca02c", linestyle="-", linewidth=1.8, label=f"Mediana: {mediana_val:.2f}%")
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel("Porcentaje municipal (%)", fontsize=8)
    ax.set_ylabel("Frecuencia (municipios)", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")

fig1.suptitle(
    "Figura 1: Histogramas de distribucion de las variables de pobreza y carencias en Jalisco (n = 125)",
    fontsize=13,
    fontweight="bold",
    y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(ARCHIVO_FIGURA_01, dpi=300)
plt.close(fig1)

# 7. Generar Figura 2: Diagramas de caja (Boxplots) para identificar asimetria y valores extremos
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 10))
axes2 = axes2.flatten()

for idx, (col, titulo) in enumerate(VARIABLES_GRAFICAS):
    ax = axes2[idx]
    datos = df[col].dropna()
    q1 = datos.quantile(0.25)
    q3 = datos.quantile(0.75)
    ric = q3 - q1
    lim_inf = q1 - 1.5 * ric
    lim_sup = q3 + 1.5 * ric

    ax.boxplot(
        datos,
        orientation="horizontal",
        patch_artist=True,
        boxprops=dict(facecolor="#8ebad9", edgecolor="black"),
        medianprops=dict(color="#d62728", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="#ff7f0e", markersize=6, markeredgecolor="black")
    )
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.set_xlabel("Porcentaje (%)", fontsize=8)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    # Anotar el municipio mas extremo si existe
    extremos_altos = df[df[col] > lim_sup].sort_values(by=col, ascending=False)
    if not extremos_altos.empty:
        top_mun = extremos_altos.iloc[0]
        ax.annotate(
            f"{top_mun['municipio']} ({top_mun[col]:.1f}%)",
            xy=(top_mun[col], 1),
            xytext=(-15, 14),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            fontweight="bold",
            color="#b22222",
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=0.8)
        )

fig2.suptitle(
    "Figura 2: Diagramas de caja y deteccion de municipios con valores extremos (Regla 1.5 x RIC)",
    fontsize=13,
    fontweight="bold",
    y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(ARCHIVO_FIGURA_02, dpi=300)
plt.close(fig2)

print("\n" + "=" * 95)
print("ARCHIVOS GUARDADOS EN 03_resultados:")
print(f"1) {ARCHIVO_TABLA_02}")
print(f"2) {ARCHIVO_TABLA_03}")
print(f"3) {ARCHIVO_TABLA_04}")
print(f"4) {ARCHIVO_TABLA_05}")
print(f"5) {ARCHIVO_TABLA_06}")
print(f"6) {ARCHIVO_FIGURA_01} (Figura 1: histogramas de las variables)")
print(f"7) {ARCHIVO_FIGURA_02} (Figura 2: diagramas de caja de las variables)")
print("=" * 95)