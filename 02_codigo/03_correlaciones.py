import os
import sys
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
from scipy import stats
from dotenv import load_dotenv

# 1. Definir rutas de salida
RUTA_BASE = r"C:\Users\cadsi\Avance_Mate"
RUTA_ENV = os.path.join(RUTA_BASE, ".env")
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "03_resultados")

ARCHIVO_TABLA_07 = os.path.join(RUTA_RESULTADOS, "tabla_07_matriz_pearson_variables_modelo.csv")
ARCHIVO_TABLA_08 = os.path.join(RUTA_RESULTADOS, "tabla_08_matriz_spearman_variables_modelo.csv")
ARCHIVO_TABLA_09 = os.path.join(RUTA_RESULTADOS, "tabla_09_pares_correlacion_y_valores_p_modelo.csv")
ARCHIVO_TABLA_10 = os.path.join(RUTA_RESULTADOS, "tabla_10_correlaciones_variables_derivadas_aparte.csv")
ARCHIVO_TABLA_11 = os.path.join(RUTA_RESULTADOS, "tabla_11_pares_fuertes_mayor_070.csv")

ARCHIVO_FIGURA_03 = os.path.join(RUTA_RESULTADOS, "figura_03_mapas_calor_correlaciones.png")
ARCHIVO_FIGURA_04 = os.path.join(RUTA_RESULTADOS, "figura_04_dispersion_y_vs_carencias.png")

# 2. Conectar a MySQL y leer avance_bd.datos_limpios
load_dotenv(RUTA_ENV)

try:
    conexion = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password="B3aM0N$t3R#W1n!",
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

# 3. Separar variables principales del modelo (y + 6 carencias) y variables derivadas/contexto
VARS_MODELO = [
    "ingreso_inferior_lpi_porcentaje",
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje"
]

VARS_DERIVADAS = [
    "pobreza_porcentaje",
    "pobreza_extrema_porcentaje",
    "carencias_promedio",
    "poblacion_2020",
    "pobreza_personas"
]

TODAS_NUMERICAS = VARS_MODELO + VARS_DERIVADAS
for col in TODAS_NUMERICAS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4. Calcular matrices de Pearson y Spearman para las 7 variables del modelo
matriz_pearson = df[VARS_MODELO].corr(method="pearson").round(4)
matriz_spearman = df[VARS_MODELO].corr(method="spearman").round(4)

matriz_pearson.to_csv(ARCHIVO_TABLA_07, index=True, encoding="utf-8-sig")
matriz_spearman.to_csv(ARCHIVO_TABLA_08, index=True, encoding="utf-8-sig")

print("=" * 95)
print("TABLA 7: MATRIZ DE CORRELACION LINEAL DE PEARSON (r) - VARIABLES DEL MODELO (n = 125)")
print("=" * 95)
print(matriz_pearson.to_string())

print("\n" + "=" * 95)
print("TABLA 8: MATRIZ DE CORRELACION DE RANGOS DE SPEARMAN (rho) - VARIABLES DEL MODELO (n = 125)")
print("=" * 95)
print(matriz_spearman.to_string())

# 5. Calcular valores p para todos los pares entre las 7 variables del modelo (Tabla 9)
pares_modelo = []
for i in range(len(VARS_MODELO)):
    for j in range(i + 1, len(VARS_MODELO)):
        v1 = VARS_MODELO[i]
        v2 = VARS_MODELO[j]
        r_p, p_p = stats.pearsonr(df[v1], df[v2])
        r_s, p_s = stats.spearmanr(df[v1], df[v2])
        pares_modelo.append({
            "variable_1": v1,
            "variable_2": v2,
            "r_pearson": round(float(r_p), 4),
            "valor_p_pearson": float(f"{p_p:.4e}"),
            "significativa_pearson_005": "SI" if p_p < 0.05 else "NO",
            "rho_spearman": round(float(r_s), 4),
            "valor_p_spearman": float(f"{p_s:.4e}"),
            "significativa_spearman_005": "SI" if p_s < 0.05 else "NO",
            "diferencia_pearson_menos_spearman": round(float(r_p - r_s), 4),
            "abs_r_pearson": round(abs(float(r_p)), 4)
        })

df_tabla_09 = pd.DataFrame(pares_modelo).sort_values(by="abs_r_pearson", ascending=False)
df_tabla_09_guardar = df_tabla_09.drop(columns=["abs_r_pearson"])
df_tabla_09_guardar.to_csv(ARCHIVO_TABLA_09, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 9: PARES DE CORRELACION Y VALORES P ENTRE LAS VARIABLES DEL MODELO (ORDENADOS POR |r|)")
print("=" * 95)
print(df_tabla_09_guardar.to_string(index=False))

# 6. Tabla aparte (Tabla 10): Correlaciones de las variables derivadas/excluidas contra las del modelo
filas_derivadas = []
for v_der in VARS_DERIVADAS:
    for v_mod in VARS_MODELO + VARS_DERIVADAS:
        if v_der == v_mod:
            continue
        r_p, p_p = stats.pearsonr(df[v_der], df[v_mod])
        r_s, p_s = stats.spearmanr(df[v_der], df[v_mod])
        filas_derivadas.append({
            "variable_derivada_o_contexto": v_der,
            "variable_comparada": v_mod,
            "r_pearson": round(float(r_p), 4),
            "valor_p_pearson": float(f"{p_p:.4e}"),
            "rho_spearman": round(float(r_s), 4),
            "valor_p_spearman": float(f"{p_s:.4e}"),
            "abs_r_pearson": round(abs(float(r_p)), 4)
        })

df_tabla_10 = pd.DataFrame(filas_derivadas).sort_values(by="abs_r_pearson", ascending=False)
df_tabla_10.drop(columns=["abs_r_pearson"]).to_csv(ARCHIVO_TABLA_10, index=False, encoding="utf-8-sig")

# 7. Tabla 11: Lista unica de pares con |r| >= 0.70 (o |rho| >= 0.70) en toda la base
pares_fuertes = []
for i in range(len(TODAS_NUMERICAS)):
    for j in range(i + 1, len(TODAS_NUMERICAS)):
        v1 = TODAS_NUMERICAS[i]
        v2 = TODAS_NUMERICAS[j]
        r_p, p_p = stats.pearsonr(df[v1], df[v2])
        r_s, p_s = stats.spearmanr(df[v1], df[v2])
        if abs(r_p) >= 0.70 or abs(r_s) >= 0.70:
            tipo_par = (
                "Entre variables del modelo (alerta colinealidad)"
                if (v1 in VARS_MODELO and v2 in VARS_MODELO)
                else "Incluye variable derivada/definicional (circularidad)"
            )
            pares_fuertes.append({
                "tipo_par": tipo_par,
                "variable_1": v1,
                "variable_2": v2,
                "r_pearson": round(float(r_p), 4),
                "valor_p_pearson": float(f"{p_p:.4e}"),
                "rho_spearman": round(float(r_s), 4),
                "valor_p_spearman": float(f"{p_s:.4e}"),
                "abs_r": abs(float(r_p))
            })

df_tabla_11 = pd.DataFrame(pares_fuertes).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])
df_tabla_11.to_csv(ARCHIVO_TABLA_11, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 11: LISTA DE PARES CON CORRELACION FUERTE (|r| >= 0.70 O |rho| >= 0.70)")
print("=" * 95)
print(df_tabla_11.to_string(index=False))

# 8. Generar Figura 3: Mapas de calor comparativos (Pearson vs Spearman) de las 7 variables del modelo
ETIQUETAS_CORTAS = [
    "Ingreso < LPI (y)",
    "Vivienda (x1)",
    "Serv. Basicos (x2)",
    "Alimentacion (x3)",
    "Rezago Educ. (x4)",
    "Salud (x5)",
    "Seg. Social (x6)"
]

fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6.8))
matrices_graf = [
    (matriz_pearson.values, "A) Correlacion Lineal de Pearson (r)"),
    (matriz_spearman.values, "B) Correlacion de Rangos de Spearman (rho)")
]

for idx, (mat, subtitulo) in enumerate(matrices_graf):
    ax = axes3[idx]
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(ETIQUETAS_CORTAS)))
    ax.set_yticks(range(len(ETIQUETAS_CORTAS)))
    ax.set_xticklabels(ETIQUETAS_CORTAS, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ETIQUETAS_CORTAS, fontsize=9)
    ax.set_title(subtitulo, fontsize=11, fontweight="bold", pad=10)

    for f in range(len(ETIQUETAS_CORTAS)):
        for c in range(len(ETIQUETAS_CORTAS)):
            valor = mat[f, c]
            color_txt = "white" if abs(valor) > 0.62 else "black"
            ax.text(
                c, f, f"{valor:.2f}",
                ha="center", va="center",
                color=color_txt, fontsize=8.5, fontweight="bold"
            )

fig3.suptitle(
    "Figura 3: Mapas de calor de las matrices de correlacion de Pearson y Spearman (n = 125 municipios)",
    fontsize=13, fontweight="bold", y=0.98
)
plt.tight_layout(rect=[0, 0.03, 0.92, 0.93])
cbar_ax = fig3.add_axes([0.93, 0.15, 0.018, 0.72])
fig3.colorbar(im, cax=cbar_ax, label="Coeficiente de correlacion (-1 a +1)")
fig3.savefig(ARCHIVO_FIGURA_03, dpi=300)
plt.close(fig3)

# 9. Generar Figura 4: Diagramas de dispersion de y contra cada una de las 6 carencias candidatas (x1 a x6)
CARENCIAS_X = [
    ("carencia_vivienda_porcentaje", "x1: Carencia por calidad y espacios de vivienda (%)"),
    ("carencia_alimentacion_porcentaje", "x2: Carencia por acceso a la alimentacion (%)"),
    ("carencia_servicios_basicos_porcentaje", "x3: Carencia por servicios basicos en vivienda (%)"),
    ("rezago_educativo_porcentaje", "x4: Rezago educativo (%)"),
    ("carencia_seguridad_social_porcentaje", "x5: Carencia por acceso a seguridad social (%)"),
    ("carencia_salud_porcentaje", "x6: Carencia por acceso a servicios de salud (%)")
]

fig4, axes4 = plt.subplots(2, 3, figsize=(15.5, 9.8))
axes4 = axes4.flatten()
y_serie = df["ingreso_inferior_lpi_porcentaje"]

for idx, (col_x, nombre_eje_x) in enumerate(CARENCIAS_X):
    ax = axes4[idx]
    x_serie = df[col_x]
    r_p, p_p = stats.pearsonr(x_serie, y_serie)
    r_s, p_s = stats.spearmanr(x_serie, y_serie)

    ax.scatter(x_serie, y_serie, color="#2b5c8f", edgecolor="black", alpha=0.78, s=40, label="Municipios (n=125)")

    # Recta de tendencia lineal por minimos cuadrados
    pendiente, intercepto = np.polyfit(x_serie, y_serie, 1)
    x_recta = np.linspace(x_serie.min(), x_serie.max(), 100)
    ax.plot(x_recta, pendiente * x_recta + intercepto, color="#d62728", linewidth=2, label="Tendencia lineal")

    # Etiquetar Mezquitic para visualizar su efecto en la correlacion
    fila_mez = df[df["municipio"] == "Mezquitic"]
    if not fila_mez.empty:
        xm = float(fila_mez[col_x].iloc[0])
        ym = float(fila_mez["ingreso_inferior_lpi_porcentaje"].iloc[0])
        ax.annotate(
            "Mezquitic",
            xy=(xm, ym),
            xytext=(-10, -14),
            textcoords="offset points",
            ha="right",
            fontsize=8,
            fontweight="bold",
            color="#b22222",
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=0.8)
        )

    ax.set_title(
        f"{nombre_eje_x}\nPearson r = {r_p:.4f} (p = {p_p:.2e}) | Spearman rho = {r_s:.4f}",
        fontsize=9.5, fontweight="bold"
    )
    ax.set_xlabel(nombre_eje_x, fontsize=8.5)
    ax.set_ylabel("Ingreso inferior a LPI (%) [y]", fontsize=8.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=7.5, loc="lower right")

fig4.suptitle(
    "Figura 4: Diagramas de dispersion de ingreso_inferior_lpi_porcentaje (y) contra las 6 carencias sociales (x)",
    fontsize=13, fontweight="bold", y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig4.savefig(ARCHIVO_FIGURA_04, dpi=300)
plt.close(fig4)

print("\n" + "=" * 95)
print("ARCHIVOS GUARDADOS EN 03_resultados:")
print(f"1) {ARCHIVO_TABLA_07}")
print(f"2) {ARCHIVO_TABLA_08}")
print(f"3) {ARCHIVO_TABLA_09}")
print(f"4) {ARCHIVO_TABLA_10}")
print(f"5) {ARCHIVO_TABLA_11}")
print(f"6) {ARCHIVO_FIGURA_03} (Figura 3: mapas de calor de correlaciones)")
print(f"7) {ARCHIVO_FIGURA_04} (Figura 4: diagramas de dispersion de y contra cada x candidata)")
print("=" * 95)