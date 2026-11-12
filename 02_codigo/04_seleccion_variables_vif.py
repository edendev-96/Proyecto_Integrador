import os
import sys
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from dotenv import load_dotenv

# 1. Definir rutas de trabajo y archivos de salida
RUTA_BASE = r"C:\Users\cadsi\Avance_Mate"
RUTA_ENV = os.path.join(RUTA_BASE, ".env")
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "03_resultados")

ARCHIVO_TABLA_12 = os.path.join(RUTA_RESULTADOS, "tabla_12_diagnostico_modelo_anterior.csv")
ARCHIVO_TABLA_13 = os.path.join(RUTA_RESULTADOS, "tabla_13_variables_excluidas_justificacion.csv")
ARCHIVO_TABLA_14 = os.path.join(RUTA_RESULTADOS, "tabla_14_comparacion_tres_opciones_y_cv5.csv")
ARCHIVO_TABLA_15 = os.path.join(RUTA_RESULTADOS, "tabla_15_vif_modelo_final.csv")
ARCHIVO_FIGURA_05 = os.path.join(RUTA_RESULTADOS, "figura_05_vif_y_comparacion_modelos_y.png")

# 2. Cargar archivo .env con override=True (misma configuracion exitosa de Parte 3)
load_dotenv(RUTA_ENV, override=True)

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
    df = pd.DataFrame(cursor.fetchall())
    cursor.close()
    conexion.close()
except Exception as e:
    print("ERROR al leer datos_limpios desde MySQL:", e)
    sys.exit(1)

# Convertir columnas numericas a float
COLS_TEXTO = ["clave_municipio", "clave_entidad", "entidad_federativa", "municipio",
              "tipo_municipio_poblacion", "nivel_intensidad_carencias"]
for col in df.columns:
    if col not in COLS_TEXTO:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Demostracion numerica del problema del modelo anterior (Circularidad y VIF)
VARS_7_ANTERIOR = [
    "ingreso_inferior_lpi_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje",
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje"
]
VARS_4_IND_ANTERIOR = [
    "ind_urbano_grande",
    "ind_semiurbano_medio",
    "ind_intensidad_alta",
    "ind_intensidad_moderada"
]
VARS_11_ANTERIOR = VARS_7_ANTERIOR + VARS_4_IND_ANTERIOR

y_anterior = df["pobreza_porcentaje"]
X_ant_7_const = sm.add_constant(df[VARS_7_ANTERIOR])
X_ant_11_const = sm.add_constant(df[VARS_11_ANTERIOR])

modelo_ant_11 = sm.OLS(y_anterior, X_ant_11_const).fit()

# Calcular VIF en el bloque de 7 variables numericas y en el bloque completo de 11 variables
mapa_vif_7 = {}
for i, col in enumerate(X_ant_7_const.columns):
    if col != "const":
        mapa_vif_7[col] = round(float(variance_inflation_factor(X_ant_7_const.values, i)), 4)

filas_diag_anterior = []
for i, col in enumerate(X_ant_11_const.columns):
    if col != "const":
        vif_11 = float(variance_inflation_factor(X_ant_11_const.values, i))
        coef = float(modelo_ant_11.params[col])
        pval = float(modelo_ant_11.pvalues[col])
        problema = "Ninguno (se conserva en X)"
        if col == "ingreso_inferior_lpi_porcentaje":
            problema = "Circularidad definicional con pobreza_porcentaje (pasa a ser Y)"
        elif col in ["ind_intensidad_alta", "ind_intensidad_moderada"]:
            problema = "Fuga de informacion (derivada de carencias_promedio)"

        filas_diag_anterior.append({
            "variable_modelo_anterior": col,
            "coeficiente_ols_anterior": round(coef, 4),
            "valor_p_anterior": float(f"{pval:.4e}"),
            "vif_bloque_7_numericas": mapa_vif_7.get(col, np.nan),
            "vif_modelo_anterior_11_vars": round(vif_11, 4),
            "diagnostico_metodologico": problema
        })

df_tabla_12 = pd.DataFrame(filas_diag_anterior)
df_tabla_12.to_csv(ARCHIVO_TABLA_12, index=False, encoding="utf-8-sig")

print("=" * 100)
print("DEMOSTRACION NUMERICA DEL PROBLEMA DEL MODELO ANTERIOR (y = pobreza_porcentaje)")
print(f"R2 modelo anterior (11 variables) = {modelo_ant_11.rsquared:.4f} | "
      f"R2 ajustado = {modelo_ant_11.rsquared_adj:.4f} | F = {modelo_ant_11.fvalue:.4f}")
print("=" * 100)
print(df_tabla_12.to_string(index=False))

# 4. Tabla 13: Lista completa de variables excluidas de los modelos y justificacion
variables_excluidas = [
    ("pobreza_porcentaje", "Variable objetivo anterior", "Circularidad definicional",
     "CONEVAL la define combinando ingreso_inferior_lpi_porcentaje y las 6 carencias sociales."),
    ("pobreza_personas", "Numerica derivada", "Fuga de informacion y escala poblacional",
     "Se calcula multiplicando pobreza_porcentaje por poblacion_2020 (r = 0.9882 con poblacion)."),
    ("pobreza_extrema_porcentaje", "Subconjunto definicional", "Circularidad parcial y asimetria extrema",
     "CONEVAL la define con ingreso inferior a LPEI y 3 o mas carencias; tiene asimetria = 5.0700."),
    ("carencias_promedio", "Resumen lineal de carencias", "Fuga de informacion / colinealidad",
     "Es el promedio directo de las 6 carencias sociales que ya entran individualmente en X."),
    ("nivel_intensidad_carencias", "Categorica derivada", "Fuga de informacion",
     "Categorizacion directa de carencias_promedio en Baja, Moderada y Alta."),
    ("ind_intensidad_alta", "Indicadora binaria (0/1)", "Fuga de informacion (VIF = 4.3241)",
     "Indicadora derivada de carencias_promedio; duplica la informacion de las 6 carencias."),
    ("ind_intensidad_moderada", "Indicadora binaria (0/1)", "Fuga de informacion (VIF = 2.5308)",
     "Indicadora derivada de carencias_promedio; duplica la informacion de las 6 carencias."),
    ("poblacion_2020", "Demografica de escala absoluta", "Asimetria extrema (5.3490) y redundancia",
     "No es porcentaje; su efecto de escala entra al modelo mediante tipo_municipio_poblacion."),
    ("tipo_municipio_poblacion", "Categorica en texto (3 niveles)", "Codificada en variables indicadoras",
     "Entra al modelo mediante ind_urbano_grande e ind_semiurbano_medio (referencia: Rural_Pequeno)."),
    ("clave_municipio", "Identificador INEGI CHAR(5)", "Identificador administrativo sin valor predictivo",
     "Llave primaria de la tabla datos_limpios (14001 a 14125)."),
    ("clave_entidad", "Identificador estatal constante", "Constante sin varianza",
     "Vale siempre 14 (Jalisco) en los 125 registros."),
    ("entidad_federativa", "Texto constante", "Constante sin varianza",
     "Vale siempre 'Jalisco' en los 125 registros."),
    ("municipio", "Texto nominal", "Identificador de etiqueta municipal",
     "Nombre oficial del municipio; se usa para identificar residuos y valores extremos, no como X.")
]

df_tabla_13 = pd.DataFrame(
    variables_excluidas,
    columns=["variable_excluida", "tipo_columna", "motivo_principal", "justificacion_tecnica"]
)
df_tabla_13.to_csv(ARCHIVO_TABLA_13, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("TABLA 13: VARIABLES EXCLUIDAS DE LOS MODELOS Y JUSTIFICACION TECNICA")
print("=" * 100)
print(df_tabla_13.to_string(index=False))

# 5. Tabla 14: Comparar 3 opciones de variable objetivo Y con las mismas 8 variables X
VARS_6_CARENCIAS = [
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje"
]
VARS_8_MODELO_FINAL = VARS_6_CARENCIAS + ["ind_urbano_grande", "ind_semiurbano_medio"]

X_8_valores = df[VARS_8_MODELO_FINAL].astype(float)
X_8_const = sm.add_constant(X_8_valores)

opciones_y = [
    ("Opcion 1 (Elegida): ingreso_inferior_lpi_porcentaje",
     "ingreso_inferior_lpi_porcentaje",
     "SE CONFIRMA: Mide vulnerabilidad de ingreso sin circularidad definicional con las carencias."),
    ("Opcion 2 (Alternativa A): pobreza_porcentaje (sin ingreso en X)",
     "pobreza_porcentaje",
     "SE DESCARTA: Conserva circularidad parcial porque las 6 carencias forman parte de su definicion."),
    ("Opcion 3 (Alternativa B): pobreza_extrema_porcentaje",
     "pobreza_extrema_porcentaje",
     "SE DESCARTA: Asimetria extrema (5.07) y circularidad con el umbral de 3 o mas carencias.")
]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
filas_comp_y = []

for nombre_op, col_y, decision in opciones_y:
    y_op = df[col_y].astype(float)
    mod_ols = sm.OLS(y_op, X_8_const).fit()
    pred_ols = mod_ols.predict(X_8_const)
    rmse_ols = float(np.sqrt(mean_squared_error(y_op, pred_ols)))

    lr_sk = LinearRegression()
    r2_cv_grupos = cross_val_score(lr_sk, X_8_valores, y_op, cv=kf, scoring="r2")
    r2_cv_media = float(r2_cv_grupos.mean())
    r2_cv_std = float(r2_cv_grupos.std(ddof=1))

    filas_comp_y.append({
        "opcion_variable_y": nombre_op,
        "columna_y": col_y,
        "R2": round(float(mod_ols.rsquared), 4),
        "R2_ajustado": round(float(mod_ols.rsquared_adj), 4),
        "RMSE": round(rmse_ols, 4),
        "R2_CV_5grupos_semilla42": round(r2_cv_media, 4),
        "desv_est_R2_CV": round(r2_cv_std, 4),
        "diferencia_R2_menos_CV": round(float(mod_ols.rsquared - r2_cv_media), 4),
        "decision_metodologica": decision
    })

df_tabla_14 = pd.DataFrame(filas_comp_y)
df_tabla_14.to_csv(ARCHIVO_TABLA_14, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("TABLA 14: COMPARACION DE 3 OPCIONES DE Y CON LAS MISMAS 8 VARIABLES X (K-FOLD = 5, SEMILLA = 42)")
print("=" * 100)
cols_mostrar_14 = ["opcion_variable_y", "R2", "R2_ajustado", "RMSE",
                   "R2_CV_5grupos_semilla42", "diferencia_R2_menos_CV"]
print(df_tabla_14[cols_mostrar_14].to_string(index=False))
for _, r in df_tabla_14.iterrows():
    print(f"  -> {r['columna_y']}: {r['decision_metodologica']}")

# 6. Tabla 15: VIF del modelo final (comparando las 6 carencias solas vs las 8 variables finales)
X_6_const = sm.add_constant(df[VARS_6_CARENCIAS].astype(float))
mapa_vif_6 = {}
for i, col in enumerate(X_6_const.columns):
    if col != "const":
        mapa_vif_6[col] = round(float(variance_inflation_factor(X_6_const.values, i)), 4)

filas_vif_final = []
for i, col in enumerate(X_8_const.columns):
    if col != "const":
        vif_8 = float(variance_inflation_factor(X_8_const.values, i))
        tol_8 = 1.0 / vif_8
        r2_aux = 1.0 - tol_8
        filas_vif_final.append({
            "variable_explicativa_x": col,
            "tipo_variable": "Carencia social (%)" if col in VARS_6_CARENCIAS else "Indicadora binaria (0/1)",
            "vif_solo_6_carencias": mapa_vif_6.get(col, np.nan),
            "vif_modelo_final_8_vars": round(vif_8, 4),
            "tolerancia_1_sobre_vif": round(tol_8, 4),
            "R2_auxiliar_con_demas_x": round(r2_aux, 4),
            "criterio_vif_menor_5": "CUMPLE (VIF < 5)" if vif_8 < 5.0 else "ALERTA (VIF >= 5)"
        })

df_tabla_15 = pd.DataFrame(filas_vif_final).sort_values(by="vif_modelo_final_8_vars", ascending=False)
df_tabla_15.to_csv(ARCHIVO_TABLA_15, index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("TABLA 15: FACTOR DE INFLACION DE LA VARIANZA (VIF) DEL MODELO FINAL (8 VARIABLES X)")
print("=" * 100)
print(df_tabla_15.to_string(index=False))

# 7. Generar Figura 5: Panel A (VIF del modelo final) y Panel B (Comparacion de R2 y R2_CV de las 3 opciones de Y)
fig5, axes5 = plt.subplots(1, 2, figsize=(15.5, 6.2))

# Panel A: VIF horizontal
ax_vif = axes5[0]
df_vif_plot = df_tabla_15.sort_values(by="vif_modelo_final_8_vars", ascending=True)
nombres_cortos_vif = {
    "carencia_vivienda_porcentaje": "Carencia vivienda (x1)",
    "carencia_servicios_basicos_porcentaje": "Servicios basicos (x2)",
    "carencia_seguridad_social_porcentaje": "Seguridad social (x6)",
    "ind_urbano_grande": "Ind. Urbano Grande (x7)",
    "rezago_educativo_porcentaje": "Rezago educativo (x4)",
    "ind_semiurbano_medio": "Ind. Semiurbano Medio (x8)",
    "carencia_alimentacion_porcentaje": "Carencia alimentacion (x3)",
    "carencia_salud_porcentaje": "Carencia salud (x5)"
}
etiquetas_y_vif = [nombres_cortos_vif[c] for c in df_vif_plot["variable_explicativa_x"]]
valores_x_vif = df_vif_plot["vif_modelo_final_8_vars"].values

barras_vif = ax_vif.barh(etiquetas_y_vif, valores_x_vif, color="#3b6998", edgecolor="black", alpha=0.88)
ax_vif.axvline(5.0, color="#d62728", linestyle="--", linewidth=2, label="Umbral de alerta (VIF = 5.0)")
ax_vif.axvline(1.0, color="#2ca02c", linestyle=":", linewidth=1.5, label="Sin colinealidad (VIF = 1.0)")
ax_vif.set_xlim(0, 5.8)
ax_vif.set_xlabel("Factor de Inflacion de la Varianza (VIF)", fontsize=9.5)
ax_vif.set_title("A) VIF de las 8 variables explicativas del modelo final", fontsize=11, fontweight="bold")
ax_vif.grid(axis="x", linestyle=":", alpha=0.6)
ax_vif.legend(fontsize=8.5, loc="lower right")

for barra in barras_vif:
    ancho = barra.get_width()
    ax_vif.text(ancho + 0.08, barra.get_y() + barra.get_height() / 2.0,
                f"{ancho:.2f}", va="center", ha="left", fontsize=8.5, fontweight="bold")

# Panel B: Comparacion de las 3 opciones de Y
ax_comp = axes5[1]
etiquetas_op = [
    "Ingreso < LPI\n(Elegida: sin circularidad)",
    "Pobreza general\n(Circularidad parcial)",
    "Pobreza extrema\n(Sesgada / circular)"
]
x_pos = np.arange(len(etiquetas_op))
ancho_b = 0.26

vals_r2 = df_tabla_14["R2"].values
vals_r2_adj = df_tabla_14["R2_ajustado"].values
vals_r2_cv = df_tabla_14["R2_CV_5grupos_semilla42"].values

b1 = ax_comp.bar(x_pos - ancho_b, vals_r2, width=ancho_b, color="#2b5c8f", edgecolor="black", label="R2 entrenamiento")
b2 = ax_comp.bar(x_pos, vals_r2_adj, width=ancho_b, color="#6baed6", edgecolor="black", label="R2 ajustado")
b3 = ax_comp.bar(x_pos + ancho_b, vals_r2_cv, width=ancho_b, color="#ff7f0e", edgecolor="black", label="R2 Validacion Cruzada (K=5)")

ax_comp.set_xticks(x_pos)
ax_comp.set_xticklabels(etiquetas_op, fontsize=9)
ax_comp.set_ylim(0, 1.05)
ax_comp.set_ylabel("Coeficiente de determinacion (R2)", fontsize=9.5)
ax_comp.set_