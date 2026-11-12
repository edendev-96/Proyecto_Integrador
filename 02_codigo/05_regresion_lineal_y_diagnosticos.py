# Archivo: 02_codigo/05_regresion_lineal_y_diagnosticos.py
import os
import sys
import getpass
import numpy as np
import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression

# 1. Definir rutas de archivos de salida en 03_resultados
RUTA_BASE = r"C:\Users\cadsi\Avance_Mate"
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "03_resultados")

ARCHIVO_TABLA_13 = os.path.join(RUTA_RESULTADOS, "tabla_13_comparacion_variables_y.csv")
ARCHIVO_TABLA_14 = os.path.join(RUTA_RESULTADOS, "tabla_14_vif_modelo_final_completo.csv")
ARCHIVO_TABLA_15 = os.path.join(RUTA_RESULTADOS, "tabla_15_coeficientes_regresion_simple.csv")
ARCHIVO_TABLA_16 = os.path.join(RUTA_RESULTADOS, "tabla_16_coeficientes_regresion_multiple_ols_y_hc3.csv")
ARCHIVO_TABLA_17 = os.path.join(RUTA_RESULTADOS, "tabla_17_comprobacion_ecuacion_normal_matrices.csv")
ARCHIVO_TABLA_18 = os.path.join(RUTA_RESULTADOS, "tabla_18_diagnostico_residuos_modelos.csv")
ARCHIVO_TABLA_19 = os.path.join(RUTA_RESULTADOS, "tabla_19_municipios_influyentes_distancia_cook.csv")
ARCHIVO_TABLA_20 = os.path.join(RUTA_RESULTADOS, "tabla_20_comparativa_general_modelos.csv")

ARCHIVO_FIGURA_05 = os.path.join(RUTA_RESULTADOS, "figura_05_regresion_simple_recta.png")
ARCHIVO_FIGURA_06 = os.path.join(RUTA_RESULTADOS, "figura_06_diagnostico_residuos_multiple.png")

### 2. Conectar a MySQL de forma interactiva y segura (mismo metodo exitoso de la Parte 4)
### print("=" * 80)
### password_segura = getpass.getpass("Contrasena de MySQL para root (no se vera al teclear): ")
### print("=" * 80)

try:
    conexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="B3aM0N$t3R#W1n!",
        database="avance_bd",
        port=3307
    )
    cursor = conexion.cursor(dictionary=True)
    cursor.execute("SELECT * FROM datos_limpios ORDER BY clave_municipio ASC;")
    df = pd.DataFrame(cursor.fetchall())
    cursor.close()
    conexion.close()
    print(f"Conexion exitosa a MySQL -> {df.shape[0]} municipios y {df.shape[1]} columnas cargadas.")
except Exception as e:
    print("ERROR al conectar a MySQL:", e)
    sys.exit(1)

for col in df.columns:
    if col not in ["entidad_federativa", "municipio", "tipo_municipio_poblacion", "nivel_intensidad_carencias"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 3. Definir variables del estudio
VAR_Y = "ingreso_inferior_lpi_porcentaje"
VAR_X_SIMPLE = "carencia_vivienda_porcentaje"
VARS_X_MULTIPLE = [
    "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje",
    "carencia_alimentacion_porcentaje",
    "rezago_educativo_porcentaje",
    "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje",
    "ind_urbano_grande",
    "ind_semiurbano_medio"
]

y = df[VAR_Y].astype(float)
X_simple = sm.add_constant(df[[VAR_X_SIMPLE]].astype(float))
X_multiple = sm.add_constant(df[VARS_X_MULTIPLE].astype(float))

# 4. Actualizar Tabla 13 con todas las metricas pedidas (R2, R2 ajustado, RMSE y R2_CV con semilla 42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
opciones_y = [
    ("pobreza_porcentaje (Circularidad parcial)", "pobreza_porcentaje"),
    ("pobreza_extrema_porcentaje (Sesgada y circular)", "pobreza_extrema_porcentaje"),
    ("ingreso_inferior_lpi_porcentaje (Elegida)", "ingreso_inferior_lpi_porcentaje")
]
filas_t13 = []
for etiqueta, col_y in opciones_y:
    y_op = df[col_y].astype(float)
    mod_op = sm.OLS(y_op, X_multiple).fit()
    rmse_op = float(np.sqrt(mean_squared_error(y_op, mod_op.fittedvalues)))
    cv_op = float(cross_val_score(LinearRegression(), df[VARS_X_MULTIPLE], y_op, cv=kf, scoring="r2").mean())
    filas_t13.append({
        "variable_objetivo_y": etiqueta,
        "r2": round(float(mod_op.rsquared), 4),
        "r2_ajustado": round(float(mod_op.rsquared_adj), 4),
        "rmse": round(rmse_op, 4),
        "r2_cv_5fold_semilla42": round(cv_op, 4)
    })
df_tabla_13 = pd.DataFrame(filas_t13)
df_tabla_13.to_csv(ARCHIVO_TABLA_13, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 13 (COMPLETA): COMPARACION DE 3 OPCIONES DE Y (R2, R2 AJUSTADO, RMSE Y CV 5-FOLD)")
print("=" * 95)
print(df_tabla_13.to_string(index=False))

# 5. Calcular Tabla 14: VIF del modelo final completo (6 carencias + 2 indicadoras)
filas_vif_final = []
for i, col in enumerate(X_multiple.columns):
    if col != "const":
        val_vif = float(variance_inflation_factor(X_multiple.values, i))
        filas_vif_final.append({
            "variable_explicativa_x": col,
            "vif_modelo_final": round(val_vif, 4),
            "criterio_menor_a_5": "Estable (< 5)" if val_vif < 5.0 else "Alerta (>= 5)"
        })
df_tabla_14 = pd.DataFrame(filas_vif_final)
df_tabla_14.to_csv(ARCHIVO_TABLA_14, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 14: VIF DEL MODELO DE REGRESION MULTIPLE FINAL (6 CARENCIAS + 2 INDICADORAS)")
print("=" * 95)
print(df_tabla_14.to_string(index=False))

# 6. Ajustar Regresion Lineal Simple (Tabla 15)
mod_simple = sm.OLS(y, X_simple).fit()
mod_simple_hc3 = mod_simple.get_robustcov_results(cov_type="HC3")
ic_simple = mod_simple.conf_int(alpha=0.05)

filas_simple = []
for idx, nombre_col in enumerate(X_simple.columns):
    filas_simple.append({
        "termino": nombre_col,
        "coeficiente_beta": round(float(mod_simple.params.iloc[idx]), 4),
        "error_estandar_ols": round(float(mod_simple.bse.iloc[idx]), 4),
        "estadistico_t_ols": round(float(mod_simple.tvalues.iloc[idx]), 4),
        "valor_p_ols": float(f"{mod_simple.pvalues.iloc[idx]:.4e}"),
        "ic95_inf_ols": round(float(ic_simple.iloc[idx, 0]), 4),
        "ic95_sup_ols": round(float(ic_simple.iloc[idx, 1]), 4),
        "error_estandar_hc3": round(float(mod_simple_hc3.bse[idx]), 4),
        "valor_p_hc3": float(f"{mod_simple_hc3.pvalues[idx]:.4e}")
    })
df_tabla_15 = pd.DataFrame(filas_simple)
df_tabla_15.to_csv(ARCHIVO_TABLA_15, index=False, encoding="utf-8-sig")

b0_s = float(mod_simple.params.iloc[0])
b1_s = float(mod_simple.params.iloc[1])
rmse_s = float(np.sqrt(mean_squared_error(y, mod_simple.fittedvalues)))
mae_s = float(mean_absolute_error(y, mod_simple.fittedvalues))

print("\n" + "=" * 95)
print("TABLA 15: REGRESION LINEAL SIMPLE (y = ingreso_inferior_lpi_porcentaje, x = vivienda)")
print("=" * 95)
print(f"Ecuacion: {VAR_Y} = {b0_s:.4f} + ({b1_s:.4f} * {VAR_X_SIMPLE})")
print(f"R2 = {mod_simple.rsquared:.4f} | R2 ajustado = {mod_simple.rsquared_adj:.4f} | "
      f"F = {mod_simple.fvalue:.4f} (p = {mod_simple.f_pvalue:.4e}) | RMSE = {rmse_s:.4f} | MAE = {mae_s:.4f}")
print(df_tabla_15.to_string(index=False))

# 7. Ajustar Regresion Lineal Multiple con OLS clasico y correccion HC3 (Tabla 16)
mod_mult = sm.OLS(y, X_multiple).fit()
mod_mult_hc3 = mod_mult.get_robustcov_results(cov_type="HC3")
ic_mult = mod_mult.conf_int(alpha=0.05)
ic_mult_hc3 = mod_mult_hc3.conf_int(alpha=0.05)

filas_mult = []
for idx, nombre_col in enumerate(X_multiple.columns):
    p_ols = float(mod_mult.pvalues.iloc[idx])
    p_hc3 = float(mod_mult_hc3.pvalues[idx])
    filas_mult.append({
        "termino": nombre_col,
        "coef_beta": round(float(mod_mult.params.iloc[idx]), 4),
        "ee_ols": round(float(mod_mult.bse.iloc[idx]), 4),
        "t_ols": round(float(mod_mult.tvalues.iloc[idx]), 4),
        "p_ols": float(f"{p_ols:.4e}"),
        "ic95_inf_ols": round(float(ic_mult.iloc[idx, 0]), 4),
        "ic95_sup_ols": round(float(ic_mult.iloc[idx, 1]), 4),
        "ee_hc3": round(float(mod_mult_hc3.bse[idx]), 4),
        "t_hc3": round(float(mod_mult_hc3.tvalues[idx]), 4),
        "p_hc3": float(f"{p_hc3:.4e}"),
        "ic95_inf_hc3": round(float(ic_mult_hc3[idx, 0]), 4),
        "ic95_sup_hc3": round(float(ic_mult_hc3[idx, 1]), 4),
        "sig_005_hc3": "SI" if p_hc3 < 0.05 else "NO"
    })
df_tabla_16 = pd.DataFrame(filas_mult)
df_tabla_16.to_csv(ARCHIVO_TABLA_16, index=False, encoding="utf-8-sig")

rmse_m = float(np.sqrt(mean_squared_error(y, mod_mult.fittedvalues)))
mae_m = float(mean_absolute_error(y, mod_mult.fittedvalues))

print("\n" + "=" * 95)
print("TABLA 16: REGRESION LINEAL MULTIPLE (OLS CLASICO Y ERRORES ROBUSTOS HC3)")
print("=" * 95)
print(f"R2 = {mod_mult.rsquared:.4f} | R2 ajustado = {mod_mult.rsquared_adj:.4f} | "
      f"F_OLS = {mod_mult.fvalue:.4f} (p = {mod_mult.f_pvalue:.4e}) | "
      f"F_HC3 = {float(mod_mult_hc3.fvalue):.4f} (p = {float(mod_mult_hc3.f_pvalue):.4e}) | "
      f"RMSE = {rmse_m:.4f} | MAE = {mae_m:.4f}")
print(df_tabla_16.to_string(index=False))

# 8. Comprobacion de coeficientes con Algebra de Matrices: beta = (X^T X)^(-1) X^T y (Tabla 17)
X_mat = X_multiple.values.astype(float)
y_vec = y.values.astype(float)
beta_numpy = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ y_vec)

filas_matrices = []
for idx, nombre_col in enumerate(X_multiple.columns):
    b_np = float(beta_numpy[idx])
    b_sm = float(mod_mult.params.iloc[idx])
    filas_matrices.append({
        "termino": nombre_col,
        "beta_ecuacion_normal_numpy": round(b_np, 8),
        "beta_statsmodels_ols": round(b_sm, 8),
        "diferencia_absoluta": float(f"{abs(b_np - b_sm):.4e}")
    })
df_tabla_17 = pd.DataFrame(filas_matrices)
df_tabla_17.to_csv(ARCHIVO_TABLA_17, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 17: COMPROBACION CON ALGEBRA DE MATRICES [beta = (X^T X)^(-1) X^T y]")
print("=" * 95)
print(df_tabla_17.to_string(index=False))

# 9. Diagnostico de residuos: media, Shapiro-Wilk, Breusch-Pagan y Distancia de Cook (Tablas 18 y 19)
def diagnosticar(nombre_mod, mod_ols):
    res = mod_ols.resid
    sw_w, sw_p = stats.shapiro(res)
    bp_lm, bp_p, _, _ = het_breuschpagan(res, mod_ols.model.exog)
    cooks = mod_ols.get_influence().cooks_distance[0]
    umbral = 4.0 / len(res)
    return {
        "modelo": nombre_mod,
        "media_residuos": round(float(np.mean(res)), 6),
        "desv_est_residuos": round(float(np.std(res, ddof=1)), 4),
        "shapiro_W": round(float(sw_w), 4),
        "shapiro_p": float(f"{sw_p:.4e}"),
        "normalidad_005": "SI" if sw_p >= 0.05 else "NO (p<0.05)",
        "breusch_pagan_LM": round(float(bp_lm), 4),
        "breusch_pagan_p": float(f"{bp_p:.4e}"),
        "homocedasticidad_005": "SI" if bp_p >= 0.05 else "NO (usar HC3)",
        "umbral_cook_4_n": round(umbral, 4),
        "n_municipios_cook_alto": int(np.sum(cooks > umbral)),
        "max_cook": round(float(np.max(cooks)), 4)
    }

df_tabla_18 = pd.DataFrame([
    diagnosticar("Regresion Simple (x = vivienda)", mod_simple),
    diagnosticar("Regresion Multiple (8 variables)", mod_mult)
])
df_tabla_18.to_csv(ARCHIVO_TABLA_18, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 18: DIAGNOSTICO DE RESIDUOS (SHAPIRO-WILK, BREUSCH-PAGAN Y DISTANCIA DE COOK)")
print("=" * 95)
print(df_tabla_18.to_string(index=False))

cooks_mult = mod_mult.get_influence().cooks_distance[0]
df_cook = pd.DataFrame({
    "clave_municipio": df["clave_municipio"],
    "municipio": df["municipio"],
    "y_real_lpi": df[VAR_Y].round(2),
    "y_ajustado": mod_mult.fittedvalues.round(4),
    "residuo": mod_mult.resid.round(4),
    "distancia_cook": np.round(cooks_mult, 4),
    "supera_4_n": np.where(cooks_mult > (4.0 / len(df)), "SI", "NO")
}).sort_values(by="distancia_cook", ascending=False)
df_cook.head(10).to_csv(ARCHIVO_TABLA_19, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 19: TOP 10 MUNICIPIOS INFLUYENTES SEGUN DISTANCIA DE COOK (MODELO MULTIPLE)")
print("=" * 95)
print(df_cook.head(10).to_string(index=False))

# 10. Tabla 20: Comparativa entre Modelo Simple, Multiple Nuevo y Modelo Anterior (Avance_BD)
cv_simple = float(cross_val_score(LinearRegression(), df[[VAR_X_SIMPLE]], y, cv=kf, scoring="r2").mean())
cv_mult = float(cross_val_score(LinearRegression(), df[VARS_X_MULTIPLE], y, cv=kf, scoring="r2").mean())

VARS_BD_ANT = [
    "ingreso_inferior_lpi_porcentaje", "rezago_educativo_porcentaje", "carencia_salud_porcentaje",
    "carencia_seguridad_social_porcentaje", "carencia_vivienda_porcentaje",
    "carencia_servicios_basicos_porcentaje", "carencia_alimentacion_porcentaje",
    "ind_urbano_grande", "ind_semiurbano_medio", "ind_intensidad_alta", "ind_intensidad_moderada"
]
y_bd = df["pobreza_porcentaje"].astype(float)
mod_bd = sm.OLS(y_bd, sm.add_constant(df[VARS_BD_ANT].astype(float))).fit()
cv_bd = float(cross_val_score(LinearRegression(), df[VARS_BD_ANT], y_bd, cv=kf, scoring="r2").mean())

df_tabla_20 = pd.DataFrame([
    {
        "modelo": "1. Simple Nuevo (y=ingreso_lpi, x=vivienda)",
        "k_vars": 1,
        "r2": round(float(mod_simple.rsquared), 4),
        "r2_ajustado": round(float(mod_simple.rsquared_adj), 4),
        "r2_cv_5fold": round(cv_simple, 4),
        "F_stat": round(float(mod_simple.fvalue), 4),
        "valor_p_F": float(f"{mod_simple.f_pvalue:.4e}"),
        "rmse": round(rmse_s, 4),
        "mae": round(mae_s, 4),
        "evaluacion_metodologica": "Valido (sin circularidad)"
    },
    {
        "modelo": "2. Multiple Nuevo (y=ingreso_lpi, 8 vars)",
        "k_vars": 8,
        "r2": round(float(mod_mult.rsquared), 4),
        "r2_ajustado": round(float(mod_mult.rsquared_adj), 4),
        "r2_cv_5fold": round(cv_mult, 4),
        "F_stat": round(float(mod_mult.fvalue), 4),
        "valor_p_F": float(f"{mod_mult.f_pvalue:.4e}"),
        "rmse": round(rmse_m, 4),
        "mae": round(mae_m, 4),
        "evaluacion_metodologica": "Valido (sin circularidad ni fuga)"
    },
    {
        "modelo": "3. Multiple Anterior BD (y=pobreza_pct, 11 vars)",
        "k_vars": 11,
        "r2": round(float(mod_bd.rsquared), 4),
        "r2_ajustado": round(float(mod_bd.rsquared_adj), 4),
        "r2_cv_5fold": round(cv_bd, 4),
        "F_stat": round(float(mod_bd.fvalue), 4),
        "valor_p_F": float(f"{mod_bd.f_pvalue:.4e}"),
        "rmse": round(float(np.sqrt(mean_squared_error(y_bd, mod_bd.fittedvalues))), 4),
        "mae": round(float(mean_absolute_error(y_bd, mod_bd.fittedvalues)), 4),
        "evaluacion_metodologica": "Descartado (circularidad definicional)"
    }
])
df_tabla_20.to_csv(ARCHIVO_TABLA_20, index=False, encoding="utf-8-sig")

print("\n" + "=" * 95)
print("TABLA 20: TABLA COMPARATIVA ENTRE MODELO SIMPLE, MULTIPLE NUEVO Y MODELO ANTERIOR")
print("=" * 95)
print(df_tabla_20.to_string(index=False))

# 11. Generar Figura 5: Grafica de Regresion Lineal Simple con recta e IC 95%
fig5, ax5 = plt.subplots(figsize=(9, 6))
x_vals = df[VAR_X_SIMPLE]
orden = np.argsort(x_vals.values)
x_ord = x_vals.iloc[orden]
pred_ic = mod_simple.get_prediction(X_simple.iloc[orden]).summary_frame(alpha=0.05)

ax5.scatter(x_vals, y, color="#2b5c8f", edgecolor="black", alpha=0.78, s=45, label="Municipios (n = 125)")
ax5.plot(
    x_ord, pred_ic["mean"], color="#d62728", linewidth=2.2,
    label=f"Recta ajustada: y = {b0_s:.4f} + {b1_s:.4f}*x (R² = {mod_simple.rsquared:.4f})"
)
ax5.fill_between(
    x_ord, pred_ic["mean_ci_lower"], pred_ic["mean_ci_upper"],
    color="#d62728", alpha=0.18, label="Intervalo de confianza al 95%"
)

for mun_nom in ["Mezquitic", "Bolaños", "Atemajac de Brizuela", "Atenguillo"]:
    fila_m = df[df["municipio"] == mun_nom]
    if not fila_m.empty:
        xm = float(fila_m[VAR_X_SIMPLE].iloc[0])
        ym = float(fila_m[VAR_Y].iloc[0])
        ax5.annotate(
            f"{mun_nom} ({xm:.1f}%, {ym:.1f}%)",
            xy=(xm, ym), xytext=(6, -10), textcoords="offset points",
            fontsize=8, fontweight="bold", color="#b22222"
        )

ax5.set_title(
    "Figura 5: Regresion lineal simple (ingreso_inferior_lpi_porcentaje vs carencia_vivienda_porcentaje)",
    fontsize=11, fontweight="bold"
)
ax5.set_xlabel("Carencia por calidad y espacios de la vivienda (%) [x]", fontsize=9.5)
ax5.set_ylabel("Poblacion con ingreso inferior a la LPI (%) [y]", fontsize=9.5)
ax5.grid(True, linestyle=":", alpha=0.6)
ax5.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
fig5.savefig(ARCHIVO_FIGURA_05, dpi=300)
plt.close(fig5)

# 12. Generar Figura 6: Diagnostico de residuos del modelo multiple (4 paneles)
fig6, axes6 = plt.subplots(2, 2, figsize=(13.5, 10))
res_m = mod_mult.resid
fit_m = mod_mult.fittedvalues

axes6[0, 0].hist(res_m, bins=15, color="#3b6998", edgecolor="black", alpha=0.82)
axes6[0, 0].axvline(0, color="#d62728", linestyle="--", linewidth=1.8, label="Media = 0.0000")
axes6[0, 0].set_title("A) Histograma de los residuos (Shapiro-Wilk p < 0.05)", fontsize=10.5, fontweight="bold")
axes6[0, 0].set_xlabel("Residuo OLS (puntos porcentuales)", fontsize=9)
axes6[0, 0].set_ylabel("Frecuencia (municipios)", fontsize=9)
axes6[0, 0].grid(axis="y", linestyle=":", alpha=0.6)
axes6[0, 0].legend(fontsize=8.5)

axes6[0, 1].scatter(fit_m, res_m, color="#2b5c8f", edgecolor="black", alpha=0.78, s=40)
axes6[0, 1].axhline(0, color="#d62728", linestyle="--", linewidth=1.8)
axes6[0, 1].set_title("B) Residuos contra valores ajustados (Breusch-Pagan p < 0.05)", fontsize=10.5, fontweight="bold")
axes6[0, 1].set_xlabel("Valores ajustados por el modelo multiple (%)", fontsize=9)
axes6[0, 1].set_ylabel("Residuos OLS (%)", fontsize=9)
axes6[0, 1].grid(True, linestyle=":", alpha=0.6)

stats.probplot(res_m, dist="norm", plot=axes6[1, 0])
axes6[1, 0].set_title("C) Grafica Q-Q de normalidad de los residuos", fontsize=10.5, fontweight="bold")
axes6[1, 0].grid(True, linestyle=":", alpha=0.6)

umbral_c = 4.0 / len(df)
axes6[1, 1].stem(range(len(cooks_mult)), cooks_mult, basefmt=" ")
axes6[1, 1].axhline(umbral_c, color="#d62728", linestyle="--", linewidth=1.8, label=f"Umbral 4/n = {umbral_c:.4f}")
for idx_m in np.argsort(cooks_mult)[::-1][:3]:
    axes6[1, 1].annotate(
        df["municipio"].iloc[idx_m],
        xy=(idx_m, cooks_mult[idx_m]), xytext=(4, 4), textcoords="offset points",
        fontsize=8, fontweight="bold", color="#b22222"
    )
axes6[1, 1].set_title("D) Distancia de Cook por municipio", fontsize=10.5, fontweight="bold")
axes6[1, 1].set_xlabel("Indice del municipio (0 a 124)", fontsize=9)
axes6[1, 1].set_ylabel("Distancia de Cook", fontsize=9)
axes6[1, 1].grid(True, linestyle=":", alpha=0.6)
axes6[1, 1].legend(fontsize=8.5, loc="upper right")

fig6.suptitle(
    "Figura 6: Diagnostico de residuos del modelo de regresion lineal multiple (n = 125 municipios)",
    fontsize=13, fontweight="bold", y=0.99
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig6.savefig(ARCHIVO_FIGURA_06, dpi=300)
plt.close(fig6)

print("\n" + "=" * 95)
print("ARCHIVOS GUARDADOS EN 03_resultados:")
print(f"1) {ARCHIVO_TABLA_13}")
print(f"2) {ARCHIVO_TABLA_14}")
print(f"3) {ARCHIVO_TABLA_15}")
print(f"4) {ARCHIVO_TABLA_16}")
print(f"5) {ARCHIVO_TABLA_17}")
print(f"6) {ARCHIVO_TABLA_18}")
print(f"7) {ARCHIVO_TABLA_19}")
print(f"8) {ARCHIVO_TABLA_20}")
print(f"9) {ARCHIVO_FIGURA_05} (Figura 5: grafica con recta de regresion simple)")
print(f"10) {ARCHIVO_FIGURA_06} (Figura 6: diagnostico de residuos del modelo multiple)")
print("=" * 95)