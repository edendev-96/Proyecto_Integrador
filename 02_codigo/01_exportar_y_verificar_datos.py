import os
import sys
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

# 1. Definir rutas de carpetas y archivos
RUTA_BASE = r"C:\Users\cadsi\Avance_Mate"
RUTA_ENV = os.path.join(RUTA_BASE, ".env")
RUTA_DATOS = os.path.join(RUTA_BASE, "01_datos")
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "03_resultados")

ARCHIVO_CRUDOS = os.path.join(RUTA_DATOS, "datos_crudos.csv")
ARCHIVO_LIMPIA = os.path.join(RUTA_DATOS, "base_limpia.csv")
ARCHIVO_ENCABEZADOS = os.path.join(RUTA_RESULTADOS, "tabla_01_encabezados_base_limpia.csv")

# 2. Cargar variables del archivo .env sin mostrar la contrasena
load_dotenv(RUTA_ENV)

db_host = os.getenv("MYSQL_HOST", "localhost")
db_user = os.getenv("MYSQL_USER", "root")
db_password = os.getenv("MYSQL_PASSWORD")
db_name = os.getenv("MYSQL_DATABASE", "avance_bd")
db_port = int(os.getenv("MYSQL_PORT", "3307"))

if db_password is None:
    print("ERROR: No se encontro la variable MYSQL_PASSWORD en el archivo .env")
    sys.exit(1)

print("=" * 75)
print("PASO 1: PRUEBA DE CONEXION A MYSQL Y EXPORTACION DE DATOS CRUDOS")
print("=" * 75)

try:
    conexion = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        port=db_port
    )
    print(f"Conexion exitosa a MySQL -> Base de datos: '{db_name}'")

    # 3. Leer tabla datos_originales sin modificarla
    consulta_originales = "SELECT * FROM datos_originales ORDER BY id_fila_original ASC;"
    df_crudos_mysql = pd.read_sql(consulta_originales, conexion)

    # Exportar a datos_crudos.csv con codificacion utf-8-sig
    df_crudos_mysql.to_csv(ARCHIVO_CRUDOS, index=False, encoding="utf-8-sig")
    print(f"Archivo exportado: {ARCHIVO_CRUDOS}")

    # 4. Leer tabla datos_limpios desde MySQL para verificar coincidencia
    consulta_limpios = "SELECT * FROM datos_limpios ORDER BY clave_municipio ASC;"
    df_limpios_mysql = pd.read_sql(consulta_limpios, conexion)

    conexion.close()

except Exception as e:
    print("ERROR al conectar o consultar MySQL:", e)
    sys.exit(1)

# 5. Comprobar los archivos CSV generados en 01_datos
print("\n" + "=" * 75)
print("VERIFICACION DE ARCHIVOS CSV EN 01_datos")
print("=" * 75)

df_crudos_csv = pd.read_csv(ARCHIVO_CRUDOS, encoding="utf-8-sig", low_memory=False)
print(f"1) datos_crudos.csv -> Filas: {df_crudos_csv.shape[0]} | Columnas: {df_crudos_csv.shape[1]}")
print(f"   ¿Tiene exactamente 2,493 filas?: {'SI' if df_crudos_csv.shape[0] == 2493 else 'NO'}")

if not os.path.exists(ARCHIVO_LIMPIA):
    print(f"AVISO: No se encontro {ARCHIVO_LIMPIA}. Exportando desde MySQL datos_limpios...")
    df_limpios_mysql.to_csv(ARCHIVO_LIMPIA, index=False, encoding="utf-8-sig")

df_limpia_csv = pd.read_csv(ARCHIVO_LIMPIA, encoding="utf-8-sig")
n_filas_limpia, n_cols_limpia = df_limpia_csv.shape
n_vacios = int(df_limpia_csv.isna().sum().sum())
n_duplicados = int(df_limpia_csv.duplicated(subset=["clave_municipio"]).sum())

print(f"2) base_limpia.csv  -> Filas: {n_filas_limpia} | Columnas: {n_cols_limpia}")
print(f"   ¿Tiene 125 filas y 22 columnas?: {'SI' if (n_filas_limpia == 125 and n_cols_limpia == 22) else 'NO'}")
print(f"   Valores vacios totales: {n_vacios} | Municipios duplicados: {n_duplicados}")
print(f"   Filas leidas desde MySQL (avance_bd.datos_limpios): {df_limpios_mysql.shape[0]} filas, {df_limpios_mysql.shape[1]} columnas")

# 6. Guardar e imprimir Tabla 1: Encabezados y tipos de datos de base_limpia.csv (Punto 6 de la rubrica)
print("\n" + "=" * 75)
print("TABLA 1: ENCABEZADOS DE LA BASE DE DATOS LIMPIA (22 COLUMNAS)")
print("=" * 75)

resumen_columnas = []
for idx, col in enumerate(df_limpia_csv.columns, start=1):
    resumen_columnas.append({
        "num_columna": idx,
        "nombre_columna": col,
        "tipo_dato_python": str(df_limpia_csv[col].dtype),
        "valores_no_vacios": int(df_limpia_csv[col].notna().sum()),
        "ejemplo_primer_registro": df_limpia_csv[col].iloc[0]
    })

df_encabezados = pd.DataFrame(resumen_columnas)
df_encabezados.to_csv(ARCHIVO_ENCABEZADOS, index=False, encoding="utf-8-sig")
print(df_encabezados.to_string(index=False))
print(f"\nTabla de encabezados guardada en: {ARCHIVO_ENCABEZADOS}")
print("=" * 75)