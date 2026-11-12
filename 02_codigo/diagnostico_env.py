import os
from dotenv import load_dotenv

ruta_env = r"C:\Users\cadsi\Avance_Mate\.env"
print(f"¿El archivo .env existe en la ruta?: {os.path.exists(ruta_env)}")

# Cargar el archivo .env
load_dotenv(ruta_env, override=True)

# Buscar las posibles combinaciones de nombres
print("\nBuscando variables tipo 'DB_...':")
print(f"DB_USER: {os.getenv('DB_USER', 'No encontrado')}")
print(f"DB_PASSWORD: {'Encontrado (oculto por seguridad)' if os.getenv('DB_PASSWORD') else 'No encontrado'}")

print("\nBuscando variables tipo 'MYSQL_...':")
print(f"MYSQL_USER: {os.getenv('MYSQL_USER', 'No encontrado')}")
print(f"MYSQL_PASSWORD: {'Encontrado (oculto por seguridad)' if os.getenv('MYSQL_PASSWORD') else 'No encontrado'}")