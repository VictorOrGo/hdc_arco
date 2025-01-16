import argparse
import json
from datetime import datetime

def calcular_diferencias(timestamps, time_lapse):
    # Convertir strings de timestamp en objetos datetime
    formato = "%Y-%m-%dT%H:%M:%S.%f"
    fechas = [datetime.strptime(ts, formato) for ts in timestamps]
    
    # Calcular diferencias entre timestamps consecutivos
    diferencias = {}
    for i in range(1, len(fechas)):
        diferencia = fechas[i] - fechas[i - 1]
        if diferencia.total_seconds() > time_lapse or diferencia.total_seconds() < time_lapse -2:
            diferencias[fechas[i]] = diferencia
    
    return diferencias


def extract_timestamps(file_path):
    timestamps = []
    try:
        # Leer el archivo línea por línea
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    # Intentar cargar cada línea como JSON
                    entry = json.loads(line)
                    # Extraer 'timestamp' si está presente
                    if 'timestamp' in entry:
                        timestamps.append(entry['timestamp'])
                except json.JSONDecodeError as e:
                    print(f"Error procesando la línea: {line.strip()}\nError: {e}")
        return timestamps
    except FileNotFoundError:
        print("El archivo no fue encontrado.")
        return []
    except Exception as e:
        print(f"Error inesperado: {e}")
        return []

# Ruta al archivo JSON

def main():
    parser = argparse.ArgumentParser(description="Calculate time differences between messages")
    parser.add_argument('input_file', help="Path to the input file containing JSON entries.")
    parser.add_argument('time_lapse', help="Time lapse in seconds. Always add 1 margin second")
    args = parser.parse_args()

    # Llamar a la función y mostrar los resultados
    timestamps = extract_timestamps(args.input_file)

    # Calcular diferencias de tiempo
    diferencias = calcular_diferencias(timestamps, args.time_lapse)

    # Mostrar resultados
    i = 1
    for clave, dif in diferencias.items():
        print(f"{i} - Diferencia {clave}: {dif}")
        i += 1

if __name__ == "__main__":
    main()