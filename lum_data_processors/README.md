# ¿Cómo lanzar los scripts?
## correlation_matrix.py
Este script cuenta con los siguientes argumentos de entrada:
- `input_file`: Ruta al fichero de entrada que contiene las entradas del fichero JSON
- `start`: Fecha de inicio de la búsqueda en formato ISO (e.g., '2024-12-05T00:00:00'). Por defecto no se usa.
- `end`: Fecha de fin de la búsqueda en formato ISO (e.g., '2024-12-05T00:00:00'). Por defecto no se usa.
- `parameters`: List de parámetros a analizar (e.g., 'cbmac_details.cbmac_load' 'buffer_usage.average'). Si no se indica ninguno se tendrán todos los parametros en cuenta.
- `image`: 1 si se quieren guardar las matrices de correlación en una imágen y 0 si no. Por defecto es 0.
- `image_output`: Directorio donde se guardarán las imagenes. Este argumento es obligatorio si `image` es 1.
- `json`: 1 si se quieren guardar los datos de las matrices de correlación en un fichero JSON. Por defecto es 1.

Ejemplo de uso:

    python3 correlation_matrix.py data/data.json --json 1 --image 1 --image_output figsData --parameters battery_level code hops hours_in_emergency hours_in_power link_quality outputState state times_in_emergency times_in_power travel_ms two_in_one_battery_level --start 2024-12-11T05:00:00 --end 2024-12-11T21:00:00

## extract_normality.py
Este script cuenta con los siguientes argumentos de entrada:
- `input_file`: Ruta al fichero de entrada que contiene las entradas del fichero JSON
- `output_file`: Nombre del archivo en el que se guardarán los resultados del filtrado
- `upper_start`: Número donde empieza el intervalo mayor. Si el intervalo es (0.1, 0.9) este número sería 0.1. Por defecto su valor es 0.6
- `upper_end`: Número donde termina el intervalo mayor. Si el intervalo es (0.1, 0.9) este número sería 0.9. Por defecto su valor es 1
- `lower_start`: Número donde empieza el intervalo menor. Si el intervalo es (-0.9, -0.1) este número sería -0.1. Por defecto su valor es -0.4
- `lower_end`: Número donde termina el intervalo menor. Si el intervalo es (-0.9, -0.1) este número sería -0.9. Por defecto su valor es -1

Ejemplo de uso:

        python3 extract_normality.py correlation_matrix.json correlation_matrix_normality_diag.json --upper_start 0.5 --upper_end 1 --lower_start -0.5 --lower_end -1

## lum_processor.py
Este script cuenta con los siguientes argumentos de entrada:
- `input_file`: Ruta al fichero de entrada que contiene las entradas del fichero JSON
- `output_file`: Nombre del archivo en el que se guardarán los resultados del filtrado
- `variables_file`: Ruta del archivo que contiene las variables a ser filtradas
- `depth_level`: Nivel de profundidad de los filtros. El valor debe de ser 0 (filtrado simple) o 1 (diccionario dentro de diccionario)

Ejemplo de uso:

    python3 lum_processor.py data.json results.json lum_processor_filters/lum_data_filter.txt 1

## timestamp.py
Este script cuenta con los siguientes argumentos de entrada:
- `input_file`: Ruta al fichero de entrada que contiene las entradas del fichero JSON
- `time_lapse`: Tiempo en segundos que debe pasar para considerarse una anomalía. Siempre sumarle 1 segundo para un mejor filtrado.

Ejemplo de uso:

    python3 timestamp.py data.json 208
