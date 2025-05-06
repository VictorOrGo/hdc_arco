# HDV y procesamiento de luminarias

Este repositorio contiene diferentes ejemplos de cómo hacer uso de los HDV (*hyperdimensional vectors*) 

También contiene diferentes scripts de procesamiento de datos que enviados por las luminarias de Zemper basadas en la tecnología de Wirepas.

## aux_scripts

Contiene scripts que realizan diferentes tareas ya sean para limpiar datos, sacar diferentes resultados intermedios etc.

### borrar.py

Script para borrar líneas de un fichero hasta encontrar una palabra clave

### calcB.py

Script para calcular los diferentes valores que puede tomar b al variar M

### range.py

Script usado para obtener el máximo y el mínimo de los valores que pueden tomar las diferentes variables de un fichero json

## examples

Este directorio contiene los diferentes ejemplos en los que se implementan HDV para resolver diferentes problemas.

### hdvColors.py

Este ejemplo propone el uso de HDV para clasificar y reconocer colores

### hdvRecipes.py

Este ejemplo propone el uso de HDV para codificar recetas y obtener los siguientes resultados:

- Dada una receta, a qué receta del dataset se parece más
- Dada una receta, a qué país puede pertenecer la receta

### hdvProteins.py

Este ejemplo propone el uso de HDV para codificar y clasificar proteínas en función de si son de origen humano o si, por otro lado, son proteínas que provienen de la levadura.

## data

Este directorio contiene los diferentes datasets que son utilizados por los ejemplos del uso de HDV.

- `recipesData.csv` Es usado en el ejemplo `hdvRecipes.py`
- `human.fasta` Es usado en el ejemplo `hdvProteins.py`
- `yeast.fasta` Es usado en el ejemplo `hdvProteins.py`

## lum_data_processors

Este directorio contiene los diferentes scripts utilizados para el procesamiento de datos enviados por las luminarias.

### correlation_matrix.py

Este script se encargará de, dado un JSON con los datos de los envios realizados por las luminarias, calcular la correlación que existe entre las diferentes variables.

### extract_normality.py

Este script se encarga de procesar el fichero JSON generado por `correlation_matrix.py` con el objetivo de, dado un intervalo, generar otro fichero JSON con las variables que tienen una relación fuerte (es decir, las variables cuyo valor de correlación se encuentra entre el intervalo dado) y ver con la frecuencia que se repiten siendo valores similares. 

### json_processor.py

Script auxiliar para el procesamiento de archivos de tipo JSON que contienen los datos enviados por las luminarias.

### lum_processor.py

Este script filtra los ficheros JSON que contienen los datos enviados por las luminarias. Dará como resultado otro fichero JSON que contiene unicamente las variables deseadas.

### timestamp.py

Este script calcula la diferencia de tiempo que existe entre los envios de una luminaria. Tomará como entrada el periódo de tiempo que, en el caso de ser excedido, se tomará como una anomalía y se mostrará por pantalla.

#### lum_processor_filters

Este directorio contiene los filtros que son usados por los diferentes scripts.

- `lum_data_filter.txt` es un ejemplo de un fichero de filtro usado por `lum_processor.py`

## synthetic_data

Este directorio contiene archivos .json con entradas de datos de luminarias sintéticos para realizas diferentes pruebas

