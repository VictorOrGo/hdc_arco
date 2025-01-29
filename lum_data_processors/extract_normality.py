import argparse
import math
import json

def process_data(data, upper_start, upper_end, lower_start, lower_end, father_key):
    """
    Processes JSON entries to extract the variables in range of the interval given.
    """
    variables_to_delete = []
    for key, value in data.items():
        if isinstance(value, dict):  # Si es un diccionario se llama recursivamente
            process_data(value, upper_start, upper_end, lower_start, lower_end, key)
            if not value:  # Verificar si el diccionario está vacío. Si esta vacío se marca para borrar
                variables_to_delete.append(key)
        else: 
            if math.isnan(value):
                variables_to_delete.append(key)
            elif (value >= 0 and (value < upper_start or value > upper_end)) or father_key == key:
                variables_to_delete.append(key)
            elif value < 0 and (value > lower_start or value < lower_end):
                variables_to_delete.append(key)

    # Eliminar claves marcadas
    for var in variables_to_delete:
        del data[var]

def group_and_count_values_by_key(data):
    """
    Recursively processes a nested JSON structure and groups positive
    and negative values by their keys.
    """
    grouped_values = {}

    def process_nested(data, father_key):
        for key, value in data.items():
            if isinstance(value, dict):  # Si es un subnivel, llamar recursivamente
                process_nested(value, key)
            else:
                # Inicializar la clave en el diccionario si no existe
                new_key = key + " - " + father_key
                new_key_tras = father_key + " - " + key
                if new_key in grouped_values:
                    if value > 0:
                        grouped_values[new_key]["positive"].append(value)
                    else:
                        grouped_values[new_key]["negative"].append(value)
                elif new_key_tras in grouped_values:
                    if value > 0:
                        grouped_values[new_key_tras]["positive"].append(value)
                    else:
                        grouped_values[new_key_tras]["negative"].append(value)
                else:
                    grouped_values[new_key] = {"positive": [], "negative": []}
                    '''
                    if value > 0:
                        grouped_values[new_key]["positive"].append(value)
                    else:
                        grouped_values[new_key]["negative"].append(value)
                    '''

    # Procesar el JSON
    process_nested(data, None)

    # Añadir conteo de valores para cada clave
    for key, values in grouped_values.items():
        values["positive_count"] = len(values["positive"])
        values["negative_count"] = len(values["negative"])
        del values["positive"]
        del values["negative"]

    return grouped_values

def main():
    parser = argparse.ArgumentParser(description="Extract variables that have a strong correlation")
    parser.add_argument('input_file', help="Path to the input JSON file containing the correlation matrix.")
    parser.add_argument('output_file', help="Path to the input JSON file where results will be written.")
    parser.add_argument('--upper_start', help="Upper interval start number", default='0.6')
    parser.add_argument('--upper_end', help="Upper interval end number", default='1')
    parser.add_argument('--lower_start', help="Lower interval start number", default='-0.4')
    parser.add_argument('--lower_end', help="Lower interval end number", default='-1')
    args = parser.parse_args()

    with open(args.input_file, 'r') as file:
        entries = json.load(file)

    process_data(entries, float(args.upper_start), float(args.upper_end), float(args.lower_start), float(args.lower_end), None)

    '''
    with open(args.output_file, 'w') as file:
        json.dump(entries, file, indent=4)
    '''

    normality = group_and_count_values_by_key(entries)

    with open(args.output_file, 'w') as file:
        json.dump(normality, file, indent=4)

if __name__ == "__main__":
    main()