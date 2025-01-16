# filter_entries.py
import argparse
from json_processor import write_entries, read_entries

def get_variables(variables_file):
    """
    Extracts the variables written in the variables_file
    These variables will the ones to be filtered
    """
    variables = []

    with open(variables_file, "r") as file:
        for line in file:
            word = line.strip()
            variables.append(word)

    return variables

def load_and_filter_entries_lvl0(input_file, variables):
    """
    Reads and filters entries based on the desired variables.
    If the value of a key is another dictionary, the new entry will have all the variables inside of that said dictionary 
    """
    filtered_data = []
    json_data = read_entries(input_file) 

    for entry in json_data:
        filtered_entry = {}  

        for key, value in entry.items():  
            if key in variables:         
                filtered_entry[key] = value  

        filtered_data.append(filtered_entry)

    return filtered_data

def load_and_filter_entries_lvl1(input_file, variables):
    """
    Reads and filters entries based on the desired variables.
    If the value of a key is another dictionary, the new entry will have only the variables of that said dictionary that appear in the variables_file
    """
    filtered_data = []
    json_data = read_entries(input_file) 

    for entry in json_data:
        filtered_entry = {}  # Diccionario vacío para almacenar claves filtradas

        for key, value in entry.items():  # Iterar sobre las claves y valores del diccionario
            if key in variables:         # Verificar si la clave está en `variables`
                if isinstance(value, dict):
                    filtered_entry[key] = {}
                    for key2, value2 in value.items():
                        if key2 in variables:
                            filtered_entry[key][key2] = value2
                else:
                    filtered_entry[key] = value  # Agregar la clave y el valor al diccionario filtrado

        filtered_data.append(filtered_entry)

    return filtered_data

def main():
    parser = argparse.ArgumentParser(description="Filter JSON entries with the selected variables.")
    parser.add_argument('input_file', help="Path to the input file containing JSON entries.")
    parser.add_argument('output_file', help="Path to the output file to write filtered entries.")
    parser.add_argument('variables_file', help="Path to the file that contains the variables that the user wants to be filtered.")
    parser.add_argument('depth_level', help="Level of depth of the filters. Value must be either 0(simple) or 1(dictionary inside dictionary).")
    
    args = parser.parse_args()

    variables = get_variables(args.variables_file)
    json_filtered = []

    if int(args.depth_level) == 0:
        json_filtered = load_and_filter_entries_lvl0(args.input_file, variables)
    elif int(args.depth_level) == 1:
        json_filtered = load_and_filter_entries_lvl1(args.input_file, variables)
    else:
        print(f"Error. Depth_level must be either 0 or 1 but it was '{args.depth_level}'.")
        return 1
    
    # Write to output
    write_entries(json_filtered, args.output_file)
    print(f"Filtering complete. {len(json_filtered)} matching entries written to '{args.output_file}'.")

if __name__ == "__main__":
    main()