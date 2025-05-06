from correlation_matrix import extract_parameters
import json
import argparse
def lower_and_higher_value(data_file, parameters):
    '''
    Funtion that will read the data_file and get for each parameter the higher and lower value
    '''
    result = {}

    for parameter in parameters:
        result[parameter] = {"higher":-1, "lower": 1e8}

    with open(data_file, 'r') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")

            extracted_params = extract_parameters(entry, parameters)

            for parameter in parameters:
                if parameter not in extracted_params.keys(): continue
                elif extracted_params[parameter] == None: continue
                elif extracted_params[parameter] > result[parameter]["higher"]:
                    result[parameter]["higher"] = int(extracted_params[parameter] + 10) # We add 10 in order to give a margin of different values
                elif extracted_params[parameter] < result[parameter]["lower"]:
                    result[parameter]["lower"] = int(extracted_params[parameter] - 10) # We substract 10 in order to give a margin of different values

    return result # Dict{dict} // result['hours_in_power'] = {'higher': 20555, 'lower': 44}

def main():
    parser = argparse.ArgumentParser(description="Visualize correlations of parameters from JSON entries grouped by source_address.")
    parser.add_argument('--parameters', nargs='+', help="List of hierarchical parameter paths to analyze, e.g., 'cbmac_details.cbmac_load' 'buffer_usage.average'. If not provided, all available parameters will be used.", default=None)
    args = parser.parse_args()

    file = "/home/victor/Descargas/new_file.json"
    result = lower_and_higher_value(file,args.parameters)
    print(result)

if __name__ == "__main__":
    main()