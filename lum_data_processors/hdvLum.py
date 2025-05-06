import torch
import torchhd
import pandas as pd
import numpy as np
import argparse
from correlation_matrix import process_data, discover_all_parameters, extract_parameters
import sys
import json
import psutil
import time
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt

'python3 lum_data_processors/hdvLum.py /home/victor/Descargas/xmas_merged_data.json --parameters battery_level hops hours_in_emergency hours_in_power link_quality outputState state times_in_emergency times_in_power travel_ms two_in_one_battery_level WBN_rssi_correction_val buffer_usage.average buffer_usage.maximum Network_channel_PER cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_rx_ack_other_reasons cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail unknown unkown network_scans_amount scanstat_avg_routers cfmac_pending_broadcast_le_member cluster_channel packets_dropped Unack_broadcast_channel cluster_members nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address nexthop_details.next_hop_quality nexthop_details.next_hop_rssi nexthop_details.next_hop_power'
'python3 lum_data_processors/hdvLum.py /home/victor/Descargas/new_file.json --already_saved_hdv_matrices 0 --already_saved_hdvs_prot 0 --parameters battery_level hops hours_in_emergency times_in_emergency times_in_power two_in_one_battery_level nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address state cbmac_details.cbmac_rx_ack_other_reasons buffer_usage.average outputState buffer_usage.maximum nexthop_details.next_hop_quality nexthop_details.next_hop_rssi cluster_channel nexthop_details.next_hop_power cluster_members link_quality hours_in_power travel_ms WBN_rssi_correction_val cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail network_scans_amount cfmac_pending_broadcast_le_member Unack_broadcast_channel'

params_hdv = [] # params_hdv[i] accede a la matriz de esa posición (la matriz de un parametro). params_hdv[i][j] accede al hdv de la matriz de un parametro. ojo, la matriz es un diccionario {0: tensor, 1: tensor ...}
params_hdv_order = []

hdv_matrices_path = 'hdv_matrices'
hdv_prototypes_path = 'hdv_prototypes'

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

def generate_hdvs(data_file, vector_size, parameters, ranges, m_levels):
    '''
    Recogeremos los valores más altos y más bajos encontrados para cada variable
    '''
    # Ranges = Dict{dict} // result['hours_in_power'] = {'higher': 20555, 'lower': 44}

    '''
    Una vez tenemos los intervalos de las diferentes variables, crearemos las matrices de HDV.
    Cada matriz de HDV representa una variable. Si una variable tiene siempre el mismo valor se generará un único HDV
    y no será una matriz.
    '''
    print("Generating HDVs...")
    total_time = time.time()
    times = [] # Tiempo(s)
    params = [] # Nombre variable
    for param in ranges.keys():
        time_ini = time.time()
        range_hdv_levels(ranges[param]["higher"],ranges[param]["lower"],int(vector_size),param,m_levels) 
        time_fin = round(time.time() - time_ini, 4)
        times.append(time_fin)
        params.append(param)
    total_time = round(time.time() - total_time, 4) 

    total_size = 0
    lengths = [] # Nº filas
    sizes = [] # Tamano matriz
    for i in range(len(params_hdv)):
        length = len(params_hdv[i])
        size = round(length * int(vector_size) / 1024 / 1024 ,4)
        total_size += size
        
        lengths.append(length)
        sizes.append(size)

    table = Table()
    table.add_column("Variable", style="cyan", no_wrap=True)
    table.add_column("Nº de Filas", style="magenta")
    table.add_column("Tamaño Matriz (Mb)", justify="left", style="green")
    table.add_column("Tiempo(s)", justify="left", style="yellow")
    table.add_column("Tamaño Total (Mb)", justify="center", style="green")
    table.add_column("Tiempo Total(s)",justify="center", style="yellow")
    for i in range(len(times)):
        table.add_row(str(params[i]),str(lengths[i]),str(sizes[i]),str(times[i]),"","")
    table.add_row("", "", "", "", str(total_size),str(total_time))
    console = Console()
    console.print(table)

    mem = psutil.virtual_memory()
    uso_ram_gb = mem.used / (1024 ** 3)
    print(f"USO MEM RAM : {uso_ram_gb}")

def save_matrix_to_files():

    for i in range(len(params_hdv)):
        tensor_list = list(params_hdv[i].values())
        matrix_tensor = torch.stack(tensor_list)
        torch.save(matrix_tensor, f'{hdv_matrices_path}/{params_hdv_order[i]}.pt') #params_hdv[i] accede a la matriz de esa posición (la matriz de un parametro). params_hdv[i][j] accede al hdv de la matriz de un parametro

def read_matrix_file(input_file_name, param_name, ranges):

    loaded_matrix_tensor = torch.load(input_file_name, weights_only=False)
    matrix = {}
    higher = ranges[param_name]['higher']
    lower = ranges[param_name]['lower']
    j = 0
    for i in range(lower, higher + 1):  # matrix_tensor.size(0) da el número de filas
        matrix[i] = loaded_matrix_tensor[j]
        j += 1

    params_hdv.append(matrix)
    params_hdv_order.append(param_name)

def read_selected_matrix_rows(input_file_name, lower, higher):

    loaded_matrix_tensor = torch.load(input_file_name, weights_only=False)
    rows = loaded_matrix_tensor[lower:higher]
    del loaded_matrix_tensor
    return rows

def range_hdv_levels(higher, lower, vector_size, parameter, num_levels):
    '''
    Function that generates related HDVs on the GPU and saves them in a matrix

    This function will firstly generate a matrix of completely random HDVs. After the matrix is created, its HDVs will
    be modified so the values that are close are related and the values that are far apart are not so much. This modification
    will occur if, given a random number between 0 and 1, it is smaller than 1 / k.  
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    matrix = {}
    
    fst_hdv = torchhd.random(1, vector_size, "MAP",dtype=torch.int8) # First level HDV generated randomly
    matrix[lower] = fst_hdv[0].to(device)

    def create_related_matrix(lower, higher, matrix_s, num_levels, vector_size, device):

        increment = round((abs(higher) + abs(lower)) / num_levels, 4)
        aux = lower

        b = round(vector_size / (2 * (num_levels - 1)))  # b = D / 2(M-1)
        if b == 0: b = 1

        used_indices = set()

        keys = [round(lower + i * increment, 4) for i in range(num_levels)]

        for i in range(num_levels):
            key = keys[i]
            
            if i == 0:
                prev_tensor = matrix_s[lower].to(device).clone()
            else:
                prev_tensor = matrix_s[keys[i - 1]].to(device).clone()

            # Selección de índices
            available = list(set(range(vector_size)) - used_indices)
            if len(available) < b:
                used_indices = set()
                available = list(range(vector_size))
                
            selected = random.sample(available, b)
            used_indices.update(selected)
            indices = torch.tensor(selected, device=device)

            prev_tensor[indices] *= -1

            matrix_s[key] = prev_tensor

    create_related_matrix(lower, higher, matrix, num_levels, vector_size, device)

    increment = round((abs(higher) + abs(lower)) / num_levels, 4)
    aux = lower
    for i in range(num_levels):
        matrix[aux] = matrix[aux].cpu()  # We move the matrix to RAM in order to save space in the GPU for the next matrix if needed
        aux = round(aux + increment, 4)

    torch.cuda.empty_cache()

    params_hdv_order.append(parameter)
    params_hdv.append(matrix)
    print(f"Matriz de HDVs creada para {parameter}")

def lum_to_hdv_levels(df_lum:pd.DataFrame):
    '''
    Function that given a dataframe that contains ONLY the entries from ONLY one lum

    This function will iter every row in the dataframe. Then per every row it will take the value of each column
    and look for its HDV. Once found it will do the bundle operation to add the previous information with the new one.
    This will result in the HDV prototype of the lum
    '''

    lum_hdv_entry = None
    lum_hdv_prot = None
    i = 0

    #df_lum = df_lum.drop(columns=["src", "timestamp"])
    try:
        df_lum = df_lum.drop(columns=["src"])
    except:
        pass

    try:
        df_lum = df_lum.drop(columns=["timestamp"])
    except:
        pass
    
    bundleCount = 0

    if df_lum.empty:
        print("Dataset empty, continuing with next node")
        return None

    for row in df_lum.itertuples():
        row = row._asdict()
        del(row["Index"])

        for variable, value in row.items():
            if pd.isna(value):  
                i += 1
                continue
            if lum_hdv_entry == None:
                try:
                    indices = params_hdv[i].keys()
                    nearest_value = min(indices, key=lambda v: abs(v - value))
                    lum_hdv_entry = params_hdv[i][nearest_value]
                except:
                    print(f"ERROR {variable} : {value}")
            else:
                try:
                    indices = params_hdv[i].keys()
                    nearest_value = min(indices, key=lambda v: abs(v - value))
                    lum_hdv_entry = torchhd.bundle(lum_hdv_entry, params_hdv[i][nearest_value])
                except:
                    print(f"ERROR {variable} : {value}")
            
            i += 1

        if lum_hdv_prot == None:
            lum_hdv_prot = lum_hdv_entry
        else:
            try:
                if bundleCount >= 100:
                    lum_hdv_entry = torch.where(lum_hdv_entry > 0, torch.tensor(1, device=lum_hdv_entry.device), torch.tensor(-1, device=lum_hdv_entry.device))
                    lum_hdv_prot = torchhd.bundle(lum_hdv_prot, lum_hdv_entry)
                    lum_hdv_prot = torch.where(lum_hdv_prot > 0, torch.tensor(1, device=lum_hdv_prot.device), torch.tensor(-1, device=lum_hdv_prot.device))
                    bundleCount = 0
                else:
                    lum_hdv_entry = torch.where(lum_hdv_entry > 0, torch.tensor(1, device=lum_hdv_entry.device), torch.tensor(-1, device=lum_hdv_entry.device))
                    lum_hdv_prot = torchhd.bundle(lum_hdv_prot, lum_hdv_entry)
                    bundleCount += 1
            except:
                print(f"Data entry with all None values found, skipping line")

        lum_hdv_entry = None
        i = 0
    
    lum_hdv_prot = torch.where(lum_hdv_prot > 0, torch.tensor(1, device=lum_hdv_prot.device), torch.tensor(-1, device=lum_hdv_prot.device))
    return lum_hdv_prot

def entry_to_hdv_levels(entry):

    lum_hdv_entry = None
    if isinstance(entry, dict):
        row = entry
    else:
        row = entry._asdict()
    try:
        del(row["Index"])
    except:
        pass

    try:
        del(row["src"])
    except:
        pass

    try:
        del(row["timestamp"])
    except:
        pass

    i = 0
    for __, value in row.items():
        if pd.isna(value):  
            i += 1
            continue
        
        if lum_hdv_entry == None:
            indices = params_hdv[i].keys()
            nearest_value = min(indices, key=lambda v: abs(v - value))
            lum_hdv_entry = params_hdv[i][nearest_value]

            keys = list(params_hdv[i])
            indice = keys.index(nearest_value)
            # previous_key = keys[indice - 1] if indice > 0 else None
            # next_key = keys[indice + 1] if indice < len(keys) - 1 else None
            # print(f"For battery_level {value} -> M level {indice} (previous mapped level {previous_key} | mapped value: {nearest_value} | next mapped level {next_key})")
        else:
            indices = params_hdv[i].keys()
            nearest_value = min(indices, key=lambda v: abs(v - value))
            lum_hdv_entry = torchhd.bundle(lum_hdv_entry, params_hdv[i][nearest_value])
        
        i+=1

    lum_hdv_entry = torch.where(lum_hdv_entry > 0, torch.tensor(1, device=lum_hdv_entry.device), torch.tensor(-1, device=lum_hdv_entry.device))
    return lum_hdv_entry

def df_add_label(df:pd.DataFrame, k_classes, min_value, max_value):
    increment = round((max_value - min_value) / k_classes) 
    k_classes_max_value = []

    count = increment
    for i in range(k_classes):
        k_classes_max_value.append(count)
        count += increment

    df.loc['Ek_class'] = None
    for index, travel_ms in df.itertuples():
        if index == 'Ek_class': continue
        k_class = np.searchsorted(k_classes_max_value, travel_ms, side="left")
        df.at[index, 'Ek_class'] = k_class
    
    return df


def main():

    parser = argparse.ArgumentParser(description="Visualize correlations of parameters from JSON entries grouped by source_address.")
    parser.add_argument('data_file', help="Path to the input file containing JSON entries.")
    parser.add_argument('--vector_size', help="Size of the hdv vectors", default=10000)
    parser.add_argument('--already_saved_hdv_matrices', help="If 1 the matrices will not be generated and the files containing them must exist and match the parameters name, if 0 matrices will be created and saved in files", default=0)
    parser.add_argument('--already_saved_hdvs_prot', help="If 1 the HDV prototypes will not be generated and the files containing them must exist and match the node number, if 0 HDV prototypes will be created and saved in files", default=0)
    parser.add_argument('--parameters', nargs='+', help="List of hierarchical parameter paths to analyze, e.g., 'cbmac_details.cbmac_load' 'buffer_usage.average'. If not provided, all available parameters will be used.", default=None)
    parser.add_argument('--m_levels', help="Number of levels", default=3336)
    parser.add_argument('--k_classes', help="Number of classes", default=10)
    parser.add_argument('--output_image', help="Name of output image")
    args = parser.parse_args()

    '''
    Comprobamos si se han introducido parámetros. Si no se ha introducido ninguno como argumento se
    buscará en el fichero todas las que existen.
    '''

    if not args.parameters:
        print("No parameters specified. Discovering all available parameters from the data...")
        all_parameters = discover_all_parameters(args.data_file)
        if not all_parameters:
            print("No parameters found in entries.")
            sys.exit(0)
        print(f"Discovered {len(all_parameters)} parameters:")
        for param in sorted(all_parameters):
            print(f" - {param}")
        parameters = sorted(all_parameters)
    else:
        parameters = args.parameters

    if int(args.already_saved_hdv_matrices) == 1:
        print("Getting parameter ranges...")
        ranges = lower_and_higher_value(args.data_file, parameters)
        print("Reading hdv files...")
        for parameter in parameters:
            read_matrix_file(f'{hdv_matrices_path}/{parameter}.pt', parameter, ranges)
    else:
        print("Getting parameter ranges...")
        ranges = lower_and_higher_value(args.data_file, parameters)
        m_levels = int(args.m_levels)#3336#(ranges["travel_ms"]["higher"] - ranges["travel_ms"]["lower"]) -160000
        generate_hdvs(args.data_file, args.vector_size, parameters, ranges, m_levels)
        #save_matrix_to_files()

    valores = list(params_hdv[0].values())

    # Comprobamos si hay duplicados
    if len(valores) == len(set(valores)):
        print("No hay valores repetidos.")
    else:
        print("Hay valores repetidos.")

    '''
    Una vez calculadas y generadas las matrices de HDVs para cada variable procederemos con la creación de los HDV prototipo de cada luminaria.
    Para esto, se generará un HDV para cada entrada de datos (datos y diagnostico) que haya en el fichero de datos por luminaria. Estos HDV irán
    sucesivamente agregando la información del anterior consigo mismo para lograr un único HDV el cuál tendrá toda la información que haya de la luminaria
    en el fichero de datos. Esto será el vector prototipo.

    Para la generación de un HDV por cada entrada de datos del fichero, lo que se hará es procesar cada entrada por variable, es decir, se irá variable
    por variable que haya en esa entrada y en función de su valor, seleccionaremos el HDV correspondiente de la matriz de HDVs de esa variable que 
    habíamos generado anteriormente. Estos HDVs irán agregando su información al anterior según se vayan procesando (mismo procedimiento que con el HDV
    prototipo). El HDV resultante de la agregación de los HDVs de las variables será el HDV que representa a la luminaria para esa entrada de datos.   
    '''

    print("Processing Data...")
    # df = process_data(args.data_file, parameters, None, None)
    # nodes = df["src"].unique()
    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    # del df
    
    df_train = process_data('/home/victor/hdvExamples/lum15Train.json', parameters, None, None)
    df_test = process_data('/home/victor/hdvExamples/lum15Test.json', parameters, None, None)

    # df_train = df_train[df_train["src"] == 15]
    # df_test = df_test[df_test["src"] == 15]
    # df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    # df_train['timestamp'] = df_train['timestamp'].apply(lambda x: x.isoformat())
    # df_train.to_json('lum15Train.json', orient='records', lines=True)
    # df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
    # df_test['timestamp'] = df_test['timestamp'].apply(lambda x: x.isoformat())
    # df_test.to_json('lum15Test.json', orient='records', lines=True)
    
    lum_hdvs = {}

    df_test = df_test.drop(columns=['timestamp'])
    
    # hdv_nodes = {}
    # for node in nodes:
    #     hdv_nodes[node] = torchhd.random(1, int(args.vector_size), "MAP", dtype=torch.int8)

    # params_hdv.append(hdv_nodes) 
    # params_hdv_order.append("src")

    ''' ---- CLASSIFIER ENCODING ---- '''
    print("Encoding Classifiers...")

    lum_hdvs = {} # key = class, value = hdv

    df_train = df_train.dropna()
    lum_hdvs[15] = lum_to_hdv_levels(df_train)
    print(f"Lum {int(15)} processed.")

    # for node in nodes:
    #     subdf = df_train[df_train["src"] == node]
    #     lum_hdvs[node] = lum_to_hdv_levels(subdf)
    #     torch.save(lum_hdvs[node], f'{hdv_prototypes_path}/{int(node)}.pt')

    # for node in nodes:
    #     try:
    #         print(f"Reading {int(node)} HDV file")
    #         lum_hdvs[node] = torch.load(f'{hdv_prototypes_path}/{int(node)}.pt', weights_only=False)
    #     except:
    #         print(f"No file found for lum {int(node)}")

    '''
    if(int(args.already_saved_hdvs_prot) == 0):
        for node in nodes:
            subdf = df_train[df_train["src"] == node]
            lum_hdvs[node] = lum_to_hdv_levels(subdf)
            print(f"Lum {int(node)} processed. Saving HDV...")
            torch.save(lum_hdvs[node], f'{hdv_prototypes_path}/{int(node)}.pt')

    else:
        for node in nodes:
            try:
                print(f"Reading {int(node)} HDV file")
                lum_hdvs[node] = torch.load(f'{hdv_prototypes_path}/{int(node)}.pt', weights_only=False)
            except:
                print(f"No file found for lum {int(node)}")
    
    '''
    ''' ---- TESTING ---- '''
    print("Testing Classifiers...")

    #df_test = process_data("test15.json", parameters, None, None)
    df_test = df_test.dropna()
    df_test = df_test[df_test["src"] == 15]

    i = 0
    cosine_values = []
    for row in df_test.itertuples():
        hdv = entry_to_hdv_levels(row)
        if lum_hdvs[15] == None:
            continue
        cos = torchhd.cosine_similarity(lum_hdvs[15],hdv)
        ham = torchhd.hamming_similarity(lum_hdvs[15],hdv)
        print(f"Cosine: {cos} ||  Ham: {ham}")
        cosine_values.append(cos)

        #if i ==39: break
        i+= 1

    ids = list(range(len(cosine_values)))  # eje X: 0 a 9999

    plt.figure(figsize=(12, 5))
    plt.scatter(ids, cosine_values, s=10, alpha=0.6, color='dodgerblue')  # s ajusta el tamaño del punto
    plt.ylim(0, 1)  # límites del eje Y
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.ylim(0, 1)
    plt.xlabel('ID de entrada')
    plt.ylabel('Similitud')
    plt.title('Similitud por entrada')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f'{args.output_image}.png', dpi=300)

    mem = psutil.virtual_memory()
    uso_ram_gb = mem.used / (1024 ** 3)
    print(f"USO MEM RAM : {uso_ram_gb}")

    return 
    '''
    - Creación de HDV
        · Cada variable tendrá realmente que ser una matriz en la que cada fila representa un valor distinto de una variable. Como nº de 
          columnas tendremos la dimensión de los hipervectores (igual que en el ejemplo de los colores). La matriz se puede implementar como
          un diccionario.
        · De entre las filas de la matriz se seleccionará aquella que más se aproxima al valor de la nueva entrada de datos. 
        · Se hace lo mismo para todas las variables y se agrupan en un solo hdv siendo este ya el de la luminaria en ese momento
        · Hacer un hdv por cada instancia de datos de la luminaria en el JSON
        · Fusionar todos esos hdv en un único hdv que representará a la luminaria (hdv prototipo)
        · IDEA: Fusionar a parte todos los hdv prototipo para conseguir un único hdv que represente un comportamiento normal
          y general. Puede ser útil para luego la clasificación.
    
    - Clasificación
        · Para ver si una luminaria se comporta de forma anómala o no se debe de realizar una comparación
        · Primero deberemos crear el hdv de los NUEVOS datos (los que no han sido utilizados para crear el hdv prototipo)
          recibidos por la luminaria. Se creará al igual que cada hdv individual utilizado para el hdv prototipo de la luminaria.
        · Se realizará la comparación del hdv nuevo con el prototipo de la luminaria y se verá como de parecidos son
        · Cuanto menos se parezcan más anómalos serán los datos enviados y puede ser indicarivo de un mal funcionamiento
        · IDEA: Si son muy parecidos (comportamiento normal de la luminaria), añadir esa nueva información al hdv prototipo
          para reforzarlo.
    
    '''

if __name__ == "__main__":
    main()
