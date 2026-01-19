import torch
import torchhd
import pandas as pd
import numpy as np
import argparse
from correlation_matrix import discover_all_parameters
import sys
import json
import psutil
import time
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import serial
from pympler import asizeof

'python3 lum_data_processors/hdvLum.py /home/victor/Descargas/xmas_merged_data.json --parameters battery_level hops hours_in_emergency hours_in_power link_quality outputState state times_in_emergency times_in_power travel_ms two_in_one_battery_level WBN_rssi_correction_val buffer_usage.average buffer_usage.maximum Network_channel_PER cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_rx_ack_other_reasons cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail unknown unkown network_scans_amount scanstat_avg_routers cfmac_pending_broadcast_le_member cluster_channel packets_dropped Unack_broadcast_channel cluster_members nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address nexthop_details.next_hop_quality nexthop_details.next_hop_rssi nexthop_details.next_hop_power'
'python3 lum_data_processors/hdvLum.py /home/victor/Descargas/new_file.json --already_saved_hdv_matrices 0 --already_saved_hdvs_prot 0 --parameters battery_level hops hours_in_emergency times_in_emergency times_in_power two_in_one_battery_level nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address state cbmac_details.cbmac_rx_ack_other_reasons buffer_usage.average outputState buffer_usage.maximum nexthop_details.next_hop_quality nexthop_details.next_hop_rssi cluster_channel nexthop_details.next_hop_power cluster_members link_quality hours_in_power travel_ms WBN_rssi_correction_val cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail network_scans_amount cfmac_pending_broadcast_le_member Unack_broadcast_channel'

params_hdv = [] # params_hdv[i] access the matrix in that position (the matrix of a parameter). params_hdv[i][j] access the hdv of a parameter. the matrix is a dictionary {0: tensor, 1: tensor ...}
params_hdv_order = []

hdv_matrices_path = 'hdv_matrices'
hdv_prototypes_path = 'hdv_prototypes'

def process_data(file_path, parameters):
    batch_size = 100_000  # número de líneas por lote
    batches = []  # para acumulación temporal

    for i, line in enumerate(open(file_path, "r", encoding="utf-8"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            batches.append(obj)
        except json.JSONDecodeError:
            continue

        # cada cierto número de líneas, convertimos y guardamos en disco o acumulamos
        if len(batches) >= batch_size:
            batch_df = pd.json_normalize(batches)
            if i == batch_size:
                data = batch_df  # primer lote
            else:
                data = pd.concat([data, batch_df], ignore_index=True)
            batches.clear()

    # procesar el resto
    if batches:
        batch_df = pd.json_normalize(batches)
        data = pd.concat([data, batch_df], ignore_index=True) if "data" in locals() else batch_df
    
    #nos quedamos unicamente con las columnas que nos interesan
    return data[parameters]

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
                pass
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")

            #extracted_params = extract_parameters(entry,parameters)
            extracted_params = {}
            for key in parameters:
                if key in entry:
                    extracted_params[key] = entry[key]

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
    times = [] # Time(s)
    params = [] # Variable names
    for param in ranges.keys():
        time_ini = time.time()
        range_hdv_levels(ranges[param]["higher"],ranges[param]["lower"],int(vector_size),param,m_levels)
        time_fin = round(time.time() - time_ini, 4)
        times.append(time_fin)
        params.append(param)
    total_time = round(time.time() - total_time, 4)
    print(f"total_time:{total_time}")
    total_size = 0
    lengths = [] # Nº filas
    sizes = [] # Tamano matriz
    for i in range(len(params_hdv)):
        length = len(params_hdv[i])
        size = round(length * int(vector_size) / 1024 / 1024 ,4)
        total_size += size

        lengths.append(length)
        sizes.append(size)

    # table = Table()
    # table.add_column("Variable", style="cyan", no_wrap=True)
    # table.add_column("Nº de Filas", style="magenta")
    # table.add_column("Tamaño Matriz (Mb)", justify="left", style="green")
    # table.add_column("Tiempo(s)", justify="left", style="yellow")
    # table.add_column("Tamaño Total (Mb)", justify="center", style="green")
    # table.add_column("Tiempo Total(s)",justify="center", style="yellow")
    # for i in range(len(times)):
    #     table.add_row(str(params[i]),str(lengths[i]),str(sizes[i]),str(times[i]),"","")
    # table.add_row("", "", "", "", str(total_size),str(total_time))
    # console = Console()
    # console.print(table)

    # mem = psutil.virtual_memory()
    # uso_ram_gb = mem.used / (1024 ** 3)
    # print(f"USO MEM RAM : {uso_ram_gb}")

def save_matrix_to_files():

    for i in range(len(params_hdv)):
        tensor_list = list(params_hdv[i].values())
        matrix_tensor = torch.stack(tensor_list)
        torch.save(matrix_tensor, f'{hdv_matrices_path}/{params_hdv_order[i]}.pt') #params_hdv[i] access the matrix in that position (the matrix of a parameter). params_hdv[i][j] access the hdv of a parameter. the matrix is a dictionary {0: tensor, 1: tensor ...}

def read_matrix_file(input_file_name, param_name, ranges):

    loaded_matrix_tensor = torch.load(input_file_name, weights_only=False)
    matrix = {}
    higher = ranges[param_name]['higher']
    lower = ranges[param_name]['lower']
    j = 0
    for i in range(lower, higher + 1):  # matrix_tensor.size(0) gives the number of rows
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

            # Selection of indices
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

def range_hdv_levels_secuencial(higher, lower, vector_size, parameter, num_levels):
    '''
    Function that generates related HDVs on the GPU and saves them in a matrix

    This function will firstly generate a matrix of completely random HDVs. After the matrix is created, its HDVs will
    be modified so the values that are close are related and the values that are far apart are not so much. This modification
    will occur if, given a random number between 0 and 1, it is smaller than 1 / k.
    '''

    matrix = {}

    fst_hdv = torchhd.random(1, vector_size, "MAP",dtype=torch.int8) # First level HDV generated randomly
    matrix[lower] = fst_hdv[0]

    def create_related_matrix_secuencial(lower, higher, matrix_s, num_levels, vector_size):

        increment = round((abs(higher) + abs(lower)) / num_levels, 4)
        aux = lower

        b = round(vector_size / (2 * (num_levels - 1)))  # b = D / 2(M-1)
        if b == 0: b = 1

        used_indices = set()

        keys = [round(lower + i * increment, 4) for i in range(num_levels)]

        for i in range(num_levels):
            key = keys[i]

            if i == 0:
                prev_tensor = matrix_s[lower].clone()
            else:
                prev_tensor = matrix_s[keys[i - 1]].clone()

            # Selection of indices
            available = list(set(range(vector_size)) - used_indices)
            if len(available) < b:
                used_indices = set()
                available = list(range(vector_size))

            selected = random.sample(available, b)
            used_indices.update(selected)
            indices = torch.tensor(selected)

            prev_tensor[indices] *= -1

            matrix_s[key] = prev_tensor

    create_related_matrix_secuencial(lower, higher, matrix, num_levels, vector_size)

    params_hdv_order.append(parameter)
    params_hdv.append(matrix)

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
                    pass
                    print(f"ERROR {variable} : {value}")
            else:
                try:
                    indices = params_hdv[i].keys()
                    nearest_value = min(indices, key=lambda v: abs(v - value))
                    lum_hdv_entry = torchhd.bundle(lum_hdv_entry, params_hdv[i][nearest_value])
                except:
                    pass
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
                pass
                
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

    try:
        del(row["classification"])
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

        else:
            indices = params_hdv[i].keys()
            nearest_value = min(indices, key=lambda v: abs(v - value))
            lum_hdv_entry = torchhd.bundle(lum_hdv_entry, params_hdv[i][nearest_value])

        i+=1

    lum_hdv_entry = torch.where(lum_hdv_entry > 0, torch.tensor(1, device=lum_hdv_entry.device), torch.tensor(-1, device=lum_hdv_entry.device))
    return lum_hdv_entry

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
    parser.add_argument('--synthetic_file', help="Name of the file with synthetic data")
    parser.add_argument('--test_file', help="Name of the file with test data")
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
        print(ranges)
        m_levels = int(args.m_levels)#3336#(ranges["travel_ms"]["higher"] - ranges["travel_ms"]["lower"]) -160000
        generate_hdvs(args.data_file, args.vector_size, parameters, ranges, m_levels)
        #save_matrix_to_files()

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
  

    '''"------------- DATA PROCESSING --------------"'''

    print("Processing Data...")    
    lum_hdvs = {} # key = class, value = hdv
    df_train = process_data(args.data_file, parameters)

    df_test_real = process_data(args.test_file, parameters)
    df_test_real['classification'] = 'r'

    df_test_synth = process_data(args.synthetic_file, parameters)
    df_test_synth['classification'] = 's'

    print(f"Num real:{len(df_test_real)} Num synth:{len(df_test_synth)}")
    df_test = pd.concat([df_test_real, df_test_synth], ignore_index=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total train entries:{len(df_train)}")
    print(f"Total test entries:{len(df_test)}")

    "------------- ENCODE LUMS --------------"

    time_init_train = time.time()
    lum_hdvs[75] = lum_to_hdv_levels(df_train)
    print(f"total_training_time:{time.time()-time_init_train}")
    print(f"total_size_hdv_levels:{asizeof.asizeof(params_hdv)}")

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

    ''' ---- SETTING THRESHOLD FOR CLASSIFICATION ----'''
    # T = M + 2 * N where M is the average distance between sample and prototype and n is the std deviation

    print("Calculating threshold...")

    similities = []
    for entry in df_train.itertuples():
        hdv = entry_to_hdv_levels(entry)
        if hdv == None:
            print("Train entry has no valid values, continuing with next entry")
            continue
        cos = torchhd.cosine_similarity(lum_hdvs[75],hdv)
        similities.append(cos)

    mean = np.mean(similities)
    std_dev = np.std(similities)
    threshold = mean - 2 * std_dev
    print(f"threshold:{threshold} // mean:{mean}, std_dev:{std_dev}, ")

    '''------------- TESTING --------------'''

    i = 0
    hdc_classification = []
    cosine_values = []
    start = time.time()
    for row in df_test.itertuples():
        hdv = entry_to_hdv_levels(row)
        cos = torchhd.cosine_similarity(lum_hdvs[75],hdv)

        if cos >= threshold: hdc_classification.append('r')
        else: hdc_classification.append('s')
        cosine_values.append(cos)
        i+= 1

    score_real = 0
    score_synth = 0
    for j in range(len(hdc_classification)):
        comp = df_test['classification'].iloc[j]
        if hdc_classification[j] == comp and comp == 'r':
            score_real += 1
        elif hdc_classification[j] == comp and comp == 's':
            score_synth += 1
    accuracy = (score_real + score_synth) / (len(hdc_classification))


    finish = time.time()

    TP = score_real
    TN = score_synth
    FP = len(df_test_synth) - score_synth
    FN = len(df_test_real) - score_real

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    PPV = TP / (TP + FP)   # precision
    NPV = TN / (TN + FN)
    balanced_accuracy = (TPR + TNR) / 2
    f1_score = 2 * (PPV * TPR) / (PPV + TPR)

    print(f"Accuracy (ACC - Exactitud): {accuracy}") # exactitud es la fracción de predicciones correctas sobre el total de predicciones
    print(f"TPR (True Positive Rate - Sensibilidad / Recall): {TPR}") # Es la fracción de ejemplos positivos reales que han sido clasificados como positivos por el clasificador
    print(f"TNR (True Negative Rate - Especificidad): {TNR}") # especificadad es la fracción de ejemplos negativos reales que han sido clasificados como negativos por el clasificador
    print(f"FPR (False Positive Rate - Tasa de falsos positivos): {FPR}") # error negativo es la fracción de ejemplos negativos reales que han sido clasificados como positivos por el clasificador
    print(f"FNR (False Negative Rate - Tasa de falsos negativos): {FNR}") # error positivo es la fracción de ejemplos positivos reales que han sido clasificados como negativos por el clasificador
    print(f"PPV (Positive Predictive Value - Precisión): {PPV}") # precisión es la fracción de ejemplos clasificados como positivos que son realmente positivos
    print(f"NPV (Negative Predictive Value - Valor predictivo negativo): {NPV}") # valor predictivo negativo es la fracción de ejemplos clasificados como negativos que son realmente negativos
    print(f"Balanced Accuracy (BA - Exactitud balanceada): {balanced_accuracy}") # la exactitud balanceada es la media aritmética de la sensibilidad y la especificidad
    print(f"F1 (F1 Score - Media armónica Precision/Recall): {f1_score}") # la media armónica entre la precisión y el recall
    print(f"time_testing_elapsed:{finish-start}")

    return

if __name__ == "__main__":
    main()
