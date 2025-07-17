import pandas as pd
import numpy as np
import argparse
from correlation_matrix import process_data, process_data_type2, discover_all_parameters, extract_parameters, extract_parameters_data_type2
import sys
import json
import psutil
import time
from rich.console import Console
from rich.table import Table
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import array
import _thread
from pympler import asizeof

'python3 lum_data_processors/hdvLum.py /home/victor/Descargas/xmas_merged_data.json --parameters battery_level hops hours_in_emergency hours_in_power link_quality outputState state times_in_emergency times_in_power travel_ms two_in_one_battery_level WBN_rssi_correction_val buffer_usage.average buffer_usage.maximum Network_channel_PER cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_rx_ack_other_reasons cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail unknown unkown network_scans_amount scanstat_avg_routers cfmac_pending_broadcast_le_member cluster_channel packets_dropped Unack_broadcast_channel cluster_members nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address nexthop_details.next_hop_quality nexthop_details.next_hop_rssi nexthop_details.next_hop_power'
'python3 lum_data_processors/hdvLum.py /home/victor/Descargas/new_file.json --already_saved_hdv_matrices 0 --already_saved_hdvs_prot 0 --parameters battery_level hops hours_in_emergency times_in_emergency times_in_power two_in_one_battery_level nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address state cbmac_details.cbmac_rx_ack_other_reasons buffer_usage.average outputState buffer_usage.maximum nexthop_details.next_hop_quality nexthop_details.next_hop_rssi cluster_channel nexthop_details.next_hop_power cluster_members link_quality hours_in_power travel_ms WBN_rssi_correction_val cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail network_scans_amount cfmac_pending_broadcast_le_member Unack_broadcast_channel'

params_hdv = [] # params_hdv[i] access the matrix in that position (the matrix of a parameter). params_hdv[i][j] access the hdv of a parameter. the matrix is a dictionary {0: tensor, 1: tensor ...}
params_hdv_order = []

hdv_matrices_path = 'hdv_matrices'
hdv_prototypes_path = 'hdv_prototypes'

CODE_MAP = {1: 0b0, -1: 0b1} # Codification of the different values that our HDV can take
DECODE_MAP = {0b0: 1, 0b1: -1}
NUMBERS_IN_A_BYTE = 8

def gen_random_hdv(vector_size) -> array.array:
    hdv = array.array('b',[0] * vector_size)
    for i in range(vector_size):
        bit = random.randint(0, 1)
        if bit == 1: 
            hdv[i] = 1
        else: 
            hdv[i] = -1
    return hdv

def encode_array_to_hdv(values:array.array) -> bytearray:
    hdv = bytearray((len(values) + NUMBERS_IN_A_BYTE - 1) // NUMBERS_IN_A_BYTE)  # We calculate how many bytes we will need

    for i, val in enumerate(values):
        code = CODE_MAP[val] # We get the codification for the value. EJ: val = 1 -> code = 0b0
        
        byte_index = i // NUMBERS_IN_A_BYTE # 1 byte can store 8 numbers because each number is 1 bit. i is the index in the decoded hdv so this way we can get what byte will store the number. EJ: i = 1 -> 1/8 = 0(byte index), i = 12 -> 12/8 = 1(byte index)
        
        shift = (i % NUMBERS_IN_A_BYTE) # We got the byte index so now we need know the position in the byte that the number will have. We get the shift needed to get the correct position. i % NUMBERS_IN_A_BYTE gets us the division of the byte (it has 8 divisions since we use one bit per number).

        hdv[byte_index] |= (code << shift) # We shift the code bits to the position needed and then we apply a OR operation so we ONLY change the values we want. EJ: code = 0b1, shift = 4 -> 0b00001001 (byte example) || 0b00010000 (code with shift) = 0b00011001
    
    return hdv

def decode_hdv_to_array(data:bytearray, vector_size) -> array.array:
    hdv = array.array('b',[0] * vector_size) # Initialize our HDV
    
    for i in range(vector_size):
        
        byte_index = i // NUMBERS_IN_A_BYTE # 1 byte can store 8 numbers because each number is 1 bit. i is the index in the decoded hdv so this way we can get what byte stores the number. EJ: i = 1 -> 1/8 = 0(byte index), i = 11 -> 11/8 = 1(byte index)
        
        shift = (i % NUMBERS_IN_A_BYTE) # We got the byte index so now we need to retrieve the number. We get the shift needed to get the correct bit. i % NUMBERS_IN_A_BYTE gets us the division of the byte (it has 8 divisions since we use a bit per number).

        code = (data[byte_index] >> shift) & 0b1 # We access the specific byte and apply a shift so we get the number we want at the end. After this we apply a mask in order to extract only the bit we want. EJ: 00000100(byte after shift) & 00000001(mask) = 00000000 = 0b00

        hdv[i] = DECODE_MAP[code] # We decode the bit number
    
    return hdv

def get_number_from_hdv(hdv:bytearray, index:int, vector_size) -> int:
    if index >= vector_size:
        print(f"ERROR: To get number index {index} must be lower than {vector_size}")

    byte_index = index // NUMBERS_IN_A_BYTE
    shift = (index % NUMBERS_IN_A_BYTE)
    code = (hdv[byte_index] >> shift) & 0b1
    return DECODE_MAP[code]

def set_number_in_hdv(hdv:bytearray, index:int, value:int, vector_size):
    if index >= vector_size:
        print(f"ERROR: To set number index {index} must be lower than {vector_size}")

    code = CODE_MAP[value]
    byte_index = index // NUMBERS_IN_A_BYTE
    shift = (index % NUMBERS_IN_A_BYTE)

    hdv[byte_index] &= ~(0b1 << shift)  # We delete previous values by applying a mask. We shift our mask so the bits 1 end up where we want to delete the selected number. After that, we invert the mask so 0 becomes 1 and the 1 becomes 0, this way we can apply an AND operation to turn to 0 the bits we want to clean and keep the rest as they were.
    hdv[byte_index] |= (code << shift)   

def bundle_hdv(hdv1:array.array, hdv2:array.array) -> array.array:
    result_hdv = array.array('b', (a + b for a, b in zip(hdv1, hdv2))) # Zip will group each component EJ: [(1, -1), (-1, 1), ... , (0, 0)]. Then we just add the components and create a new array.
    return result_hdv

def normalize_hdv(hdv:array.array, vector_size):
    for i in range(vector_size):
        number = hdv[i] # We will evaluate number so we save 2 more possible accesses to the array
        if number > 0: hdv[i] = 1 
        elif number < 0: hdv[i] = -1

def cosine_similarity(hdv1:array.array, hdv2:array.array) -> float:
    dot = sum(x * y for x, y in zip(hdv1, hdv2))
    mag_hdv1 = None
    mag_hdv2 = {'result':None}
    
    def calculate_magnitude():
        mag_hdv2['result'] = sum(x**2 for x in hdv2) ** 0.5
    
    _thread.start_new_thread(calculate_magnitude, ())
    mag_hdv1 = sum(x**2 for x in hdv1) ** 0.5
    
    while mag_hdv2['result'] == None: # We wait for the thread to finish
        pass

    if mag_hdv1 == 0 or mag_hdv2['result'] == 0:
        return 0  
    
    return dot / (mag_hdv1 * mag_hdv2['result'])

def get_sample(lst:list, n_elements:int): # Get a sample of N elements from a list
    lst_copy = lst[:]
    result = []
    for i in range(n_elements):
        x = random.randint(0, len(lst_copy) - 1)
        result.append(lst_copy.pop(x))  
    return result

def range_hdv_levels(higher, lower, m_levels, vector_size) -> tuple[dict[float, bytearray],list[float]]:
    matrix = {}
    
    fst_hdv = encode_array_to_hdv(gen_random_hdv(vector_size)) # First level HDV generated randomly
    matrix[lower] = bytearray(fst_hdv)

    increment = round((abs(higher) + abs(lower)) / m_levels, 4)

    b = round(vector_size / (2 * (m_levels - 1)))  # b = D / 2(M-1)
    if b == 0: b = 1

    used_indices = set()

    keys = [round(lower + i * increment, 4) for i in range(m_levels)]

    for i in range(m_levels):
        key = keys[i]
        
        if i == 0:
            prev_tensor = bytearray(matrix[lower])
        else:
            prev_tensor = bytearray(matrix[keys[i - 1]])

        available = list(set(range(vector_size)) - used_indices)
        if len(available) < b:
            used_indices = set()
            available = list(range(vector_size))
        
        selected = get_sample(available, b)
        used_indices.update(selected)

        for idx in selected:
            prev_value = get_number_from_hdv(prev_tensor, idx,vector_size)
            set_number_in_hdv(prev_tensor, idx, prev_value*(-1),vector_size)

        matrix[key] = bytearray(prev_tensor)

    return matrix,keys

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
        matrix, __ = range_hdv_levels(ranges[param]["higher"],ranges[param]["lower"],m_levels,vector_size) 
        params_hdv_order.append(param)
        params_hdv.append(matrix)
        time_fin = round(time.time() - time_ini, 4)
        times.append(time_fin)
        params.append(param)
    total_time = round(time.time() - total_time, 4) 
    print(f"total_time:{total_time}")
    total_size = 0
    lengths = [] # Num Rows
    sizes = [] # Matrix size
    for i in range(len(params_hdv)):
        length = len(params_hdv[i])
        size = round(asizeof.asizeof(params_hdv[i]) / 1024 / 1024, 4)
        total_size += size
        
        lengths.append(length)
        sizes.append(size)

    # total_size = round(total_size, 4)
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
    # # console = Console()
    # # console.print(table)

    print(f'total_size:{total_size}')
    mem = psutil.virtual_memory()
    uso_ram_gb = mem.used / (1024 ** 3)
    print(f"USO MEM RAM : {uso_ram_gb}")

def entry_to_hdv_levels(entry, vector_size):

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
            if isinstance(lum_hdv_entry,bytearray): lum_hdv_entry = decode_hdv_to_array(lum_hdv_entry, vector_size)
            lum_hdv_entry = bundle_hdv(lum_hdv_entry, decode_hdv_to_array(params_hdv[i][nearest_value],vector_size))
        
        i+=1

    normalize_hdv(lum_hdv_entry,vector_size)
    return lum_hdv_entry

def lum_to_hdv_levels(df_lum:pd.DataFrame, vector_size):
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
        df_lum = df_lum.drop(columns=["src", "timestamp"])
        df_lum = df_lum.drop(columns=["classification"])
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
                indices = params_hdv[i].keys()
                nearest_value = min(indices, key=lambda v: abs(v - value))
                lum_hdv_entry = params_hdv[i][nearest_value]
            else:
                indices = params_hdv[i].keys()
                nearest_value = min(indices, key=lambda v: abs(v - value))
                if isinstance(lum_hdv_entry,bytearray):
                    lum_hdv_entry = bundle_hdv(decode_hdv_to_array(lum_hdv_entry,vector_size), decode_hdv_to_array(params_hdv[i][nearest_value],vector_size))
                else:
                    lum_hdv_entry = bundle_hdv(lum_hdv_entry, decode_hdv_to_array(params_hdv[i][nearest_value],vector_size))

            i += 1

        if lum_hdv_prot == None:
            lum_hdv_prot = lum_hdv_entry
        else:
            if bundleCount >= 100:
                normalize_hdv(lum_hdv_entry,vector_size)
                lum_hdv_prot = bundle_hdv(lum_hdv_prot, lum_hdv_entry)
                normalize_hdv(lum_hdv_prot,vector_size)
                bundleCount = 0
            else:
                if isinstance(lum_hdv_prot,bytearray): lum_hdv_prot = decode_hdv_to_array(lum_hdv_prot, vector_size)
                normalize_hdv(lum_hdv_entry,vector_size)
                lum_hdv_prot = bundle_hdv(lum_hdv_prot, lum_hdv_entry)
                bundleCount += 1

        lum_hdv_entry = None
        i = 0

    normalize_hdv(lum_hdv_prot,vector_size)
    return lum_hdv_prot

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
        generate_hdvs(args.data_file, int(args.vector_size), parameters, ranges, m_levels)
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
    df_train = process_data(args.data_file, parameters, None, None)
    df_train = df_train.dropna()
    df_train = df_train.drop(columns=['src','timestamp'])

    ''' ---- CLASSIFIER ENCODING ---- '''

    lum_hdvs = {} # key = class, value = hdv
    time_init_train = time.time()
    lum_hdvs[15] = lum_to_hdv_levels(df_train, int(args.vector_size))
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

    ''' ---- TESTING ---- '''

    print("Testing...")
    df_test_real = process_data("/home/victor/hdvExamples/lum15Test.json", parameters, None, None)
    df_test_real = df_test_real.dropna()
    df_test_real['classification'] = 'r'
    df_test_synth = process_data(args.synthetic_file, parameters, None, None)
    df_test_synth['classification'] = 's'
    df_test_synth = df_test_synth.dropna()
    df_test_synth = df_test_synth.sample(n=int(len(df_test_real)*0.1), random_state=42)
    print(f"Num real:{len(df_test_real)} Num synth:{len(df_test_synth)}")
    df_test = pd.concat([df_test_real, df_test_synth], ignore_index=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.dropna()
    df_test = df_test[df_test["src"] == 15]

    i = 0
    score_values_threshold_80 = []
    score_values_threshold_85 = []
    score_values_threshold_90 = []
    score_values_threshold_95 = []
    cosine_values = []
    start = time.time()
    for row in df_test.itertuples():
        hdv = entry_to_hdv_levels(row,int(args.vector_size))
        if lum_hdvs[15] == None:
            continue
        cos = cosine_similarity(lum_hdvs[15],hdv)

        if cos >= 0.95: score_values_threshold_95.append('r')
        else: score_values_threshold_95.append('s')
        if cos >= 0.90: score_values_threshold_90.append('r')
        else: score_values_threshold_90.append('s')
        if cos >= 0.85: score_values_threshold_85.append('r')
        else: score_values_threshold_85.append('s')
        if cos >= 0.80: score_values_threshold_80.append('r')
        else: score_values_threshold_80.append('s')
        cosine_values.append(cos)
        i+= 1

    score95 = 0
    score90 = 0
    score85 = 0
    score80 = 0
    for j in range(len(score_values_threshold_95)):
        comp = df_test['classification'].iloc[j]
        if score_values_threshold_95[j] == comp: score95 += 1
        if score_values_threshold_90[j] == comp: score90 += 1
        if score_values_threshold_85[j] == comp: score85 += 1
        if score_values_threshold_80[j] == comp: score80 += 1
    precision95 = score95 / (len(score_values_threshold_95))
    precision90 = score90 / (len(score_values_threshold_90))
    precision85 = score85 / (len(score_values_threshold_85))
    precision80 = score80 / (len(score_values_threshold_80))


    finish = time.time()

    print(f'{precision80}, {precision85}, {precision90}, {precision95}')
    print(f"time_testing_elapsed:{finish-start}")

    return 
   
if __name__ == "__main__":
    main()
