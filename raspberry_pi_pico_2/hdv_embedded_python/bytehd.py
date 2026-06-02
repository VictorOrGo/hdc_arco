import array
import random
import sys
import gc
from math import sqrt
import time

# D = 1000 M = 92
# D = 1500 M = 65
# D = 2000 M = 48
# D = 2500 M = 35
# D = 3000 M = 26
# D = 3500 M = 22
# D = 4000 M = 20


HDV_DIM = 4000
M = 13
CODE_MAP = {1: 0b0, -1: 0b1} # Codification of the different values that our HDV can take
DECODE_MAP = {0b0: 1, 0b1: -1}
NUMBERS_IN_A_BYTE = 8

RANGES = {'travel_ms': {'higher': 10.3422309835, 'lower': -11.129908607600001}, 'hops': {'higher': 9.9666576071, 'lower': -10.5285593448}, 'hours_in_emergency': {'higher': 10.7978661854, 'lower': -9.2021338146}, 'hours_in_power': {'higher': 10.5583666203, 'lower': -10.705300119}, 'times_in_emergency': {'higher': 11.0402503322, 'lower': -8.9597496678}, 'times_in_power': {'higher': 10.1804426598, 'lower': -9.8195573402}, 'battery_level': {'higher': 9.8756683545, 'lower': -9.3706225416}, 'link_quality': {'higher': 10.733673815, 'lower': -9.266326185}, 'WBN_rssi_correction_val': {'higher': 10.9541171806, 'lower': -11.7412827994}, 'cbmac_blacklisting_channels_min_to_40': {'higher': 9.681431759, 'lower': -10.2810097094}, 'cluster_channel': {'higher': 10.8128270088, 'lower': -9.6426633002}, 'scanstat_avg_routers': {'higher': 9.9393908084, 'lower': -10.0606091916}, 'network_scans_amount': {'higher': 11.4624569702, 'lower': -8.8622114835}, 'trace_options.sequence': {'higher': 10.7883938603, 'lower': -9.4283203761}, 'cbmac_details.cbmac_load': {'higher': 9.8509129535, 'lower': -11.3532607214}, 'cbmac_details.cbmac_rx_messages_ack': {'higher': 10.8331624216, 'lower': -11.643139228599999}, 'cbmac_details.cbmac_rx_messages_unack': {'higher': 9.2849584411, 'lower': -8.2238746191}, 'cbmac_details.cbmac_rx_ack_other_reasons': {'higher': 9.2607019221, 'lower': -10.7392980779}, 'cbmac_details.cbmac_tx_ack_cca_fail': {'higher': 10.8998948603, 'lower': -9.3067637164}, 'cbmac_details.cbmac_tx_ack_not_received': {'higher': 9.4613816643, 'lower': -8.4917809143}, 'cbmac_details.cbmac_tx_messages_ack': {'higher': 10.2657952991, 'lower': -10.920853884}, 'cbmac_details.cbmac_tx_messages_unack': {'higher': 11.635152705, 'lower': -8.7135986302}, 'cbmac_details.cbmac_tx_cca_unack_fail': {'higher': 11.0521759556, 'lower': -8.9506993171}, 'buffer_usage.average': {'higher': 10.3570059096, 'lower': -12.7949269663}, 'nexthop_details.advertised_cost': {'higher': 10.3007748668, 'lower': -10.2220550246}, 'nexthop_details.next_hop_quality': {'higher': 9.6393607259, 'lower': -8.8687137216}, 'nexthop_details.next_hop_rssi': {'higher': 9.4240177678, 'lower': -9.583995872}, 'nexthop_details.next_hop_power': {'higher': 10.5715385817, 'lower': -9.4284614183}, 'Installation quality.quality_indicator': {'higher': 10.5166860684, 'lower': -9.4833139316}, 'nexthop_details.next_hop_address': {'higher': 203, 'lower': 5}, 'cfmac_pending_broadcast_le_member': {'higher': 34, 'lower': 12}, 'Unack_broadcast_channel': {'higher': 21, 'lower': 6}}
NORMALIZATION_SUM = 100

TYPE_SIZES = {
    int: 28,
    float: 24,
    str: lambda s: 49 + len(s),
    dict: 240,
    bytearray: lambda b: 33 + len(b),
}

def size_of_my_dict(d, seen=None):
    if seen is None:
        seen = set()

    obj_id = id(d)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = TYPE_SIZES.get(dict, 104) 

    per_entry_overhead = 8 

    for k, v in d.items():
        size += per_entry_overhead

        k_type = type(k)
        if k_type in TYPE_SIZES:
            k_size = TYPE_SIZES[k_type](k) if callable(TYPE_SIZES[k_type]) else TYPE_SIZES[k_type]
            size += k_size
        else:
            size += 0  

        if isinstance(v, dict):
            size += size_of_my_dict(v, seen)
        else:
            v_type = type(v)
            if v_type in TYPE_SIZES:
                v_size = TYPE_SIZES[v_type](v) if callable(TYPE_SIZES[v_type]) else TYPE_SIZES[v_type]
                size += v_size
            else:
                size += 0 

    return size

def gen_random_hdv() -> array.array:
    hdv = array.array('b',[0] * HDV_DIM)
    for i in range(HDV_DIM):
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

def decode_hdv_to_array(data:bytearray) -> array.array:
    hdv = array.array('b',[0] * HDV_DIM) # Initialize our HDV
    
    for i in range(HDV_DIM):
        
        byte_index = i // NUMBERS_IN_A_BYTE # 1 byte can store 8 numbers because each number is 1 bit. i is the index in the decoded hdv so this way we can get what byte stores the number. EJ: i = 1 -> 1/8 = 0(byte index), i = 11 -> 11/8 = 1(byte index)
        
        shift = (i % NUMBERS_IN_A_BYTE) # We got the byte index so now we need to retrieve the number. We get the shift needed to get the correct bit. i % NUMBERS_IN_A_BYTE gets us the division of the byte (it has 8 divisions since we use a bit per number).

        code = (data[byte_index] >> shift) & 0b1 # We access the specific byte and apply a shift so we get the number we want at the end. After this we apply a mask in order to extract only the bit we want. EJ: 00000100(byte after shift) & 00000001(mask) = 00000000 = 0b00

        hdv[i] = DECODE_MAP[code] # We decode the bit number
    
    return hdv

def get_number_from_hdv(hdv:bytearray, index:int) -> int:
    if index >= HDV_DIM:
        print(f"ERROR: To get number index {index} must be lower than {HDV_DIM}")

    byte_index = index // NUMBERS_IN_A_BYTE
    shift = (index % NUMBERS_IN_A_BYTE)
    code = (hdv[byte_index] >> shift) & 0b1
    return DECODE_MAP[code]

def set_number_in_hdv(hdv:bytearray, index:int, value:int):
    if index >= HDV_DIM:
        print(f"ERROR: To set number index {index} must be lower than {HDV_DIM}")

    code = CODE_MAP[value]
    byte_index = index // NUMBERS_IN_A_BYTE
    shift = (index % NUMBERS_IN_A_BYTE)

    hdv[byte_index] &= ~(0b1 << shift)  # We delete previous values by applying a mask. We shift our mask so the bits 1 end up where we want to delete the selected number. After that, we invert the mask so 0 becomes 1 and the 1 becomes 0, this way we can apply an AND operation to turn to 0 the bits we want to clean and keep the rest as they were.
    hdv[byte_index] |= (code << shift)   

def bundle_hdv(hdv1:array.array, hdv2:array.array) -> array.array:
    result_hdv = array.array('b', (a + b for a, b in zip(hdv1, hdv2))) # Zip will group each component EJ: [(1, -1), (-1, 1), ... , (0, 0)]. Then we just add the components and create a new array.
    return result_hdv

def normalize_hdv(hdv:array.array) -> array.array:
    for i in range(HDV_DIM):
        number = hdv[i] # We will evaluate number so we save 2 more possible accesses to the array
        if number > 0: hdv[i] = 1 
        else: hdv[i] = -1
    return hdv

def cosine_similarity(hdv1:array.array, hdv2:array.array) -> float:
    dot = sum(x * y for x, y in zip(hdv1, hdv2))
    mag1 = sqrt(sum(x * x for x in hdv1))
    mag2 = sqrt(sum(y * y for y in hdv2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)

def hamming_similarity(hdv1:array.array, hdv2:array.array) -> float:
    sum = 0
    for i in range(HDV_DIM):
        if hdv1[i] == hdv2[i]: sum += 1
    
    return sum/HDV_DIM

def get_sample(lst:list, n_elements:int): # Get a sample of N elements from a list
    lst_copy = lst[:]
    result = []
    for i in range(n_elements):
        x = random.randint(0, len(lst_copy) - 1)
        result.append(lst_copy.pop(x))  
    return result

def range_hdv_levels(higher, lower) -> tuple[dict[float, bytearray],list[float]]:
    matrix = {}
    
    fst_hdv = encode_array_to_hdv(gen_random_hdv()) # First level HDV generated randomly
    matrix[lower] = bytearray(fst_hdv)

    increment = round((abs(higher) + abs(lower)) / M, 4)

    b = round(HDV_DIM / (2 * (M - 1)))  # b = D / 2(M-1)
    if b == 0: b = 1

    used_indices = set()

    keys = [round(lower + i * increment, 4) for i in range(M)]

    for i in range(M):
        key = keys[i]
        
        if i == 0:
            prev_tensor = bytearray(matrix[lower])
        else:
            prev_tensor = bytearray(matrix[keys[i - 1]])

        available = list(set(range(HDV_DIM)) - used_indices)
        if len(available) < b:
            used_indices = set()
            available = list(range(HDV_DIM))
        
        selected = get_sample(available, b)
        used_indices.update(selected)

        for idx in selected:
            prev_value = get_number_from_hdv(prev_tensor, idx)
            set_number_in_hdv(prev_tensor, idx, prev_value*(-1))

        matrix[key] = bytearray(prev_tensor)

    return matrix,keys


random.seed(42)
time_start_total = time.ticks_ms()

# while True:
#     print(f"M = {M}")
#     hdv_matrices = {} 
#     keys = None
#     column_names = ['travel_ms', 'hops', 'hours_in_emergency', 'hours_in_power', 'times_in_emergency', 'times_in_power', 'battery_level', 'link_quality', 'WBN_rssi_correction_val', 'cbmac_blacklisting_channels_min_to_40', 'cluster_channel', 'scanstat_avg_routers', 'network_scans_amount', 'trace_options.sequence', 'cbmac_details.cbmac_load', 'cbmac_details.cbmac_rx_messages_ack', 'cbmac_details.cbmac_rx_messages_unack', 'cbmac_details.cbmac_rx_ack_other_reasons', 'cbmac_details.cbmac_tx_ack_cca_fail', 'cbmac_details.cbmac_tx_ack_not_received', 'cbmac_details.cbmac_tx_messages_ack', 'cbmac_details.cbmac_tx_messages_unack', 'cbmac_details.cbmac_tx_cca_unack_fail', 'buffer_usage.average', 'nexthop_details.advertised_cost', 'nexthop_details.next_hop_quality', 'nexthop_details.next_hop_rssi', 'nexthop_details.next_hop_power', 'Installation quality.quality_indicator', 'nexthop_details.next_hop_address', 'cfmac_pending_broadcast_le_member', 'Unack_broadcast_channel', 'classification']
#     gc.collect()

#     for param in column_names[:-1]: # We don't create HDV for the classification column
#         hdv_matrices[param],order = range_hdv_levels(RANGES[param]['higher'], RANGES[param]['lower'])
#     M += 1

hdv_matrices = {} 
keys = None
column_names = ['travel_ms', 'hops', 'hours_in_emergency', 'hours_in_power', 'times_in_emergency', 'times_in_power', 'battery_level', 'link_quality', 'WBN_rssi_correction_val', 'cbmac_blacklisting_channels_min_to_40', 'cluster_channel', 'scanstat_avg_routers', 'network_scans_amount', 'trace_options.sequence', 'cbmac_details.cbmac_load', 'cbmac_details.cbmac_rx_messages_ack', 'cbmac_details.cbmac_rx_messages_unack', 'cbmac_details.cbmac_rx_ack_other_reasons', 'cbmac_details.cbmac_tx_ack_cca_fail', 'cbmac_details.cbmac_tx_ack_not_received', 'cbmac_details.cbmac_tx_messages_ack', 'cbmac_details.cbmac_tx_messages_unack', 'cbmac_details.cbmac_tx_cca_unack_fail', 'buffer_usage.average', 'nexthop_details.advertised_cost', 'nexthop_details.next_hop_quality', 'nexthop_details.next_hop_rssi', 'nexthop_details.next_hop_power', 'Installation quality.quality_indicator', 'nexthop_details.next_hop_address', 'cfmac_pending_broadcast_le_member', 'Unack_broadcast_channel', 'classification']
gc.collect()
free_mem = gc.mem_free()
used_mem = gc.mem_alloc()
total_mem = free_mem + used_mem

for param in column_names[:-1]: # We don't create HDV for the classification column
    hdv_matrices[param],order = range_hdv_levels(RANGES[param]['higher'], RANGES[param]['lower'])
time_end_matrix_creation = time.ticks_ms()
time_elapsed_matrix_creation = time.ticks_diff(time_end_matrix_creation, time_start_total) /1000
gc.collect()

print("Matrices created")

# print("Memoria libre:", gc.mem_free(), "bytes")
# print("Memoria usada:", gc.mem_alloc(), "bytes")
# print("Diferencia de memoria:", abs(free_mem-gc.mem_free()), "bytes")
# size = size_of_my_dict(hdv_matrices)
# print("Memoria estimada:", size, "bytes")

'''-------------------------RECEIVE THRESHOLD-------------------------'''
print("Waiting for threshold")
line = sys.stdin.readline()
if line:
    threshold = float(line.strip())
    print(f"Threshold set to: {threshold}")


'''-------------------------TRAINING-------------------------'''
hdv_prot = None
bundle_sum = 0
print("Training")

while True:
    line = sys.stdin.readline()
    if not line:
        time.sleep(0.1)
        continue
    if line == 'stop\n': # The loop ends when the connection with the PC ends
        break
    values = line.strip().split(',')
    if len(values) != len(column_names)-1:
        print("Row differs from expected")
        continue

    # Pair columns with values using zip
    row_dict = dict(zip(column_names[:-1], values))

    hdv_entry = None
    for key, value in row_dict.items(): # HDV data entry
        indices = hdv_matrices[key].keys()
        nearest_value = min(indices, key=lambda v: abs(v - float(value)))
        if hdv_entry == None:
            hdv_entry = decode_hdv_to_array(hdv_matrices[key][nearest_value])
        else:
            aux = decode_hdv_to_array(hdv_matrices[key][nearest_value])
            hdv_entry = bundle_hdv(hdv_entry, aux)
    
    hdv_entry = normalize_hdv(hdv_entry)

    if hdv_prot == None: # Firts loop iteration
        hdv_prot = hdv_entry
    else:
        if bundle_sum >= NORMALIZATION_SUM: # Normalization of the HDV 
            hdv_prot = bundle_hdv(hdv_prot, hdv_entry)
            hdv_prot = normalize_hdv(hdv_prot)
            bundle_sum = 0
        else: # No normalization of the HDV
            hdv_prot = bundle_hdv(hdv_prot, hdv_entry)
            bundle_sum += 1
    
    hdv_entry = None
    
if hdv_prot != None: hdv_prot = normalize_hdv(hdv_prot)

time_end_training = time.ticks_ms()
time_elapsed_training = time.ticks_diff(time_end_training, time_end_matrix_creation) /1000

'''-------------------------TESTING-------------------------'''

print("Training finished. Starting classification.")
total = 0
correct_real = 0
correct_synthetic = 0
real_entry = 0
synthetic_entry = 0
real_classification = None
while True:
    line = sys.stdin.readline()
    if not line:
        time.sleep(0.1)
        continue
    if line == 'stop\n': # The loop ends when the connection with the PC ends
        break
    values = line.strip().split(',')
    if len(values) != len(column_names): 
        print("Row differs from expected")
        print(values)
        print("----------------------------------------------")
        print(line)
        continue

    # Pair columns with values using zip
    row_dict = dict(zip(column_names, values))

    hdv_entry = None
    for param, value in row_dict.items(): # HDV data entry
        if value == None:
            continue
        if param == "classification" :
            real_classification = value
            if real_classification == 'r':
                real_entry += 1
            else:
                synthetic_entry += 1
            continue
        indices = hdv_matrices[param].keys()
        nearest_value = min(indices, key=lambda v: abs(v - float(value)))
        if hdv_entry == None:
            hdv_entry = decode_hdv_to_array(hdv_matrices[param][nearest_value])
        else:
            aux = decode_hdv_to_array(hdv_matrices[param][nearest_value])
            hdv_entry = bundle_hdv(hdv_entry, aux)

    if hdv_entry != None and hdv_prot != None:
        hdv_entry = normalize_hdv(hdv_entry)
        cos = cosine_similarity(hdv_prot, hdv_entry)
        if cos >= threshold and real_classification == 'r': correct_real += 1
        elif cos < threshold and real_classification == 's': correct_synthetic += 1
        total +=1
        
    gc.collect()

time_end_testing = time.ticks_ms()
time_elapsed_testing = time.ticks_diff(time_end_testing, time_end_training) /1000
time_elapsed_total= time.ticks_diff(time_end_testing, time_start_total) /1000

print(f"Time training: {time_elapsed_training}s")
print(f"Time testing: {time_elapsed_testing}s")
print(f"Accuracy: {(correct_real+correct_synthetic)/total}")
print(f"True positive rate (TPR): {correct_real/real_entry}")
print(f"True negative rate (TNR): {correct_synthetic/synthetic_entry}")
print(f"False positive rate (FPR): {1 - (correct_synthetic/synthetic_entry)}")
print(f"False negative rate (FNR): {1 - (correct_real/real_entry)}")
print(f"Precision (PPV): {correct_real/(correct_real + (synthetic_entry - correct_synthetic))}")
print(f"Negative predictive value (NPV): {correct_synthetic/(correct_synthetic + (real_entry - correct_real))}")
print(f"Balanced accuracy (BA): {(correct_real/real_entry + correct_synthetic/synthetic_entry)/2}")
print(f"F1 score: {2 * (correct_synthetic+correct_real) / (2*total)}")
print(f"time matrix creation(s): {time_elapsed_matrix_creation}")
print(f"time training(s): {time_elapsed_training}")
print(f"time testing(s): {time_elapsed_testing}")
print(f"time total:{time_elapsed_total}")
print("Memoria libre:", gc.mem_free(), "bytes")
print("Memoria usada:", gc.mem_alloc(), "bytes")