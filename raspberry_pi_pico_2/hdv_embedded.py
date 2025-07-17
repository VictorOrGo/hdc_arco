import array
import random
import sys
import gc
from math import sqrt
import time

HDV_DIM = 1000
M = 74
CODE_MAP = {1: 0b0, -1: 0b1} # Codification of the different values that our HDV can take
DECODE_MAP = {0b0: 1, 0b1: -1}
NUMBERS_IN_A_BYTE = 8

RANGES = {'battery_level': {'higher': 4102, 'lower': 2810}, 'hops': {'higher': 11, 'lower': -9}, 'hours_in_emergency': {'higher': 10, 'lower': -10}, 'times_in_emergency': {'higher': 10, 'lower': -10}, 'times_in_power': {'higher': 10, 'lower': -10}, 'two_in_one_battery_level': {'higher': 10, 'lower': -10}, 'nexthop_details.advertised_cost': {'higher': 12, 'lower': -8}, 'nexthop_details.sink_address': {'higher': 1010, 'lower': 990}, 'nexthop_details.next_hop_address': {'higher': 1010, 'lower': 990}, 'state': {'higher': 11, 'lower': -9}, 'cbmac_details.cbmac_rx_ack_other_reasons': {'higher': 16617, 'lower': 15798}, 'buffer_usage.average': {'higher': 17, 'lower': -3}, 'outputState': {'higher': 16651, 'lower': 56}, 'buffer_usage.maximum': {'higher': 31, 'lower': -2}, 'nexthop_details.next_hop_quality': {'higher': 263, 'lower': 244}, 'nexthop_details.next_hop_rssi': {'higher': -1, 'lower': -68}, 'cluster_channel': {'higher': 46, 'lower': -7}, 'nexthop_details.next_hop_power': {'higher': 10, 'lower': -14}, 'cluster_members': {'higher': 15, 'lower': -5}, 'link_quality': {'higher': 265, 'lower': 112}, 'hours_in_power': {'higher': 4053, 'lower': 3504}, 'travel_ms': {'higher': 166, 'lower': -10}, 'WBN_rssi_correction_val': {'higher': -1, 'lower': -14}, 'cbmac_details.cbmac_load': {'higher': 31, 'lower': -3}, 'cbmac_details.cbmac_rx_messages_ack': {'higher': 65535, 'lower': -5}, 'cbmac_details.cbmac_rx_messages_unack': {'higher': 26917, 'lower': 10758}, 'cbmac_details.cbmac_tx_ack_cca_fail': {'higher': 65541, 'lower': -4}, 'cbmac_details.cbmac_tx_ack_not_received': {'higher': 65536, 'lower': 5}, 'cbmac_details.cbmac_tx_messages_ack': {'higher': 65538, 'lower': -3}, 'cbmac_details.cbmac_tx_messages_unack': {'higher': 64216, 'lower': 52374}, 'cbmac_details.cbmac_tx_cca_unack_fail': {'higher': 37410, 'lower': 36942}, 'network_scans_amount': {'higher': 153, 'lower': 98}, 'cfmac_pending_broadcast_le_member': {'higher': 41, 'lower': 21}, 'Unack_broadcast_channel': {'higher': 30, 'lower': 8}}
THRESHOLD = 0.8
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

time_start_total = time.ticks_ms()

hdv_matrices = {} 
keys = None
column_names = ['battery_level', 'hops', 'hours_in_emergency', 'times_in_emergency', 'times_in_power', 'two_in_one_battery_level', 'state', 'outputState', 'cluster_channel', 'cluster_members', 'link_quality', 'hours_in_power', 'travel_ms', 'WBN_rssi_correction_val', 'network_scans_amount', 'cfmac_pending_broadcast_le_member', 'Unack_broadcast_channel', "classification"]
gc.collect()
free_mem = gc.mem_free()
used_mem = gc.mem_alloc()
total_mem = free_mem + used_mem
for param in column_names[:-1]:
    hdv_matrices[param],order = range_hdv_levels(RANGES[param]['higher'], RANGES[param]['lower'])
time_end_matrix_creation = time.ticks_ms()
time_elapsed_matrix_creation = time.ticks_diff(time_end_matrix_creation, time_start_total) /1000
gc.collect()
print("Matrices created")
print("Memoria libre:", gc.mem_free(), "bytes")
print("Memoria usada:", gc.mem_alloc(), "bytes")
print("Diferencia de memoria:", abs(free_mem-gc.mem_free()), "bytes")
size = size_of_my_dict(hdv_matrices)
print("Memoria estimada:", size, "bytes")

'''-------------------------TRAINING-------------------------'''
hdv_prot = None
bind_sum = 0
print("Training")
while True:
    line = sys.stdin.readline()
    if not line:
        time.sleep(0.1)
        continue
    if line == 'stop\n': # The loop ends when the connection with the PC ends
        break
    values = line.strip().split(',')
    if len(values) != len(column_names[:-1]):
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
    
    if hdv_prot == None: # Firts loop iteration
        hdv_prot = hdv_entry

    elif bind_sum >= NORMALIZATION_SUM and hdv_prot != None and hdv_entry != None: # Normalization of the HDV 
        hdv_entry = normalize_hdv(hdv_entry)
        hdv_prot = bundle_hdv(hdv_prot, hdv_entry)
        hdv_prot = normalize_hdv(hdv_prot)
        bind_sum = 0

    elif hdv_prot != None and hdv_entry != None: # No normalization of the HDV
        hdv_entry = normalize_hdv(hdv_entry)
        hdv_prot = bundle_hdv(hdv_prot, hdv_entry)
        bind_sum += 1
    
if hdv_prot != None: hdv_prot = normalize_hdv(hdv_prot)

time_end_training = time.ticks_ms()
time_elapsed_training = time.ticks_diff(time_end_training, time_end_matrix_creation) /1000
'''-------------------------TESTING-------------------------'''

print("Training finished. Starting classification.")
total = 0
real_classification = []
results85 = []
results90 = []
results95 = []
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
        if param == "classification" :
            real_classification.append(value)
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
        if cos >= 0.85: results85.append('r')
        else: results85.append('s')
        if cos >= 0.9: results90.append('r')
        else: results90.append('s')
        if cos >= 0.95: results95.append('r')
        else: results95.append('s')
        
        total +=1

time_end_testing = time.ticks_ms()
time_elapsed_testing = time.ticks_diff(time_end_testing, time_end_training) /1000
time_elapsed_total= time.ticks_diff(time_end_testing, time_start_total) /1000

correct85 = 0
correct90 = 0
correct95 = 0
for j in range(len(results85)):
    if real_classification[j] == results85[j]: correct85 += 1
    if real_classification[j] == results90[j]: correct90 += 1
    if real_classification[j] == results95[j]: correct95 += 1
    
print(f"accuracy85:{correct85/total}")
print(f"accuracy90:{correct90/total}")
print(f"accuracy95:{correct95/total}")
print(f"time matrix creation(s): {time_elapsed_matrix_creation}")
print(f"time training(s): {time_elapsed_training}")
print(f"time testing(s): {time_elapsed_testing}")
print(f"time total:{time_elapsed_total}")
print("Memoria libre:", gc.mem_free(), "bytes")
print("Memoria usada:", gc.mem_alloc(), "bytes")