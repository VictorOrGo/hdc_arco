import json
from correlation_matrix import extract_parameters_data_type2, process_data_type2
import numpy as np
import pandas as pd

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

            extracted_params = extract_parameters_data_type2(entry, parameters)

            for parameter in parameters:
                if parameter not in extracted_params.keys(): continue
                elif extracted_params[parameter] == None: continue
                elif extracted_params[parameter] > result[parameter]["higher"]:
                    result[parameter]["higher"] = int(extracted_params[parameter] + 10) # We add 10 in order to give a margin of different values
                elif extracted_params[parameter] < result[parameter]["lower"]:
                    result[parameter]["lower"] = int(extracted_params[parameter] - 10) # We substract 10 in order to give a margin of different values

    return result # Dict{dict} // result['hours_in_power'] = {'higher': 20555, 'lower': 44}

parameters = ['battery_level', 'hops', 'hours_in_emergency', 'times_in_emergency', 'times_in_power', 'two_in_one_battery_level', 'nexthop_details.advertised_cost', 'nexthop_details.sink_address', 'nexthop_details.next_hop_address', 'state', 'cbmac_details.cbmac_rx_ack_other_reasons', 'buffer_usage.average', 'outputState', 'buffer_usage.maximum', 'nexthop_details.next_hop_quality', 'nexthop_details.next_hop_rssi', 'cluster_channel', 'nexthop_details.next_hop_power', 'cluster_members', 'link_quality', 'hours_in_power', 'travel_ms', 'WBN_rssi_correction_val', 'cbmac_details.cbmac_load', 'cbmac_details.cbmac_rx_messages_ack', 'cbmac_details.cbmac_rx_messages_unack', 'cbmac_details.cbmac_tx_ack_cca_fail', 'cbmac_details.cbmac_tx_ack_not_received', 'cbmac_details.cbmac_tx_messages_ack', 'cbmac_details.cbmac_tx_messages_unack', 'cbmac_details.cbmac_tx_cca_unack_fail', 'network_scans_amount', 'cfmac_pending_broadcast_le_member', 'Unack_broadcast_channel']

file = "/home/victor/hdvExamples/lum15Train.json"
ranges = lower_and_higher_value(file,parameters)

df = process_data_type2('/home/victor/hdvExamples/lum15Test.json', parameters, None, None)
result_df = None
while parameters:
    df_aux = df.head(100).copy()
    for i in range(len(df_aux)):
        for parameter in parameters:
            x = df_aux.at[i, parameter]
            if pd.isna(x):
                pass
            else: 
                std = abs(1 * x)  # 0.1 = 10% de desviación estándar
                if std == 0: std = 1

                random_val = np.random.normal(loc=x, scale=std)
                df_aux.at[i, parameter] = round(abs(random_val))
    result_df = pd.concat([result_df, df_aux], ignore_index=True)
    parameters.pop(0)
            

result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
result_df['timestamp'] = result_df['timestamp'].apply(lambda x: x.isoformat())
result_df.to_json('lum15TestSyntheticDes1.json', orient='records', lines=True)

