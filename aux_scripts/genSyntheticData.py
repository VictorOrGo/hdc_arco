import json
import numpy as np
import pandas as pd


def calculate_new_value(value, alpha): # heteroscedastic noise ε∼N(0​,(α∣Xreal​∣)^2)
    sigma = alpha * abs(value) 
    epsilon = np.random.normal(0, sigma)
    return value + epsilon

parameters = ['travel_ms', 'hops', 'hours_in_emergency', 'hours_in_power', 'times_in_emergency', 'times_in_power', 'battery_level', 'link_quality', 'WBN_rssi_correction_val', 'cbmac_blacklisting_channels_min_to_40', 'cluster_channel', 'scanstat_avg_routers', 'network_scans_amount', 'trace_options.sequence', 'cbmac_details.cbmac_rx_messages_ack', 'cbmac_details.cbmac_rx_messages_unack', 'cbmac_details.cbmac_rx_ack_other_reasons', 'cbmac_details.cbmac_tx_ack_cca_fail', 'cbmac_details.cbmac_tx_ack_not_received', 'cbmac_details.cbmac_tx_messages_ack', 'cbmac_details.cbmac_tx_messages_unack', 'cbmac_details.cbmac_tx_cca_unack_fail', 'nexthop_details.advertised_cost', 'nexthop_details.next_hop_quality', 'nexthop_details.next_hop_rssi', 'nexthop_details.next_hop_address', 'cfmac_pending_broadcast_le_member', 'Unack_broadcast_channel']

file = "lum_75_test.json"

print("Processing Data...")

batch_size = 100_000 
batches = []  

for i, line in enumerate(open(file, "r", encoding="utf-8"), start=1):
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        batches.append(obj)
    except json.JSONDecodeError:
        continue

    if len(batches) >= batch_size:
        batch_df = pd.json_normalize(batches)
        if i == batch_size:
            data = batch_df  
        else:
            data = pd.concat([data, batch_df], ignore_index=True)
        batches.clear()
        print(f"Procesadas {i:,} líneas...")

if batches:
    batch_df = pd.json_normalize(batches)
    data = pd.concat([data, batch_df], ignore_index=True) if "data" in locals() else batch_df

np.random.seed(42)

df_aux = data.copy()

for param in parameters:
    des_std = data[param].std()
    if param in data.columns:
        alpha = 0.01  # Light noise	0.01 – 0.05 // Medium noise	0.05 – 0.2. // Strong noise	0.2 – 0.5 
        df_aux[param] = data[param].apply(lambda x: calculate_new_value(x, alpha) if pd.notnull(x) else x)

df_aux.to_json('lum_75_Test_Synthetic_Alpha_005.json', orient='records', lines=True)

df_aux = data.copy()

for param in parameters:
    des_std = data[param].std()
    if param in data.columns:
        alpha = 0.2  
        df_aux[param] = data[param].apply(lambda x: calculate_new_value(x, alpha) if pd.notnull(x) else x)

df_aux.to_json('lum_75_Test_Synthetic_Alpha_02.json', orient='records', lines=True)

df_aux = data.copy()

for param in parameters:
    des_std = data[param].std()
    if param in data.columns:
        alpha = 0.5  
        df_aux[param] = data[param].apply(lambda x: calculate_new_value(x, alpha) if pd.notnull(x) else x)

df_aux.to_json('lum_75_Test_Synthetic_Alpha_05.json', orient='records', lines=True)
