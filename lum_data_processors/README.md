# How to launch the scripts?
## correlation_matrix.py
Input args:
- `input_file`: Path to the input file containing the JSON entries.  
- `start`: Start date of the search in ISO format (e.g., `2024-12-05T00:00:00`). Not used by default.  
- `end`: End date of the search in ISO format (e.g., `2024-12-05T00:00:00`). Not used by default.  
- `parameters`: List of parameters to analyze (e.g., `cbmac_details.cbmac_load`, `buffer_usage.average`).  
  If none are specified, all parameters will be considered.  
- `image`: `1` to save correlation matrices as an image, `0` otherwise. Default is `0`.  
- `image_output`: Directory where the images will be saved. This argument is required if `image` is set to `1`.  
- `json`: `1` to save correlation matrix data to a JSON file. Default is `1`.

Example usage:

    python3 correlation_matrix.py data/data.json --json 1 --image 1 --image_output figsData --parameters battery_level code hops hours_in_emergency hours_in_power link_quality outputState state times_in_emergency times_in_power travel_ms two_in_one_battery_level --start 2024-12-11T05:00:00 --end 2024-12-11T21:00:00

## extract_normality.py
Input args:

- `input_file`: Path to the input file containing the JSON entries.  
- `output_file`: Name of the file where the filtered results will be saved.  
- `upper_start`: Number where the upper interval starts.  
  For example, for the interval (0.1, 0.9), this value would be `0.1`. Default is `0.6`.  
- `upper_end`: Number where the upper interval ends.  
  For example, for the interval (0.1, 0.9), this value would be `0.9`. Default is `1`.  
- `lower_start`: Number where the lower interval starts.  
  For example, for the interval (-0.9, -0.1), this value would be `-0.1`. Default is `-0.4`.  
- `lower_end`: Number where the lower interval ends.  
  For example, for the interval (-0.9, -0.1), this value would be `-0.9`. Default is `-1`.


Example usage:

        python3 extract_normality.py correlation_matrix.json correlation_matrix_normality_diag.json --upper_start 0.5 --upper_end 1 --lower_start -0.5 --lower_end -1

## lum_processor.py
Input args:

- `input_file`: Path to the input file containing the JSON entries.  
- `output_file`: Name of the file where the filtered results will be saved.  
- `variables_file`: Path to the file containing the variables to be filtered.  
- `depth_level`: Filter depth level. Must be `0` (simple filtering) or `1` (dictionary within a dictionary).


Example usage:

    python3 lum_processor.py data.json results.json lum_processor_filters/lum_data_filter.txt 1

## timestamp.py
Input args:

- `input_file`: Path to the input file containing the JSON entries.  
- `time_lapse`: Time in seconds that must pass to be considered an anomaly.  
  Add 1 extra second for better filtering.


Example usage:

    python3 timestamp.py data.json 208

## hdvLum.py & hdvLumOpt.py

Input args:

- `data_file`: Path to the input file containing JSON entries.  
- `vector_size`: Size of the HDV vectors. Default is `10000`.  
- `already_saved_hdv_matrices`:  `1` → Matrices will **not** be generated; existing matrix files must already exist and match the parameter names.  `0` → Matrices will be created and saved as files. Default is `0`.  
- `already_saved_hdvs_prot`: `1` → HDV prototypes will **not** be generated; existing prototype files must already exist and match the node numbers. `0` → HDV prototypes will be created and saved as files.  Default is `0`.  
- `parameters`: List of hierarchical parameter paths to analyze, e.g. `cbmac_details.cbmac_load buffer_usage.average`. If not provided, all available parameters will be used.  
- `m_levels`: Number of levels. Default is `3336`.  
- `k_classes`: Number of classes. Default is `10`.  
- `output_image`: Name of the output image file.  
- `synthetic_file`: Name of the file containing synthetic data.

Example usage:

    python3 lum_data_processors/hdvLum.py /home/victor/Descargas/new_file.json --already_saved_hdv_matrices 0 --already_saved_hdvs_prot 0 --parameters battery_level hops hours_in_emergency times_in_emergency times_in_power two_in_one_battery_level nexthop_details.advertised_cost nexthop_details.sink_address nexthop_details.next_hop_address state cbmac_details.cbmac_rx_ack_other_reasons buffer_usage.average outputState buffer_usage.maximum nexthop_details.next_hop_quality nexthop_details.next_hop_rssi cluster_channel nexthop_details.next_hop_power cluster_members link_quality hours_in_power travel_ms WBN_rssi_correction_val cbmac_details.cbmac_load cbmac_details.cbmac_rx_messages_ack cbmac_details.cbmac_rx_messages_unack cbmac_details.cbmac_tx_ack_cca_fail cbmac_details.cbmac_tx_ack_not_received cbmac_details.cbmac_tx_messages_ack cbmac_details.cbmac_tx_messages_unack cbmac_details.cbmac_tx_cca_unack_fail network_scans_amount cfmac_pending_broadcast_le_member Unack_broadcast_channel

## merge_json.py

Input args:

- `df_data`: Path to the first JSON dataset.  
- `df_diag`: Path to the second JSON dataset.  
- `output`: Path to the desired output file.

Example usage:

    python3 merge_json.py data1.json data2.json merged_output.json

