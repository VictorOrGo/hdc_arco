import argparse
import sys
from datetime import datetime
from typing import List, Dict, Any, Set
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from json_processor import get_source_address, get_timestamp
import json

def extract_parameters(entry: Dict[str, Any], parameter_paths: List[str]) -> Dict[str, Any]:
    """
    Extracts specified parameters from a JSON entry using hierarchical paths.
    Returns a dictionary with parameter names and their values.
    """
    extracted = {}
    for path in parameter_paths:
        keys = path.split('.')
        value = entry
        try:
            for key in keys:
                value = value[key]
            extracted[path] = value
        except (KeyError, TypeError):
            continue 
    return extracted

def extract_parameters_data_type2(entry: Dict[str, Any], parameter_paths: List[str]) -> Dict[str, Any]:
    """
    Extracts specified parameters from a JSON entry using hierarchical paths.
    Returns a dictionary with parameter names and their values.
    """
    extracted = {}
    for path in parameter_paths:
        value = entry
        try:
            value = entry[path]
            extracted[path] = value
        except (KeyError, TypeError):
            continue 
    return extracted

def parse_datetime(datetime_str: str) -> datetime:
    """
    Parses a datetime string in ISO format and returns a datetime object.
    """
    try:
        return datetime.fromisoformat(datetime_str)
    except ValueError:
        raise ValueError(f"Invalid datetime format: '{datetime_str}'. Use ISO format YYYY-MM-DDTHH:MM:SS.")

def discover_all_parameters(input_file: str) -> Set[str]:

    parameter_paths = set()

    def traverse(entry: Dict[str, Any], parent_key: str = ''):
        for key, value in entry.items():
            # Skip 'trace_options', 'source_address', and 'timestamp' as they are handled separately
            if key in ['trace_options', 'source_address', 'timestamp', 'src', 'unknown_field', 'Tx_power_table']:
                continue
            new_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, dict):
                traverse(value, new_key)
            elif isinstance(value, str):
                continue
            else:
                parameter_paths.add(new_key)

    with open(input_file, 'r') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")

            traverse(entry)
            
    return parameter_paths

def process_data(input_file: str, parameter_paths: List[str], start_dt: datetime = None, end_dt: datetime = None) -> pd.DataFrame:
    """
    Reads the input file line by line and parses JSON objects.
    Returns a list of JSON objects.
    """
    df = pd.DataFrame()
    data = []
    timestamp_list = []
    with open(input_file, 'r') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")
            
            timestamp_str = get_timestamp(entry)
            if not timestamp_str:
                continue  # Skip if no timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                print(f"Warning: Skipping entry with invalid timestamp format: {timestamp_str}")
                continue
                
            # Filter by date range if specified
            if start_dt and timestamp < start_dt:
                continue
            if end_dt and timestamp > end_dt:
                continue

            source_addr_diag = get_source_address(entry)
            source_addr_dat = entry.get("src")
            if source_addr_diag is None and source_addr_dat is None:
                continue  # Skip if no source_address

            params = extract_parameters(entry, parameter_paths)
            if source_addr_diag is None:
                params['src'] = source_addr_dat
            else:
                params['source_address'] = source_addr_diag

            # Convertir las nuevas filas a un DataFrame
            data.append(params)
            timestamp_list.append(timestamp)

    df = pd.DataFrame(data, dtype='float32')
    df['timestamp'] = timestamp_list
    return df

def process_data_type2(input_file: str, parameter_paths: List[str], start_dt: datetime = None, end_dt: datetime = None) -> pd.DataFrame:
    """
    Reads the input file line by line and parses JSON objects.
    Returns a list of JSON objects.
    """
    df = pd.DataFrame()
    data = []
    timestamp_list = []
    with open(input_file, 'r') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")
            
            timestamp_str = get_timestamp(entry)
            if not timestamp_str:
                continue  # Skip if no timestamp
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                print(f"Warning: Skipping entry with invalid timestamp format: {timestamp_str}")
                continue
                
            # Filter by date range if specified
            if start_dt and timestamp < start_dt:
                continue
            if end_dt and timestamp > end_dt:
                continue

            source_addr_diag = get_source_address(entry)
            source_addr_dat = entry.get("src")
            if source_addr_diag is None and source_addr_dat is None:
                continue  # Skip if no source_address

            params = extract_parameters_data_type2(entry, parameter_paths)
            if source_addr_diag is None:
                params['src'] = source_addr_dat
            else:
                params['source_address'] = source_addr_diag

            # Convertir las nuevas filas a un DataFrame
            data.append(params)
            timestamp_list.append(timestamp)

    df = pd.DataFrame(data, dtype='float32')
    df['timestamp'] = timestamp_list
    return df

def save_image_correlation_matrix(df: pd.DataFrame, source_address: int, parameters: List[str], image_output):
    """
    Plots a correlation matrix heatmap for the given source_address.
    """
    try:
        subset = df[df['source_address'] == source_address]
    except:
        subset = df[df['src'] == source_address]
    
    if subset.empty:
        print(f"No data available for source_address {source_address}.")
        return
    
    filtered_parameters = [col for col in parameters if col in subset.columns]
    missing_columns = [col for col in parameters if col not in subset.columns]
    if missing_columns:
        print(f"WARNING: The following columns are not in the dataset: {missing_columns}")
    
    subset_params = subset[filtered_parameters]
    del subset, filtered_parameters, missing_columns

    if subset_params.empty:
        print(f"No parameter data available for source_address {source_address}.")
        return

    if subset_params.shape[0] < 2:
        print(f"Not enough data points to compute correlations for source_address {source_address}.")
        return
    
    corr = subset_params.corr()
    del subset_params

    plt.figure(figsize=(16, 9))
    plt.subplots_adjust(top=0.97, bottom=0.27, left=0.23, right=0.9)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f'Correlation Matrix for Source Address {source_address}')
    #plt.show()
    name = image_output + "/correlation_plot_" + str(source_address) + ".png"
    try:
        plt.savefig(name)
    except FileNotFoundError:
        print(f"Error: Directory {image_output} not found")
        sys.exit(1)

def json_correlation_matrix(df: pd.DataFrame, source_address: int, parameters: List[str]):
    """
    Saves in a JSON file the correlation matrix for the given source_address.
    """
    try:
        subset = df[df['source_address'] == source_address]
    except:
        subset = df[df['src'] == source_address]
    
    if subset.empty:
        print(f"No data available for source_address {source_address}.")
        return
    
    filtered_parameters = [col for col in parameters if col in subset.columns]
    missing_columns = [col for col in parameters if col not in subset.columns]
    if missing_columns:
        print(f"WARNING: The following columns are not in the dataset: {missing_columns}")
    
    subset_params = subset[filtered_parameters]

    if subset_params.empty:
        print(f"No parameter data available for source_address {source_address}.")
        return

    if subset_params.shape[0] < 2:
        print(f"Not enough data points to compute correlations for source_address {source_address}.")
        return
    
    corr = subset_params.corr()
    return corr

def main():
    parser = argparse.ArgumentParser(description="Visualize correlations of parameters from JSON entries grouped by source_address.")
    parser.add_argument('input_file', help="Path to the input file containing JSON entries.")
    parser.add_argument('--start', help="Start datetime in ISO format (e.g., '2024-12-05T00:00:00').", default=None)
    parser.add_argument('--end', help="End datetime in ISO format (e.g., '2024-12-07T23:59:59').", default=None)
    parser.add_argument('--parameters', nargs='+', help="List of hierarchical parameter paths to analyze, e.g., 'cbmac_details.cbmac_load' 'buffer_usage.average'. If not provided, all available parameters will be used.", default=None)
    parser.add_argument('--image', choices=['1', '0'], default='0', help="1 if you want to save the correlation matrix plot in images.")
    parser.add_argument('--image_output', default=None, help="directory where images will be saved. only used when image is 1")
    parser.add_argument('--json', choices=['1', '0'], default='1', help="1 if you want to save the correlation matrix in a json type file.")
    args = parser.parse_args()

    if args.image == '1':
        if args.image_output == None:
            print(f"Error: Argument image is {args.image} but image_output is {args.image_output}")
            sys.exit(1)

    # Parse date range if provided
    start_dt = None
    end_dt = None
    try:
        if args.start:
            start_dt = parse_datetime(args.start)
        if args.end:
            end_dt = parse_datetime(args.end)
    except ValueError as ve:
        print(f"Error: {ve}")
        sys.exit(1)

    if start_dt and end_dt and start_dt > end_dt:
        print("Error: Start datetime cannot be after end datetime.")
        sys.exit(1)

    # Discover all parameters if none are specified
    if not args.parameters:
        print("No parameters specified. Discovering all available parameters from the data...")
        all_parameters = discover_all_parameters(args.input_file)
        if not all_parameters:
            print("No parameters found in entries.")
            sys.exit(0)
        print(f"Discovered {len(all_parameters)} parameters:")
        for param in sorted(all_parameters):
            print(f" - {param}")
        parameters = sorted(all_parameters)
    else:
        parameters = args.parameters
    
    # Process data
    df = process_data(args.input_file, parameters, start_dt, end_dt)

    if df.empty:
        print("No data available after applying filters.")
        sys.exit(0)

    # Get unique source_addresses
    try:
        source_addresses = df['source_address'].unique()
    except:
        source_addresses = df['src'].unique()

    source_addresses = source_addresses.astype('int16')


    print(f"Found {len(source_addresses)} unique source_address(es): {source_addresses}")

    # Generate plots for each source_address
    correlations = [] # USED FOR DUMPING CORRELATION INTO JSON
    sources = [] # USED FOR DUMPING CORRELATION INTO JSON
    for src in source_addresses:
        print(f"\nGenerating plots for source_address {src}...")
        if args.image == '1':
            save_image_correlation_matrix(df, src, parameters, args.image_output)
        
        if args.json == '1':
            correlations.append(json_correlation_matrix(df, src, parameters)) 
            sources.append(src) 
            
            matrices_dict = {}
            for idx, corr_matrix_df in enumerate(correlations):
                matrices_dict[f"lum_{sources[idx]}"] = corr_matrix_df.to_dict()

            with open("correlation_matrix.json", "w") as json_file:
                json.dump(matrices_dict, json_file, indent=4)
        
if __name__ == "__main__":
    main()
