# json_processor.py

import json
import sys
import re
from datetime import datetime
from typing import Any, Dict, List, Set, Optional

import re
from typing import Set

def parse_ids(id_string: str) -> Set[int]:
    """
    Parses an ID string that can include:
    - Single IDs: "1"
    - Comma-separated IDs: "1,2,3"
    - Ranges: "1-5"
    - Mixed formats: "1, 2-4, 6, [7, 8], 10-12"
    
    Returns a set of integer IDs.
    """
    ids = set()
    try:
        # Elimina corchetes y espacios
        cleaned = re.sub(r'[\[\]\s]', '', id_string)
        # Divide la cadena por comas
        parts = cleaned.split(',')

        for part in parts:
            if '-' in part:  # Detecta un rango
                start, end = map(int, part.split('-'))
                if start > end:
                    raise ValueError(f"Start of range '{start}' cannot be greater than end '{end}'.")
                ids.update(range(start, end + 1))
            else:  # Es un ID único
                ids.add(int(part))
    except ValueError as ve:
        raise ValueError(f"Invalid ID or range format: {ve}")
    
    return ids

def parse_datetime(datetime_str: Optional[str]) -> Optional[datetime]:
    """
    Parses a datetime string in ISO format and returns a datetime object.
    If the string is None, returns None.
    """
    if datetime_str is None:
        return None
    try:
        return datetime.fromisoformat(datetime_str)
    except ValueError:
        raise ValueError(f"Invalid datetime format: '{datetime_str}'. Use ISO format YYYY-MM-DDTHH:MM:SS.")


def read_entries(input_file: str) -> List[Dict[str, Any]]:
    """
    Reads the input file line by line and parses JSON objects.
    Returns a list of JSON objects.
    """
    entries = []
    with open(input_file, 'r') as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_number}: {e}")
    return entries

def filter_entries(entries: List[Dict[str, Any]], id_set: Set[int]) -> List[Dict[str, Any]]:
    """
    Filters entries where 'src' or 'source_address' matches any ID in id_set.
    Handles multiple message formats.
    """
    filtered = []
    for entry in entries:
        # Determine which key to use for source ID based on message type
        src = entry.get("src") or entry.get("source_address")
        if src in id_set:
            filtered.append(entry)
    return filtered

def filter_type1_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters entries where 'trace_type' is 1, indicating Type 1 entries.
    """
    filtered = []
    for entry in entries:
        trace_options = entry.get("trace_options", {})
        if trace_options.get("trace_type") == 1:
            filtered.append(entry)
    return filtered

def write_entries(entries: List[Dict[str, Any]], output_file: str) -> None:
    """
    Writes the list of JSON objects to the output file, one per line.
    """
    with open(output_file, 'w') as outfile:
        for entry in entries:
            outfile.write(json.dumps(entry) + '\n')

def extract_parameter(entry: Dict[str, Any], parameter_path: str) -> Optional[Any]:
    """
    Extracts the value of a parameter from a JSON object using a hierarchical path.
    Example: 'cbmac_details.cbmac_load'
    """
    keys = parameter_path.split('.')
    value = entry
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return None

def get_all_parameters(entries: List[Dict[str, Any]], parameter_path: str) -> List[Dict[str, Any]]:
    """
    Extracts the parameter and timestamp from each entry.
    Returns a list of dictionaries with 'timestamp' and the filtered parameter.
    """
    data = []
    for entry in entries:
        timestamp = entry.get("timestamp")
        parameter = extract_parameter(entry, parameter_path)
        if timestamp and parameter is not None:
            data.append({"timestamp": timestamp, f"{parameter_path}": parameter, "src": entry.get("src") or entry.get("source_address")})
    return data

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
            extracted[path] = None  # Assign None if path doesn't exist
    return extracted

def get_source_address(entry: Dict[str, Any]) -> Optional[int]:
    """
    Retrieves the 'source_address' from an entry.
    """
    return entry.get("source_address")

def get_timestamp(entry: Dict[str, Any]) -> Optional[str]:
    """
    Retrieves the 'timestamp' from an entry.
    """
    return entry.get("timestamp")

def parse_inputs(ids, start, end):
    """
    Parses IDs, start and end datetimes. Handles validation.
    """
    try:
        id_set = parse_ids(ids)
        if not id_set:
            raise ValueError("No valid IDs provided.")
    except ValueError as ve:
        print(f"Error: {ve}")
        sys.exit(1)
    
    try:
        start_dt = parse_datetime(start)
        end_dt = parse_datetime(end)
        if start_dt and end_dt and start_dt > end_dt:
            raise ValueError("Start datetime cannot be after end datetime.")
    except ValueError as ve:
        print(f"Error: {ve}")
        sys.exit(1)

    return id_set, start_dt, end_dt

def load_and_filter_entries(input_file, ids, start_dt=None, end_dt=None):
    """
    Reads and filters entries based on IDs and optional date range.
    """
    entries = read_entries(input_file)
    filtered = filter_entries(entries, ids)

    if not filtered:
        print("No matching entries found for the specified IDs.")
        sys.exit(0)
    
    if start_dt or end_dt:
        filtered = [
            entry for entry in filtered 
            if "timestamp" in entry and _within_date_range(entry['timestamp'], start_dt, end_dt)
        ]

    if not filtered:
        print("No data found within the specified date/time range.")
        sys.exit(0)
    
    return filtered

def _within_date_range(timestamp, start_dt, end_dt):
    """Checks if a timestamp is within the date range."""
    try:
        entry_time = parse_datetime(timestamp)
    except ValueError:
        return False
    if start_dt and entry_time < start_dt:
        return False
    if end_dt and entry_time > end_dt:
        return False
    return True