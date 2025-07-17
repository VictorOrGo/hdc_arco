import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge two jsons where timestamp and src match.")
    parser.add_argument('df_data', help="Path to the first JSON dataset.")
    parser.add_argument('df_diag', help="Path to the second JSON dataset.")
    parser.add_argument('output', help="Path to the desired output file.")

    args = parser.parse_args()
    # Leer los archivos JSON
    df_data = pd.read_json(args.df_data, lines=True)
    df_diag = pd.read_json(args.df_diag, lines=True)

    # Renombrar la columna 'source_address' a 'src' en df_diag
    df_data.rename(columns={'source_address': 'src'}, inplace=True)
    df_diag.rename(columns={'source_address': 'src'}, inplace=True)
    
    if len(df_diag) > len(df_data):
        df_merged = pd.merge_asof(df_diag, df_data, on='timestamp', by='src', direction='forward')
    else:
        df_merged = pd.merge_asof(df_data, df_diag, on='timestamp', by='src', direction='forward')

    df_merged.to_json(args.output, orient='records', lines=True, date_format='iso')
    print("Archivo JSON combinado creado\n")

if __name__ == "__main__":
    main()