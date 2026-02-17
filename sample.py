import pandas as pd

file_path="raw_data/STOC_pressions_bio_assol_meteo_pesti_140525.csv"

def get_first_rows_save_csv(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    df_head = df.head(10)
    output_path = "sample_data/sample_stoc_data.csv"
    df_head.to_csv(output_path, index=False)
    print(f"Saved first 10 rows to {output_path}")

if __name__ == "__main__":
    get_first_rows_save_csv(file_path)