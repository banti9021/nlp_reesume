import os
import pandas as pd

def load_csv_folder(folder_path):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    df_list = []
    for file in csv_files:
        full_path = os.path.join(folder_path, file)
        df = pd.read_csv(full_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


if __name__ == "__main__":
    folder_path = r"E:\Ressumme_nlp\archive\Resume"  # ✅ folder path
    df = load_csv_folder(folder_path)
    print(df.shape)
    print(df.head())
