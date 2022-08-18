import pandas as pd

csv = "/media/rory/RDT VIDS/ABET_files_opto/RRD81 11042019.csv"

df = pd.read_csv(csv, sep=",", error_bad_lines=False).fillna(0)
print("here")
print(df.head())