import pandas as pd

csv = "/media/rory/Padlock_DT/BLA_Analysis/Decoding/Pearson_Correlation_Datasets/BLA-Insc-1/RDT D1/Shock Ocurred_Choice Time (s)/True/trial_1_corrmap.csv"

df = pd.read_csv(csv)
df = df.iloc[:,1:]
print(df.head())
print(list(df.columns))
print(list(df.index))
print(len(df))