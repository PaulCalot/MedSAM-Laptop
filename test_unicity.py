import pathlib
import pandas as pd

df = pd.read_csv("MAPPER.csv")

print(df["id"].nunique() / len(df))
print(df["primary_key"].nunique() / len(df))
