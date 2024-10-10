import pandas as pd

df = pd.read_csv("Australian Vehicle Prices.csv")

df.info()

#print(df.isna().sum())
#print(df.tail(5))

print(len(df))
for name, col in df.items():
    mask = df[name] == "Other" 
    df = df[~mask]
    mask = df[name] == "-" ``
    df = df[~mask]
    df.dropna(inplace=True)


for name, coloum in df.items():
    if name in ["Year", "Model", "Title", "Price", "Kilometres", "Location", "ColourExtInt", "Car/Suv"]:
        print(len(set(coloum)), ": " + name + "\n")
        continue
    print(set(coloum))
    print(len(set(coloum)), ": " + name + "\n")
    
print(len(df))