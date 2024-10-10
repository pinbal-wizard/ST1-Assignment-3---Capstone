'''
Analysise the file dependent on data to clean and import data
'''

'''
        Rubric parts
Data exploration at basic level
Visual and statistic correlation analysis for selection of best features (Analyse)
'''


import data


class Analysis():
    def __init__(self) -> None:
        pass
    
    
    
if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv("Australian Vehicle Prices.csv")

    df.info()

    #print(df.isna().sum())
    #print(df.tail(5))

    print(len(df))
    for name, col in df.items():
        mask = df[name] == "Other" 
        df = df[~mask]
        mask = df[name] == "-"
        df = df[~mask]
        df.dropna(inplace=True)


    for name, coloum in df.items():
        if name in ["Year", "Model", "Title", "Price", "Kilometres", "Location", "ColourExtInt", "Car/Suv"]:
            print(len(set(coloum)), ": " + name + "\n")
            continue
        print(set(coloum))
        print(len(set(coloum)), ": " + name + "\n")
        
    print(len(df))