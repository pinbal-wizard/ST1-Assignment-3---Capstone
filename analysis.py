'''
Analysise the file dependent on data to clean and import data
'''

'''
        Rubric parts
Data exploration at basic level
Visual and statistic correlation analysis for selection of best features (Analyse)
'''

import data
import pandas as pd



class Analysis():
    def displayData(self) -> None:
        '''
        Displays data in CSV
        '''
        #self.df.info()
        print(self.df.isna().sum())
        print(self.df.tail(5))
        
        for name, coloum in self.df.items():
            if name in ["Year", "Model", "Title", "Price", "Kilometres", "Location", "ColourExtInt", "Car/Suv"]:
                print(len(set(coloum)), ": " + name)
                continue
            print(set(coloum))
            print(len(set(coloum)), ": " + name)
        
    def analyseValue(self, column : str, value : any) -> int:
        '''
        Get the amount of rows that have the same value in the same column

        Parameters:
            column (str): The name of the column to analyze.
            value (any): The value to count in the specified column.

        Returns:
            int: The count of rows with the specified value in the column.
        '''
        if column in self.df.columns:
            count = (self.df[column] == value).sum()
            return count
        else:
            print(f"Column '{column}' does not exist in data.")
            return 0
        


    def __init__(self, data) -> None:
        self.df = data.getData()
        self.displayData()
        print("------ Random statitics ------")
        print(f"Electric vehicles: {self.analyseValue("FuelConsumption", "0")}")
        print("-------- Column Info ---------")

        print(self.df.dtypes)
        pass
    
    
    
if __name__ == '__main__':
    dataAnalysis = Analysis()