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
            #print(set(coloum))
            print(len(set(coloum)), ": " + name) 


    def __init__(self, data) -> None:
        self.df = data.getData()
        self.displayData()
        pass
    
    
    
if __name__ == '__main__':
    dataAnalysis = Analysis()