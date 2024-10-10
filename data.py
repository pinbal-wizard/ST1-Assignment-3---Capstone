'''
Read data and filter and stuff
'''

'''
        Rubric parts
Reading the dataset
Identifying and rejecting unwanted columns
Removal of outliers and missing values
'''

import pandas as pd

class Data():

    def cleanData(self) -> None:
        '''
        Cleans the CSV data by removing null values
        '''
        # Drop data thats 'N/A', 'Other' or '-'
        for name, col in self.df.items():
            mask = self.df[name] == "Other" 
            self.df = self.df[~mask]
            mask = self.df[name] == "-"
            self.df = self.df[~mask]
            self.df.dropna(inplace=True)

    def getData(self):
        return self.df


    def __init__(self) -> None:
        self.df = pd.read_csv("Australian Vehicle Prices.csv")
        print(f"{len(self.df)} : Rows before cleaning")
        self.cleanData()
        print(f"{len(self.df)} : Rows after cleaning")
