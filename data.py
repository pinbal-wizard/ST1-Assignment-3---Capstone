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
        def removeNull(self) -> None:
            '''
            Remove values that are 'Other', '-' or NULL
            '''
            for name, col in self.df.items():
                mask = self.df[name] == "Other" 
                self.df = self.df[~mask]
                mask = self.df[name] == "-"
                self.df = self.df[~mask]
                self.df.dropna(inplace=True)

        def removeFluff(self) -> None:
            '''
            Remove unessecary fluff from columns like FuelConsumption
            '''
            # Engine category will be cleaned once we have a discussion on how these values are going to be used
            #self.df['Engine'] = self.df['Engine'].str.replace(' cyl', '')
            #self.df['Engine'] = self.df['Engine'].str.replace(' L', '')
            

            self.df['FuelConsumption'] = self.df['FuelConsumption'].str.replace(' L / 100 km', '')
            self.df['FuelConsumption'] = self.df['FuelConsumption'].astype(float)

            self.df['CylindersinEngine'] = self.df['CylindersinEngine'].str.replace('cyl', '')
            # Electric vechicals have no cylinders
            self.df['CylindersinEngine'] = self.df['CylindersinEngine'].str.replace(' L', '')
            self.df['CylindersinEngine'] = self.df['CylindersinEngine'].astype(int)

            self.df['Doors'] = self.df['Doors'].str.replace(' Doors', '')
            self.df['Doors'] = self.df['Doors'].astype(int)

            self.df['Seats'] = self.df['Seats'].str.replace(' Seats', '')
            self.df['Seats'] = self.df['Seats'].astype(int)

        removeNull(self)
        removeFluff(self)
            

        

    def getData(self):
        return self.df


    def __init__(self) -> None:
        self.df = pd.read_csv("Australian Vehicle Prices.csv")
        print(f"{len(self.df)} : Rows before cleaning")
        self.cleanData()
        print(f"{len(self.df)} : Rows after cleaning")
