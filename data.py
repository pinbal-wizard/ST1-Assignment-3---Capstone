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
from scipy import stats

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
            

    def convertColumnTypes(self) -> None:
        '''
        Converts some of the colums to the correct datatype to do ML

        Returns:
            Nothing
        '''
        self.df['Kilometres'] = self.df['Kilometres'].astype({'Kilometres': 'int32'})
        self.df['CylindersinEngine'] = self.df['CylindersinEngine'].astype({'CylindersinEngine': 'int32'})
        self.df['Year'] = self.df['Year'].astype({'Year': 'int32'})
        
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')

        
        self.df.dropna(subset=['Price'],inplace=True)
        self.df['Price'] = self.df['Price'].astype('int32')
        

    def getData(self):
        return self.df
    

    def getColumnAvg(self, conditionColumn : str = None, condition : str = None, target : str = None ) -> float:
        '''
        Find the mean value of a target column given a condtion

        Parameters:
            conditionColumn (str): The column to check the condition in.
            condition (str): The condition to check for.
            target (str): The value to find the mean of

        Returns:
            mean value (float): The mean value of target variable given condition
        '''
        return self.df.loc[self.df[conditionColumn] == condition, target].mean()
    

    def getBrandAvg(self) -> dict:
        '''
        Gets the average price for each brand and returns a dictionary of the brand and the average price

        Returns:
            Dictionary where the keys are the brand name and the values is the average price of the brand
        '''
        result = {}
        for i, brand in enumerate(self.df['Brand'].unique(), start=1):
            avg_value = round(self.getColumnAvg("Brand", str(brand), "Price"), 2)
            result[brand] = "$" + str(avg_value)
        return result
    

    def removeOutliers(self, method: str = "IQR", columns: list = None, zThreshold: float = 3.0) -> None:
        '''
        To remove the outliers from columns using either IQR or Z-score method
        
        Parameters:
        method (str): Detection method, defaults to IQR but can use Z-score
        columns (list): The list of columns to remove outliers from. Default to None, meaning all numerical columns
        zThreshold (float): Z-score threshold for removal, defaults to 3
        '''
        
        if columns is None:
            columns = self.df.select_dtypes(include=[float, int]).columns
        
        if method == 'IQR':
            for column in columns:
                Q1 = self.df[column].quantile(0.25)
                Q3 = self.df[column].quantile(0.75)
                IQR = Q3 - Q1
                lowerBound = Q1 - 1.5 * IQR
                upperBound = Q3 + 1.5 * IQR
                self.df = self.df[(self.df[column] >= lowerBound) & (self.df[column] <= upperBound)]
                
        elif method == 'Z':
            for column in columns:
                zScores = stats.zscore(self.df[column])
                self.df = self.df[(zScores < zThreshold) & (zScores > -zThreshold)]
        
        else:
            print(f"Invalid removal method: {method}")


    def __init__(self) -> None:
        self.df = pd.read_csv("Australian Vehicle Prices.csv")
        print(f"{len(self.df)} : Rows before cleaning")
        self.cleanData()
        print(f"{len(self.df)} : Rows after cleaning")
        self.convertColumnTypes()
        print(f"{len(self.df)} : Rows after Filtering")
        self.removeOutliers(method='IQR', columns=['Price'])

        
if __name__ == "__main__":
    data = Data()
    # Function test
    # Seems to be working as intended, extreme outliers have been removed
    print("Before outliers removal: ")
    print(data.df.describe())
    data.removeOutliers(method='IQR', columns=['Price']) # We can decide on other outliers later, this is just a test
    print("After removing outliers: ")
    print(data.df.describe())
    print(data.df.dtypes)
