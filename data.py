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
import numpy as np
from scipy import stats
import warnings

# Suppress pandas stupid "FutureWarning" stuff, it's my console not yours pandas.
# Yes, I know it's 'bad pratice' but we gave pip versions in requirements.txt so use that if there's an issue.
warnings.simplefilter(action='ignore', category=FutureWarning)

class Data():
    def cleanData(self, mode : str = "Delete Rows",) -> None:
        '''
        Cleans the CSV data by removing null values

        Parameters:
            mode (str): ["Delete Rows", "Median Value", "Mode Value", "Interpolate"]

        returns:
            None
        '''

        def removeNull(self, mode) -> None:
            '''
            Remove values that are 'Other', '-' or NULL
            '''

            # If the value is in nanValues then convert it to NaN
            nanValues = ["Other", "-", "NaN", "- / -"]
            for name, col in self.df.items():
                self.df[name] = self.df[name].replace(nanValues, np.nan)

            # Remove strings from cols before conversion or there will be issues
            removeFluff(self)
            # Columns such as 'Price' & 'Kilometers' can be converted to numeric to estimate values
            colsToConvert = ['Kilometres', 'Price', 'Doors', 'Seats', 'CylindersinEngine', 'FuelConsumption']
            for col in colsToConvert:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            if mode == "Delete Rows":
                    self.df.dropna(inplace=True)

            elif mode == "Median Value":  
                    # Get colums that can be numeric for median NaN estimation
                    NonNumericCols = self.df.select_dtypes(exclude='number').columns.tolist()
                    numericCols = self.df.select_dtypes(include='number').columns.tolist()

                    # Replace NaN values with median; drop columns if they cannot be filled
                    for col in numericCols:
                        median = self.df[col].median()
                        if pd.notna(median):
                            self.df[col].fillna(median, inplace=True)
                        else:
                            self.df.drop(columns=[col], inplace=True)

                    # Drop NaN values in columns where they cannot be Numeric (e.g. Brand)
                    self.df.dropna(subset=NonNumericCols, inplace=True)
                    
            elif mode == "Mode Value":
                 # Get colums that can be numeric for mode NaN estimation
                    NonNumericCols = self.df.select_dtypes(exclude='number').columns.tolist()
                    numericCols = self.df.select_dtypes(include='number').columns.tolist()

                    # Replace NaN values with mode; drop columns if they cannot be filled
                    for col in numericCols:
                        mode = self.df[col].mode()
                        if not mode.empty:
                            self.df[col].fillna(mode, inplace=True)
                        else:
                            self.df.drop(columns=[col], inplace=True)


                    # Drop NaN values in columns where they cannot be Numeric (e.g. Brand)
                    self.df.dropna(subset=NonNumericCols, inplace=True)
                    # Drop any NaN values that didn't get passed the mode cleaning process (We did all we can)
                    self.df.dropna(inplace=True)

            elif mode == "Interpolate":
                    # Get colums that can be numeric for interpolate NaN estimation
                    NonNumericCols = self.df.select_dtypes(exclude='number').columns.tolist()
                    numericCols = self.df.select_dtypes(include='number').columns.tolist()

                    # Replace NaN values with mode; drop columns if they cannot be filled
                    for col in numericCols:
                        self.df[col].interpolate(method='linear', inplace=True)

                    # Drop NaN values in columns where they cannot be Numeric (e.g. Brand)
                    self.df.dropna(subset=NonNumericCols, inplace=True)


        def removeFluff(self) -> None:
            '''
            Remove unessecary fluff from columns like FuelConsumption
            '''
            # Engine category will be cleaned once we have a discussion on how these values are going to be used
            #self.df['Engine'] = self.df['Engine'].str.replace(' cyl', '')
            #self.df['Engine'] = self.df['Engine'].str.replace(' L', '')
            

            self.df['FuelConsumption'] = self.df['FuelConsumption'].str.replace(' L / 100 km', '', regex=False)
            self.df['FuelConsumption'] = pd.to_numeric(self.df['FuelConsumption'], errors='coerce')

            self.df['CylindersinEngine'] = self.df['CylindersinEngine'].str.replace(' cyl', '', regex=False)
            self.df['CylindersinEngine'] = self.df['CylindersinEngine'].str.replace(' L', '', regex=False)
            self.df['CylindersinEngine'] = pd.to_numeric(self.df['CylindersinEngine'], errors='coerce')

            self.df['Doors'] = self.df['Doors'].str.replace(' Doors', '', regex=False)
            self.df['Doors'] = pd.to_numeric(self.df['Doors'], errors='coerce')

            self.df['Seats'] = self.df['Seats'].str.replace(' Seats', '', regex=False)
            self.df['Seats'] = pd.to_numeric(self.df['Seats'], errors='coerce')

        removeNull(self, mode)
            

    def convertColumnTypes(self) -> None:
        '''
        Converts some of the colums to the correct datatype to do ML

        Returns:
            Nothing
        '''
        self.df['Kilometres'] = self.df['Kilometres'].astype({'Kilometres': 'float'})
        self.df['CylindersinEngine'] = self.df['CylindersinEngine'].astype({'CylindersinEngine': 'int32'})
        self.df['Year'] = self.df['Year'].astype({'Year': 'int32'})
        
        self.df['Price'] = pd.to_numeric(self.df['Price'], errors='coerce')

        
        self.df.dropna(subset=['Price'],inplace=True)

        # Convert to category
        self.df['Brand'] = self.df['Brand'].astype("category")
        self.df['Model'] = self.df['Model'].astype("category")
        self.df['Car/Suv'] = self.df['Car/Suv'].astype("category")
        self.df['Title'] = self.df['Title'].astype("category")
        self.df['UsedOrNew'] = self.df['UsedOrNew'].astype("category")
        self.df['Transmission'] = self.df['Transmission'].astype("category")
        self.df['Engine'] = self.df['Engine'].astype("category")
        self.df['DriveType'] = self.df['DriveType'].astype("category")
        self.df['FuelType'] = self.df['FuelType'].astype("category")
        self.df['ColourExtInt'] = self.df['ColourExtInt'].astype("category")
        self.df['Location'] = self.df['Location'].astype("category")
        self.df['BodyType'] = self.df['BodyType'].astype("category")

        

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

    def rankColumn(self, column : str = None) -> None:
        '''
        Ranks column provided via their avg price
        
        Parameters:
            column (str): Selected column to rank, default is None
        '''
        if column is None:
            return
        
        avgPrice = self.df.groupby(column)['Price'].mean()
        orderCol = avgPrice.sort_values().index.tolist()
        self.df[column] = pd.Categorical(self.df[column], categories=orderCol, ordered=True)
        self.df.sort_values(column, inplace=True)


    def iterateOverColumns(self) -> None:
        raise NotImplementedError
        for col in ["Brand","Model","UsedOrNew","Transmission"]:
            self.rankColumn(col)
            self.df[col] = self.df[col].cat.codes

    def mapRankToValue(self, column : str = None) -> dict:
        '''
        Gets the name back from the ranked value
        
        Parameters:
            column (str): Selected column to get, default is None

        Returns:
            mapping (dict): Dictionary of {Rank, label}
        '''
        mapping = {code: label for code, label in enumerate(self.df[column].cat.categories)}
        return mapping


    def __init__(self, mode = "Delete Rows") -> None:
        self.df = pd.read_csv("Australian Vehicle Prices.csv")
        #print("Null values before cleaning", self.df['Kilometres'].isna().sum())
        print(f"{len(self.df)} : Rows before cleaning")
        self.cleanData(mode)
        print(f"{len(self.df)} : Rows after fluff removed")
        self.convertColumnTypes()
        print(f"{len(self.df)} : Rows after Filtering")
        self.removeOutliers(method='IQR', columns=['Price'])  # We can decide on other outliers later, this is just a test
        print(f"{len(self.df)} : Rows after Outlier Removal")
        #self.iterateOverColumns() removed to keep cols as categories
        print(f"{len(self.df)} : Rows after cleaning")





        
if __name__ == "__main__":
    data = Data()
    # Function test
    # Seems to be working as intended, extreme outliers have been removed
    #print(f"After removing outliers: {len(data.df)}")
    #print(data.df.describe())
    #print(data.df.dtypes)
    #print(data.df.isna().sum())
    #print("Null values after cleaning", data.df['Kilometres'].isna().sum())

