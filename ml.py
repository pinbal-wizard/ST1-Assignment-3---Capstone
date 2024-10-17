'''
File for anything to do with machine learning
'''

'''
        Rubric Parts
Data conversion to numeric values for machine learning/predictive analysis.
Training/testing sampling and K-fold cross validation
Investigating multiple regression algorithms (Investigating)
Selection of the best model (Finding)
Deployment of the best model in production
Algorithms to be implemented: XGBoost, K-Nearest Neighbour, SVM
'''

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from data import Data

class ML:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        # Can uncomment if you want to see how data is before and after conversion
        #print("Data before type conversion: ")
        #print(self.df.head(20))
        #
        #self.convertStringColumns()
        #
        #print("Data after type conversion: ")
        #print(self.df.head(20))
    
    def run(self, algorithm: str, features: list, split: float):
        '''
        This method acts as a controller to take the settings from the GUI and give them to the respective model

        Parameters:
        algorithm (str): The selected algorithm
        features (list): List of feature columns selected by the user
        split (float): The train/test split used selected by the user
        max_lines (int): Maximum lines to print out
        '''
        
        algorithm = algorithm.lower()
        
        # Dispatching
        # Ugly af
        if algorithm == 'linear':
            self.linearRegression(features, split)
        elif algorithm == 'decision tree':
            self.decisionTree(features, split)
        elif algorithm == 'random forest':
            self.randomForest(features, split)
        elif algorithm == 'adaboost':
            self.adaBoost(features, split)
        else:
            print(f"Error: {algorithm} is not a supported algorithm.")
            
    #def convertStringColumns(self):
    #    '''
    #    Attempts to convert non-numerical values into numerical using Label Encoding
    #    Skips columns that are already in numerical
    #    '''
    #    # This works but struggles with nomial data, i.e. brands, locations, exterior/interior etc.
    #    # Will introduct bias in data, because the model will compare the numerical values as lower/higher
    #    nonNumericColumns = self.df.select_dtypes(exclude=['int', 'float']).columns
    #    
    #    for col in nonNumericColumns:
    #        le = LabelEncoder()
    #        self.df[col] = le.fit_transform(self.df[col])

    def linearRegression(self, features: list, split: float):
        '''
        Train and evaluate linear regression based on user-selected settings
        
        Parameters:
        features (list): List of feature columns selected by the user
        split (float): The train/test split used selected by the user
        
        Returns:
        None: Prints the evaluation result and the actual vs. predicted results
        '''
        X = self.df[features]  # Selected features
        y = self.df['Price']  # 'Price' as the target variable
        
        # Debug
        print(f"Selected predictors: {list(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Linear Regression Results")
        self.evaluateModel(y_test, y_pred)
            
    def decisionTree(self, features: list, split: float):
        '''
        Train and evaluate using Decision Tree based on user settings
        
        Parameters:
        features (list): The select predictors from the user 
        split (float): The train/test split from the user
        
        Returns:
        None: Print out the predicted vs. the actual results directly 
        '''
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        print(f"Selected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1 - split, random_state = 42)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Decision Tree results")
        self.evaluateModel(y_test, y_pred)

    def randomForest(self, features: list, split: float):
        '''
        Train and evaluate using Random Forest regressor based on user settings
        
        Parameters:
        features (list): The select predictors from the user 
        split (float): The train/test split from the user
        
        Returns:
        None: Print out the predicted vs. the actual results directly 
        '''
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        print(f"Selected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 1 - split, random_state = 42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Random Forest results")
        self.evaluateModel(y_test, y_pred)
        
    def adaBoost(self, features: list, split: float):
        '''
        Train and evaluate using Adaboost model
        
        Params:
        features (list): Selected features by the user
        split (float): Specified train/test split
        
        Returns:
        None: Print the prediction directly
        '''
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        print(f"Selected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 1 - split, random_state = 42)
        
        # Initialize Adaboost with Decision Tree as base leanrner
        # n_estimators: Maximum number of estimator at which boosting is terminated
        model = AdaBoostRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Adaboost Results")
        self.evaluateModel(y_test, y_pred)


    def evaluateModel(self, y_test, y_pred):
        '''
        Evaluate the model using these metrics: R-squared, mean absolute error and mean absolute percentage error
        Mean absolute percentage error (MAPE): Percentage of how "off" the prediction is. 
        Ex: 10% MAPE means the prediction is off by 10%
        '''
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 # MAPE in percentage
        
        print(f"R-squared: {r2}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Absolute Percentage Error: {mape: .2f}%")
        
        print("\nActual vs Predicted values:")
        for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
            if i >= 10:
                break
            print(f"Actual: {actual}$, Predicted: {predicted: .2f}$")

        # Print how many lines are left
        totalPredictions = len(y_test)
        remainingLines = totalPredictions - min(10, totalPredictions)

        if remainingLines > 0:
            print(f"... And {remainingLines} more predictions left.")
        
        
if __name__ == "__main__":
    print("Run this from the GUI.")