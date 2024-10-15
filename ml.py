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
Algorithms to be implemented: Linear Regression, Decision Tree, Random Forest, Adaboost, XGBoost, K-Nearest Neighbour, SVM
'''


import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from data import Data

class ML():
    def linearRegression(self, max_lines = 50):
        '''
        Train and evaluate linear regression based on cleaned dataset
        
        Parameters:
        max_lines (int): Display the maximum number of lines in the predicted output. Current default is 50
        
        Returns:
        None: Prints the evaluation result directly and visualize with plot
        '''
        
        # Define features and target variable, can be adjusted if needed
        X = self.df[['Kilometres', 'FuelConsumption', 'CylindersinEngine', 'Year']]
        y = self.df['Price'] # This value can be changed
        self.df = self.df[self.df['Price'] >= 0]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"R-squared: {r2}")
        print(f"Mean absolute error: {mae}")
        
        print('\nActual and Predicted Values: ')
        for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
            if i >= max_lines:
                print(f"... and {len(y_test) - max_lines} more lines.")
                break
            print(f"Actual ($): {actual}, Predicted ($): {predicted: .2f}")
    
    def __init__(self) -> None:
        data = Data()
        self.df = data.getData()
        # Testing correlation values
        print(self.df[['FuelConsumption', 'Kilometres', 'Year', 'CylindersinEngine', 'Price']].corr())
        
if __name__ == "__main__":
    mlInstance = ML()
    mlInstance.linearRegression()