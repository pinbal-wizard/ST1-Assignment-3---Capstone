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

class ML:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        # Testing correlation values
        print(self.df[['FuelConsumption', 'Kilometres', 'Year', 'CylindersinEngine', 'Price']].corr())
    
    def run(self, algorithm: str, features: list, split: float):
        '''
        This method acts as a controller to take the settings from the GUI and give them to the respective model

        Parameters:
        algorithm (str): The selected algorithm
        features (list): List of feature columns selected by the user
        split (float): The train/test split used selected by the user
        max_lines (int): Maximum lines to print out
        '''
        
        # Dispatching
        if algorithm == 'Linear':
            self.linearRegression(features, split)
        elif algorithm == 'Decision Tree':
            print("Decision Tree algorithm not implemented yet.")
        else:
            print(f"Error: {algorithm} is not a supported algorithm.")

    def linearRegression(self, features: list, split: float, max_lines = 50):
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
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Linear Regression Results")
        print(f"R-squared: {r2}")
        print(f"Mean absolute error: {mae}")
        
        print("\nActual vs Predicted values:")
        for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
            if i >= max_lines:
                print(f"... And {len(y_test) - max_lines} more lines")
                break
            print(f"Actual ($): {actual}, Predicted ($): {predicted: .2f}")


if __name__ == "__main__":
    print("Run this from the GUI.")