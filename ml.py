'''
File for anything to do with machine learning
'''

'''
        Rubric Parts
Data conversion to numeric values for machine learning/predictive analysis. (Somewhat done?)
Training/testing sampling and K-fold cross validation (Done)
Investigating multiple regression algorithms (Investigating)    | These 2 things can be done using the output of each model
Selection of the best model (Finding)                           | and changing how Nan values are handled when using the GUI
Deployment of the best model in production
Algorithms are all implemented
'''

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
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
        algorithms = {
            'linear': self.linearRegression,
            'decision tree': self.decisionTree,
            'random forest': self.randomForest,
            'adaboost': self.adaBoost,
            'xgboost': self.XGBoost,
            'k-nearest neighbour': self.KNNRegressor,
            'svr': self.SVRRegression
        }
        
        algorithmMethod = algorithms.get(algorithm)
        
        # Checking for NaN
        self.df = self.df.dropna(subset=features + ['Price'])
        # Debug
        print(f"\nRemaining NaN values in features or target: \n{self.df.isna().sum()}")
        
        # Check if algorithm exist
        if algorithmMethod:
            algorithmMethod(features, split)
        else:
            print(f"Error: {algorithm} has not been implemented")
            
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
        print(f"\nSelected predictors: {list(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Linear Regression Results")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)
            
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
        print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1 - split, random_state = 42)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Decision Tree results")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)

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
        print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 1 - split, random_state = 42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Random Forest results")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)
        
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
        print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 1 - split, random_state = 42)
        
        # Initialize Adaboost with Decision Tree as base leanrner
        # n_estimators: Maximum number of estimator at which boosting is terminated
        model = AdaBoostRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        print(f"Adaboost Results")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)
        
    def XGBoost(self, features: list, split: float):
        '''
        Train and evaluate using XGBoost
        '''
        X = self.df[features]
        y = self.df['Price']

        # Debug
        print(f"\nSelected predictors: {list(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - split, random_state = 42)

        model = XGBRegressor(n_estimators = 100, random_state = 42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(f"XGBoost results")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)

    def KNNRegressor(self, features: list, split: float, k = 5):
        '''
        Train and evaluate using K-nearest neighbours with k = 5 (can be changed)
        
        Params:
        features (list): The predictors used by the user
        split (float): The train/test split
        k (int): K-Neighbours 
        '''
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
        
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("K-Nearest Neighbours Result")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)
        
    def SVRRegression(self, features: list, split: float, kernel = 'linear', C = 1.0, epsilon = 0.1):
        '''
        Training and evaluating using SVR because this model is more suitable for prediction
        It just explodes due to how large the data set is
        
        Params:
        features (list): Selected predictors by the user
        split (float): Test/train split specified by the user
        kernel (str): Functions to deterine the similarity between input vectors. Default is linear, a dot product between input vectors. 
        (kernel cont.) Determines how data input is transformed
        
        C (float): Regularization parameter. Controls the error tolerance of the data. High means less tolerance, low means more tolerance
        epsilon (float): Defines a margin of tolerance where predictions are considered "close enough" to the actual values, and no penalty is applied
        
        Documentations: https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVR.html
        '''
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1-split, random_state=42)
        
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        print("SVR Result")
        self.evaluateModel(y_test, y_pred)
        self.performCrossValidation(model, X_train, y_train)
        
    def evaluateModel(self, y_test, y_pred):
        '''
        Evaluate the model using these metrics: R-squared, mean absolute error and mean absolute percentage error
        Mean absolute percentage error (MAPE): Percentage of how "off" the prediction is. 
        Ex: 10% MAPE means the prediction is off by 10%
        '''
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 # MAPE in percentage
        
        # List to store results
        predictionResults = []
        
        predictionResults.append(f"R-squared: {r2}")
        predictionResults.append(f"Mean Absolute Error: {mae}")
        predictionResults.append(f"Mean Absolute Percentage Error: {mape: .2f}%")
        
        predictionResults.append("\nActual vs Predicted values:")
        for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
            if i >= 10:
                predictionResults.append(f"... And {len(y_test)} more predictions left")
                break
            predictionResults.append(f"Actual: {actual}$, Predicted: {predicted: .2f}$")
            
        # Debugging    
        for lines in predictionResults:
            print(lines)

        return predictionResults
    
    def performCrossValidation(self, model, X_train, y_train, k_folds = 5):
        '''
        Perform a K-fold cross validation on a model
        
        Params:
        model: The model to be evaluated
        X_train (dataframe): The training set
        y_train (series): The target set (in this case, 'Price')
        k_folds (int): Number of folds to do cross validation, default is 5
        '''
        
        print(f"Performing {k_folds}-folds cross validation")
        cvScores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='r2')
        cvScoresRounded = np.round(cvScores, 2)
        avgScore = round(np.mean(cvScores), 2)
        
        # list to store the output
        cvResult = []
        
        # Append the rounded cross validation score
        for i, score in enumerate(cvScoresRounded):
            cvResult.append(f"Fold {i+1} R-squared: {score}")
        
        # Append the average score    
        cvResult.append(f"Average R-squared across {k_folds}: {avgScore}")
        
        # Debug
        for lines in cvResult:
            print(lines)
            
        return cvResult
        
        
if __name__ == "__main__":
    print("Run this from the GUI.")