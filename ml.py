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
    
    def runAllAlgorithm(self, features: list, split: float):
        '''
        Run all algorithm and select the best performing one based on R-squared score

        Parameters:
        features (list): List of feature columns selected by the user
        split (float): The train/test split used selected by the user
        '''
        
        print(f"\nSelected predictors are: {features}")

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
        
        # List to store evaluation results
        results = []
        resultsButAsADict = {}
        
        for algoName, algoMethod in algorithms.items():
            #print(f"\nRunning {algoName.capitalize()}... ")
            metrics = algoMethod(features, split) # Capturing all metrics
            results.append((algoName, metrics))
            resultsButAsADict[algoName] = metrics
            
        bestAlgo = self.compareAlgorithms(results)
        print(f"\nThe best performing algorithm: {bestAlgo.capitalize()}")

        return resultsButAsADict, bestAlgo
        
    def compareAlgorithms(self, results: list):
        '''
        Compare and output the best algorithm based R-squared, MAE, MAPE, cross validation results
        Print the best performing algorithm based on lowest MAPE and highest R-squared

        Params:
        results (list): Tuple of algorithm name and metrics
        
        Returns:
        str: Name of the best algo
        '''
        
        rankedResults = []
        
        for algoName, metrics in results:
            r2, mae, mape, avg_cv_r2 = metrics # avg_cv_r2: average R-squared from cross validation
            #print(f"\nEvaluation metrics for {algoName.capitalize()}:")
            #print(f"R-squared: {r2}, MAE: {mae}, MAPE: {mape: .2f}%, Average CV R-squared: {avg_cv_r2}")
            
            # Compare metrics, prio: MAPE and R-squared
            # Float values can be changed depending on our priority
            # Current priority: 50% for R-squared, 25% for MAE and MAPE
            score = (0.5 * r2) + (0.25 * (1 / mae)) + (0.25 * (1 / mape))
            
            rankedResults.append((algoName, score))
        
        # Rank by highest score    
        rankedResults.sort(key=lambda x: x[1], reverse=True)
        
        return rankedResults[0][0]
    
    def evaluateModel(self, y_test, y_pred, model, X_train, y_train):
        '''
        Evaluate the model using these metrics: R-squared, mean absolute error and mean absolute percentage error
        Mean absolute percentage error (MAPE): Percentage of how "off" the prediction is. 
        Ex: 10% MAPE means the prediction is off by 10%
        '''
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 # MAPE in percentage
        
        
        #print(f"R-squared: {r2}")
        #print(f"Mean Absolute Error: {mae}")
        #print(f"Mean Absolute Percentage Error: {mape: .2f}%")
        
        #print("\nActual vs Predicted values:")
        for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
            if i >= 10:
                #print(f"... And {len(y_test)} more predictions left")
                break
            #print(f"Actual: {actual}$, Predicted: {predicted: .2f}$")
        
        # Perform cross validation to get average cross validation r-squared    
        avg_cv_r2 = self.performCrossValidation(model, X_train, y_train)
        
        return r2, mae, mape, avg_cv_r2
    
    def performCrossValidation(self, model, X_train, y_train, k_folds = 5):
        '''
        Perform a K-fold cross validation on a model
        
        Params:
        model: The model to be evaluated
        X_train (dataframe): The training set
        y_train (series): The target set (in this case, 'Price')
        k_folds (int): Number of folds to do cross validation, default is 5
        '''
        
        #print(f"Performing {k_folds}-folds cross validation")
        cvScores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='r2')
        cvScoresRounded = np.round(cvScores, 2)
        avg_cv_r2 = round(np.mean(cvScores), 2)
        
        # Print R-squared across each fold
        for i, score in enumerate(cvScoresRounded):
            pass
            #print(f"Fold {i+1} R-squared: {score}")
        
        # Print average score   
        #print(f"Average R-squared across {k_folds} folds: {avg_cv_r2}")
        
        return avg_cv_r2

    def linearRegression(self, features: list, split: float):
        '''
        Train and evaluate linear regression based on user-selected settings
        
        Parameters:
        features (list): List of feature columns selected by the user
        split (float): The train/test split used selected by the user
        
        Returns:
        None: Prints the evaluation result and the actual vs. predicted results
        '''
        # Fix for category datatype not working with this algorithm
        for feature in features:
            if self.df[feature].dtype == 'category':
                self.df[feature] = self.df[feature].cat.codes

        X = self.df[features]  # Selected features
        y = self.df['Price']  # 'Price' as the target variable
        
        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        #print(f"Linear Regression Results")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)
            
    def decisionTree(self, features: list, split: float):
        '''
        Train and evaluate using Decision Tree based on user settings
        
        Parameters:
        features (list): The select predictors from the user 
        split (float): The train/test split from the user
        
        Returns:
        None: Print out the predicted vs. the actual results directly 
        '''
        # Fix for category datatype not working with this algorithm
        for feature in features:
            if self.df[feature].dtype == 'category':
                self.df[feature] = self.df[feature].cat.codes
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1 - split, random_state = 42)

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        #print(f"Decision Tree results")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)

    def randomForest(self, features: list, split: float):
        '''
        Train and evaluate using Random Forest regressor based on user settings
        
        Parameters:
        features (list): The select predictors from the user 
        split (float): The train/test split from the user
        
        Returns:
        None: Print out the predicted vs. the actual results directly 
        '''
        # Fix for category datatype not working with this algorithm
        for feature in features:
            if self.df[feature].dtype == 'category':
                self.df[feature] = self.df[feature].cat.codes
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 1 - split, random_state = 42)
        
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        #print(f"Random Forest results")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)
        
    def adaBoost(self, features: list, split: float):
        '''
        Train and evaluate using Adaboost model
        
        Params:
        features (list): Selected features by the user
        split (float): Specified train/test split
        
        Returns:
        None: Print the prediction directly
        '''
        # Fix for category datatype not working with this algorithm
        for feature in features:
            if self.df[feature].dtype == 'category':
                self.df[feature] = self.df[feature].cat.codes
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 1 - split, random_state = 42)
        
        # Initialize Adaboost with Decision Tree as base leanrner
        # n_estimators: Maximum number of estimator at which boosting is terminated
        model = AdaBoostRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        #print(f"Adaboost Results")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)
        
    def XGBoost(self, features: list, split: float):
        '''
        Train and evaluate using XGBoost
        '''
        X = self.df[features]
        y = self.df['Price']

        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - split, random_state = 42)

        model = XGBRegressor(n_estimators = 100, random_state = 42,enable_categorical=True)  ############################## apparently enable_categorical=True is experimental so GL
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        #print(f"XGBoost results")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)

    def KNNRegressor(self, features: list, split: float, k = 5):
        '''
        Train and evaluate using K-nearest neighbours with k = 5 (can be changed)
        
        Params:
        features (list): The predictors used by the user
        split (float): The train/test split
        k (int): K-Neighbours 
        '''
        # Fix for category datatype not working with this algorithm
        for feature in features:
            if self.df[feature].dtype == 'category':
                self.df[feature] = self.df[feature].cat.codes
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split, random_state=42)
        
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        #print("K-Nearest Neighbours Result")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)
    
    # Commented out because this algorithm just explodes    
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
        # Fix for category datatype not working with this algorithm
        for feature in features:
            if self.df[feature].dtype == 'category':
                self.df[feature] = self.df[feature].cat.codes
        
        X = self.df[features]
        y = self.df['Price']
        
        # Debug
        #print(f"\nSelected predictors: {list(X.columns)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1-split, random_state=42)
        
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        #print("SVR Result")
        return self.evaluateModel(y_test, y_pred, model, X_train, y_train)

        
if __name__ == "__main__":
    print("Run this from the GUI.")