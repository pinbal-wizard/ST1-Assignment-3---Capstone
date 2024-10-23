import pandas as pd
import xgboost as xgb
import joblib
from data import Data

def removeOutliers(df):
    """
    Automatically removes outliers from all numeric columns in the DataFrame using the IQR method.
    
    Params:
        df (pd.DataFrame): DataFrame containing the data.
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    numericCols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numericCols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lowerBound = Q1 - 1.5 * IQR
        upperBound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lowerBound) & (df[col] <= upperBound)]
    return df

# Load the data
data = Data()
df = data.getData()

dfCleaned = removeOutliers(df)

# Convert categorical columns
for feature in dfCleaned.select_dtypes(include=['object']).columns:
    dfCleaned[feature] = dfCleaned[feature].astype('category')

# Save the mapping of each categorical column
valueMap = {}
for feature in dfCleaned.select_dtypes(include=['category']).columns:
    valueMap[feature] = dict(enumerate(dfCleaned[feature].cat.categories))
    dfCleaned[feature] = dfCleaned[feature].cat.codes

X = dfCleaned.drop(columns=['Price'])
y = dfCleaned['Price']

xgbModel = xgb.XGBRegressor(n_estimators=100, max_depth=8, random_state=42)
xgbModel.fit(X, y)

joblib.dump(xgbModel, 'XGBOOSTMODEL.pkl')
joblib.dump(valueMap, 'DATAMAP.pkl')

print("Model and data mappings have been successfully saved.")
