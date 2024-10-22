import joblib
from xgboost import XGBRegressor
from data import Data  

class TrainFullXGBoostWithCleaning:
    def __init__(self, data):
        self.df = data.getData() 

    def train_full_XGBoost(self, output_file: str):
        '''
        Train XGBoost using 100% of the cleaned dataset and serialize the model.
        
        Parameters:
        output_file (str): The path to save the serialized model
        '''

        features = [col for col in self.df.columns if col != 'Price']
        
        X = self.df[features]
        y = self.df['Price'] 

        model = XGBRegressor(n_estimators=100, random_state=42, enable_categorical=True)
        model.fit(X, y)

        joblib.dump(model, output_file)
        print(f"Model saved to {output_file}")

if __name__ == "__main__":
    # Load and clean the data
    data = Data(mode="Delete Rows")  
    
    trainer = TrainFullXGBoostWithCleaning(data)
    trainer.train_full_XGBoost('xgboost_model.pkl')
