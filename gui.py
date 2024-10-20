'''
Man its the gui what else
Uses nicegui
'''

'''
        Rubric Parts
Problem statement definition
Target variable identification
Visualising the distribution of target variable
Visual Exploratory Data Analysis of data (with histogram and barcharts)
Feature selection based on data distribution
GUI/web deployment using either tkinter/Flask/Streamlit
Visual and statistic correlation analysis for selection of best features (Visualise)
Investigating multiple regression algorithms (Displaying)
Selection of the best model (Displaying)
'''

import data
import analysis
import ml

from nicegui import app, run, ui
from nicegui import events
from nicegui.events import ValueChangeEventArguments

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import numpy as np

class GUI():
    def __init__(self) -> None:
        d = data.Data()
        a = analysis.Analysis(d)
        self.df = d.getData()
        
        app.storage.general['DataExploration'] = "Kilometres"

        with ui.header():
            ui.markdown("# **Software Technology Group 42**")

        # region problem Statement
            
        ui.markdown("### Problem statement")
        ui.markdown(f"This project is based on the Price data of Australian cars, the dataset is available [here](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices). It contains 16734 entries.\
                     We are trying to predict the price of cars. There are many different variables in this problem, the dependent one is the price of the car. Our dataset has \
                     {len(self.df.columns)} other possible colums for independent variables however we will only be using 12 of them these columns. These are: Brand (Categorical),\
                     Year (Quantitative), UsedOrNew (Qualitative), Transmission (Qualitative), Engine (Quantitative), DriveType(Qualitative), FuelType(Qualitative), FuelConsumption(Quantitative)\
                     Kilometers (Quantitative), CylindersInEngine (Quantitative), Doors (Quantitative), Seats (Quantitative). The Columns we won\'t be using are: Model, Car/Suv, Title, ColorExtInt,\
                     Location, and BodyType.")    
            
        # endregion    
        
        # region reading the data
        ui.markdown("#### Reading the data")
        ui.markdown("Reading the data is critical as without it you cant do anything. below is a \
                    snipped from out code, specifically it is the code that reads and cleans our data \
                    before it is used in any way.")
        ui.code("    def __init__(self, mode = \"Delete Rows\") -> None:\n\
        self.df = pd.read_csv(\"Australian Vehicle Prices.csv\")\n\
        print(f\"{len(self.df)} : Rows before cleaning\")\n\
        self.cleanData(mode)\n\
        print(f\"{len(self.df)} : Rows after cleaning\")\n\
        self.convertColumnTypes()\n\
        print(f\"{len(self.df)} : Rows after Filtering\")\n\
        self.removeOutliers(method='IQR', columns=['Price'])\n\
        print(f\"{len(self.df)} : Rows after Outlier Removal\")")
        # endregion

        # region cols
        ui.markdown("#### Columns and Types")
        with ui.scroll_area().classes('w-50 h-100 border'):
            with ui.grid(columns=2).props('dense separator'):
                cols = self.df.dtypes.to_string().split("\n")
                for col in cols:
                    colName, colType = col.split()
                    ui.label(colName + ": ")
                    ui.label(colType).props('inline font-size=100')

        ui.markdown("> ##### Observations\n A few of the colums are int32 or float64 these are our numbers, or quantitative colums. \
                    There are also colums that have a type 'object' these are our catagorical types as they are strings.")
                
        # endregion
        
        # region missing values
        ui.markdown("#### Missing values")
        ui.markdown("What to do with rows with missing or invalid values. Note this is only applicable for rows with numbers, rows with invalid Word entries are removed.")
        def missingValuesRefresh():
            d = data.Data(self.missingValuesRadio.value)
            self.df = d.getData()
            self.refreshAll()

        ## I can see no way that using strings and not enums or something can go wrong
        self.missingValuesRadio = ui.radio(["Delete Rows", "Median Value", "Mode Value", "Interpolate"], value="Delete Rows",on_change=missingValuesRefresh)   ### Make functional

        ui.markdown("##### Observations")
        ui.markdown("> We have decided that we will remove our outlier prior to where it was expected as our large dataset does have quite a few\
                     Outliers however with the above Radio you can change how those outliers are removed.")

        # endregion

        # region dist target var
        ui.markdown("#### Distribution of Target variable")
        targetCol = "Price"
        with ui.pyplot() as fig:  ### Replace With Histogram of the data
            plt.title(targetCol.capitalize())         
            plt.ylabel("Amount")
            plt.xlabel("Price in $AUD")

            plt.hist(self.df[targetCol])     
            
        ui.markdown(f'> ##### Observations\n\
                    > The histogram shows a majority of cars with a price between 20k and 30k with a steep dropoff under 20k and a slower decline on the higher side.\
                     This histogram shows all values within the IQR. This is due to an outlier of 649k which would skew the entire graph to a single bar.')
        
        # endregion
               
        # region data exploration
        # basic exploration of data
        ui.markdown("#### Data exploration")
        catagorical = self.df.describe(exclude=[np.number])
        catagorical.insert(0,"",["Count","Unique","Top","Freq"])
        ui.table.from_pandas(catagorical)
        numbered = self.df.describe(include=[np.number])
        numbered.insert(0,"",["Count","Mean","Std","Min","25%","50%","75%","Max"])
        ui.table.from_pandas(numbered)
        
        self.selectedExploration = ui.select(list(self.df.columns),value=app.storage.general['DataExploration'],on_change=self.dataExploration.refresh)
        self.dataExploration()
        
        ui.markdown(f"> ##### Observations\n> Using the above analysis the following colums have been selected to go to the next step before deciding\
                     if they are to be used for the final predictors.")
        ui.markdown("> - Brand (Categorical)\n\
                    > - Year (Quantitative)\n  \
                    > + UsedOrNew (Qualitative)\n\
                    > + Transmission (Qualitative)\n\
                    > + Engine (Quantitative)\n\
                    > + DriveType(Qualitative)\n\
                    > + FuelType(Qualitative)\n\
                    > + FuelConsumption(Quantitative)\n\
                    > + Kilometers (Quantitative)\n\
                    > + CylindersInEngine (Quantitative)\n\
                    > + Doors (Quantitative)\n\
                    > + Seats (Quantitative)")
        
        # endregion

        
        # region correlation analysis
        # When the target variable is continuous, and the predictor variable is categorical we analyse the relation using box plots. 
        ui.markdown("#### Visual and statistic correlation analysis")
        self.selectedCorrelation = ui.select(list(self.df.loc[:, self.df.columns != 'Price'].select_dtypes(include=np.number))
                                             ,value=app.storage.general['DataExploration'],on_change=self.correlationAnalysis.refresh)
        self.correlationAnalysis()

        correlationDF = self.df.loc[:, self.df.columns].select_dtypes(include=np.number).corr()
        correlationDF.insert(0,' ',correlationDF.columns)
        correlationDF.round(2)
        ui.table.from_pandas(correlationDF)

        ui.markdown("##### Observations")
        ui.markdown("> We are calculating the Pearson's correlation coefficient. This value can be calculated only between two numeric columns\
                     A Correlation value between [-1,0) means inversely proportional, the scatter plot will show a downward trend\
                     A Correlation value between (0,1] means directly proportional, the scatter plot will show a upward trend\
                     Correlation near 0 means No relationship, the scatter plot will show no clear trend.\
                     If the Correlation value is between two variables is > 0.5 in magnitude, it indicates good relationship the sign does not matter")
        ui.markdown("> From these tests we can confirm:   \n\
                     > Kilometers and Year have a strong relation  \n\
                     > The fuel consumption and the number of cylinders in the engine have a strong relation   \n\
                     > And that the price and the year have a strong relation")
        #TODO maybe add sum more stuff here
        
        # endregion
        
        # region ANOVA tests
        ui.markdown("#### Statistical feature selection (categorical vs. continuous) using ANOVA test")
        self.selectedANOVA = ui.select(list(self.df.loc[:, self.df.columns != 'Price'].select_dtypes(exclude=np.number))
                                      ,value="Transmission",on_change=self.ANOVAAnalysis.refresh)
        self.ANOVAAnalysis()

        #TODO add more
        ui.markdown("##### Observations")
        ui.markdown("> Some of the columns are not available as there are too any unique entries to graph them.   Rest assured they are calculated nevertheless")

        # endregion
        
        # region machine learning selectors
        
        # Updating selectors based on user input    
        def updatePredictors(col, isChecked):
            if isChecked:
                self.selectedPredictors.append(col)
            else:
                self.selectedPredictors.remove(col)
        
        ui.markdown("####  Final predictors/features")
        self.selectedPredictors = [] # Store selected predictors/features
        with ui.list():                      ### make updates to this reflect in code
            cols = self.df.dtypes.to_string().split("\n")
            for col in cols:
                colName, _ = col.split()
                checkbox = ui.checkbox(colName)
                checkbox.on_value_change(lambda e, col=colName: updatePredictors(col, e.value)) # Update predictors list with selected values
                ui.space()  
        
        ui.markdown("#### Train/test data split")
        self.trainTestSplitSlider = ui.slider(min=20,max=80, value=80) ### Make functional
        ui.label().bind_text_from(self.trainTestSplitSlider, 'value', backward=lambda v: f"Train {v}%, Test {100-v}%") 
        
        ui.markdown("#### Regression algorithms")
        listofRegressionAlgorithms = ["Linear", "Decision Tree", "Random forest", "Adaboost", "XGBoost", "K-Nearest neighbour", "SVR"]
        self.regressionAlgorithm = ui.radio(listofRegressionAlgorithms, value="Linear")   ### Make functional
        
        # endregion
        
        ui.markdown("#### Launch")
        ui.button("Start!", on_click=self.runML)
        
        ui.image()
        ui.run() 


    @ui.refreshable
    def correlationAnalysis(self) -> None:
        text:str = self.selectedCorrelation.value

        with ui.pyplot() as fig:  ### Replace With Histogram of the data        
            plt.ylabel("Price")
            plt.xlabel(text.capitalize())
            plt.scatter(self.df[text],self.df["Price"],s=0.5)  


    @ui.refreshable
    def ANOVAAnalysis(self) -> None:
        text:str = self.selectedANOVA.value

        if text in ["Model","Title", "Car/Suv", "ColourExtInt","Location"]:
            ui.markdown(f"> Too many unique entries in {text}")
            return

        with ui.pyplot() as ANOVA:
            ANOVA = plt.subplot()
 
            self.df.boxplot(column='Price',by=text, ax=ANOVA)
            ANOVA.set_title(f" ")
            

    @ui.refreshable
    def dataExploration(self) -> None:
        text:str = self.selectedExploration.value

        if(text in ["Location", "Brand", "Model", "Car/Suv", "Title", "Engine", "ColourExtInt"]):
            ui.label("N\A")
            return

        with ui.pyplot() as fig:  ### Replace With Histogram of the data        
            plt.ylabel("Amount")
            plt.xlabel(text.capitalize())
            plt.hist(self.df[text])  

        with ui.grid(columns=3):
            ui.markdown("**Mean**")
            ui.markdown("**Median**")
            ui.markdown("**Mode**")

            if self.df[text].dtype == object:
                ui.label("N\A")
            else:
                ui.label(f"{self.df[text].mean():.2f}")

            if self.df[text].dtype == object:
                ui.label("N\A")
            else:
                ui.label(f"{self.df[text].median():.2f}")

            if self.df[text].dtype == object:
                ui.label("N\A")
            else:
                ui.label(f"{self.df[text].mode()[0]}")


    def refreshAll(self):
        self.dataExploration.refresh()
    
    # Passing the settings to machine learning file        
    def runML(self):
        selectedAlgorithm = self.regressionAlgorithm.value 
        print(f"The selected algorithm is: {selectedAlgorithm}") # Debugging
        
        selectedSplit = self.trainTestSplitSlider.value / 100
        print(f"The selected training/testing split is: {selectedSplit}") # Debugging
        
        cleanedDF = self.df
        
        mlInstance = ml.ML(cleanedDF)
        mlInstance.run(selectedAlgorithm, self.selectedPredictors, selectedSplit)
    
if __name__ in {"__main__", "__mp_main__"}:
    print("Run main.py, this won't work")
    print("Thank you")
    # Keeping this here if you really want to run this file without the dataset and watch it error out
    #gui = GUI()