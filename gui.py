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

import matplotlib.pyplot as plt


class GUI():
    def __init__(self, data) -> None:
        df = data.getData()
        
        with ui.header():
            ui.markdown("# **Software Technology Group 42**")
            
        ui.markdown("### Problem statement")
        ui.markdown(f"We are trying to predict the price of cars. There are many different variables in this problem, the dependent one is the price of the car. Our dataset has \
                     {len(df.columns)} other possible colums for independent variables however we will only be using {'INSERHERE'} of them these are: {'INSERHERE'}.")    
            
        ui.markdown("#### Columns and Types")
        with ui.scroll_area().classes('w-50 h-100 border'):
            with ui.grid(columns=2).props('dense separator'):
                cols = df.dtypes.to_string().split("\n")
                for col in cols:
                    colName, colType = col.split()
                    ui.label(colName + ": ")
                    ui.label(colType).props('inline font-size=100')
                
        ui.markdown("#### Distribution of Target variable")
        targetCol = "Price"
        with ui.pyplot() as fig:  ### Replace With Histogram of the data
            plt.title(targetCol.capitalize())         
            plt.ylabel("Amount")
            plt.xlabel("Price in $AUD")

            plt.hist(df[targetCol])         
                
        ui.markdown("#### Data exploration")### Replace with segment to do basic exploration of data
        selectedExploration = ui.select(list(df.columns), value="Price",on_change= lambda: dataExploration.refresh(selectedExploration.value))

        @ui.refreshable
        def dataExploration(text = "Price") -> None:
            if(text in ["Location", "Brand", "Model", "Car/Suv", "Title", "Engine", "ColourExtInt"]):
                ui.label("N\A")
                return

            with ui.pyplot() as fig:  ### Replace With Histogram of the data
                plt.title(text.capitalize())         
                plt.ylabel("Amount")
                plt.xlabel("Price in $AUD")
                plt.hist(df[text])  

            with ui.grid(columns=3):
                ui.markdown("**Mean**")
                ui.markdown("**Median**")
                ui.markdown("**Mode**")
                

                if df[text].dtype == object:
                    ui.label("N\A")
                else:
                    ui.label(f"{df[text].mean():.2f}")

                if df[text].dtype == object:
                    ui.label("N\A")
                else:
                    ui.label(f"{df[text].median():.2f}")

                if df[text].dtype == object:
                    ui.label("N\A")
                else:
                    ui.label(f"{df[text].mode()[0]}")

            
        dataExploration()
        
        ui.markdown("#### Visual Exploratory Data Analysis")
        ui.skeleton().classes('w-full')        ### Replace with exploration of data but using graphs and shit
        
        ui.markdown("#### Outlier analysis")
        ui.skeleton().classes('w-full')        ### Find outliers
        
        ui.markdown("#### Missing values")
        missingValuesRadio = ui.radio(["Delete Rows", "Median Value", "Mode Value", "Interpolate"], value="Delete Rows")   ### Make functional
        
        ui.markdown("#### Visual and statistic correlation analysis")
        ui.skeleton().classes('w-full')      ### Cause our inputs are continuous we can make correlation analysis (r^2)
        
        ui.markdown("#### Statistical feature selection (categorical vs. continuous) using ANOVA test")
        ui.skeleton().classes('w-full')      ### idk go figure out yourself  its step 9
        
        ui.markdown("####  Final predictors/features")
        with ui.list():                      ### make updates to this reflect in code
            cols = df.dtypes.to_string().split("\n")
            for col in cols:
                colName, _ = col.split()
                ui.checkbox(colName)
                ui.space() 
        
        ui.markdown("#### Train/test data split")
        trainTestSplitSlider = ui.slider(min=20,max=80, value=80) ### Make functional
        ui.label().bind_text_from(trainTestSplitSlider, 'value', backward=lambda v: f"Train {v}%, Test {100-v}%") 
        
        ui.markdown("#### Regression algorithms")
        listofRegressionAlgorithms = ["Linear", "Decision tree", "Random forest", "Adaboost", "XGBoost", "K-Nearest neighbour", "SVM"]
        regressionAlgorithm = ui.radio(listofRegressionAlgorithms, value="Linear")   ### Make functional
        
        ui.markdown("#### Launch")
        ui.button("Start!", on_click=exit)
        
        ui.image()
        ui.run() 
    

if __name__ in {"__main__", "__mp_main__"}:
    print("Run main.py, this won't work")
    print("Thank you")
    # Keeping this here if you really want to run this file without the dataset and watch it error out
    #gui = GUI()