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

from scipy.stats import f_oneway

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

        app.add_static_file(local_file="Group assessment coversheet.jpg")
        ui.image("Group assessment coversheet.jpg").props(f"width=786px height=1123px").style("display: block; margin-left: auto; margin-right: auto;").force_reload()


        # region background
            
        ui.markdown("### Background")
        ui.markdown("""This project addresses the problem of determining an accurate price to market your your vehicle. Additioanlly, the project focuses on display important statistics about Australian vehicles, such as most common brand, etc.
                    The progam will need to be able to read and clean a given dataset, find any important infomation on the dataset then use the dataset to predict the price of a vehicle given a set of infomation. Online tools, such as carguides
                    [vehicle price estimator](https://www.carsguide.com.au/price), already exsist on the internet, however those tools only take the vehicles model and year into account to estimate a price for it, while this tool puts more variables into play, 
                    which can alter the price of the vehicle.
                    """)    
        # endregion    

        # region problem Statement
        ui.markdown("### Problem statement")
        ui.markdown(f"""
                    This project is based on the Price data of Australian cars, available [here](https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices), containing 16734 entries.
                    The goal for this project is to predict the vehicle price using various factors, with the price being the target variable. While the dataset gives us {len(self.df.columns)} columns,
                    we will only focus on 12. Them being: \n
                    - **Brand** (Categorical)
                    - **Year** (Quantitative)
                    - **UsedOrNew** (Qualitative)
                    - **Transmission** (Qualitative)
                    - **Engine** (Qualitative)
                    - **DriveType** (Qualitative)
                    - **FuelType** (Qualitative)
                    - **FuelConsumption** (Qualitative)
                    - **Kilometers** (Qualitative)
                    - **CylindersInEngine** (Qualitative)
                    - **Doors** (Qualitative)
                    - **Seats** (Qualitative)\n
                    The Columns we won\'t be using are: \n
                    - **Model** 
                    - **Car/Suv** 
                    - **Title**
                    - **ColorExtInt**
                    - **Location**
                    - **BodyType**\n
                    A python-based, object-oriented approach was chosen because it allows for better code organization, making this project easy to manage and easy to scale further if needed.
                    We splitted the program into multiple classes over a few files which allowed us to develop the program quickly as we could split the workload between each class. Every class had their own purpose
                    in the program, for example the gui class handled the gui while the data class handled cleaning the dataset, using this method we could edit the dataset in the data class and see the GUI update with
                    with the new changes. This meant that each group member could make edits to the program, push their changes to the public github and avoid most merge conflicts, as each member would stick to their own file.
                    This project aimed to analyse the dataset and extract any useful infomation, such as the distribution of the target variable, (vehicle price) and any interesting paatterns, like the most
                    common brand or type. The other main goal was to develop a accurate model that could estimate the price of a vehicle given a set of variables. This system focues helping sellers price their vehicle
                    effectively.
                    """)    
            
        # endregion    

        # region covering the design
        ui.markdown("### The design")
        app.add_static_file(local_file="UML.png")
        ui.image("UML.png").props(f"width=750px height=700px")
        ui.markdown("""As seen in the image provided above, object-oriented principles were used to organise the structure of the program into sections, each section would provide it's own feature to the final product.
                    The user would first run main.py and view the website, in the backend, main.py would deploy the GUI through the GUI class, the GUI class would use the Data class to fetch the dataset and clean it,
                    using the now cleaned set, the program would then chuck the new dataset into the ML class, which would attempt to create a model based off the infomation in the dataset and the predictors
                    that the user has chosen, the clean dataset would also be fed into the Analysis class which would read through the dataset and pick at any intresting infomation. To run the model, the user just needs to run application.py, they can choose to train the dataset using retrain.py, however we have already provided a trained model
                    """)
        # endregion

        # region covering the deployment
        ui.markdown("### Deployment")
        ui.markdown("""This project utilises the following packages to interpret and clean the data, run the GUI, etc.  
        - **NiceGUI**: Runs the GUI.  
        - **Pandas**: Reads the data.  
        - **PyArrow**: Dependency.  
        - **NumPy**: Used for certain mathematical operations.  
        - **SciPy**: Used to perform ANOVA analysis.  
        - **Scikit-learn**: Used for machine learning.  
        - **Matplotlib**: Used for creating diagrams.  
        - **XGBoost**: Optimised gradient boosting algorithm.  
        - **Joblib**: Used to deploy the trained model into serialized file.  
        """)
        ui.markdown("""The classes and their responsibilities can be seen below, etc.  
        - **GUI**: Runs the GUI.  
        - **Data**: Reads & Cleans the data.  
        - **ML**: Runs all the machine learning algorithms.  
        - **Analysis**: Runs any analysis on the cleaned dataset.  
        """)
        

        ui.markdown("""Throughout the project, we made sure that when we were designing a feature, we would create a completely new function to handle any logic, that way, when implementing a new feature
                            into the program, we could just call for the new function when we wanted it. This meant that when debugging code, it's easier to understand whats happening. Additionally, when merging commits, we could easily,
                            find out what lines we want to keep and what lines we want to throw away. Finally, when defining functions, we made sure to include as much infomation as possible, such as required input/output types and a brief description of what the function does.""")
        app.add_static_file(local_file="BestPracticExample.png")
        ui.image("BestPracticExample.png").props(f"width=770px height=300px")
        # endregion
        ui.markdown("### Testing")
        ui.markdown("""
                    We did a few unit tests on our code, using selenium we could play with the inputs and see how they interacted with our GUI, checking input validation and whatnot.

                    Throughout the unit testing, we found no bugs.
                    
                    """)
        
        ui.markdown("### Intergration")
        ui.markdown("""
                    Intergrating the modules of the program together was pretty easy, considering that we followed an OOP structure; intergrating each part into the GUI was just calling a function or class.
                    There was a time when one group member was training a model and a bug that seemed to come out of nowhere crashed the model during training. This bug ended up happening because the data
                    in one of the colums wasn't cleaned correctly, leaving some weird data. This was the only time our integration technique worked against us, as the person experiencing the bug didn't know that
                    it was the reason for the bug, as the bug was in a completely different file. Luckly though, fixing said bug was much easier because the code was split up, so tracking down a subprocess was simple.
                    """)
        app.add_static_file(local_file="BugImage.png")
        ui.image("BugImage.png").props(f"width=635px height=397px")
        ui.markdown("""We didn't use any third party API in our system as we believe everything should be done in-house so we can show our skills and knowledge in the relevant fields.
                    """)
                    
        
        
        
        ui.markdown("### Refrences")
        ui.markdown("""
                    **Dataset Used for model:** \n
                        Nelgiri Yewithana. (n.d.). Australian Vehicle Prices. Kaggle. https://www.kaggle.com/datasets/nelgiriyewithana/australian-vehicle-prices/data \n
                    **Links in website:** \n
                        Beers, B. (2024) P-value: What it is, how to calculate it, and why it matters, Investopedia. Available at: https://www.investopedia.com/terms/p/p-value.asp (Accessed: 23 October 2024). 

                    
                    """)

        # region reading the data
        ui.markdown("#### Reading the data")
        ui.markdown("Reading the data is a fundamental process in the program. below is a \
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
        ui.markdown("""
                Reading and cleaning the data was handled in the Data class, the program will first call this class to fetch and clean the dataset, with this infomation
                the program can run further analysis or generate a model based off the predictors the user chooses.
                    """)
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

        ui.markdown("> #### Observations\n A few of the colums are int32 or float64 these are our numbers, or quantitative colums. \
                    There are also colums that have a type 'category' these are our catagorical types.")
                
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
        ui.markdown("> We have decided that we will remove our outliers prior to where it was expected, as our large dataset does have quite a few\
                     outliers. However, with the above Radio you can change how those outliers are removed.")

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
                    > - Year (Continuous)\n  \
                    > + UsedOrNew (Categorical)\n\
                    > + Transmission (Categorical)\n\
                    > + Engine (Categorical)\n\
                    > + DriveType(Categorical)\n\
                    > + FuelType(Categorical)\n\
                    > + FuelConsumption(Continuous)\n\
                    > + Kilometers (Continuous)\n\
                    > + CylindersInEngine (Continuous)\n\
                    > + Doors (Continuous)\n\
                    > + Seats (Continuous)")
        
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
        ui.markdown("> Some of the columns are not available as there are too any unique entries to graph them.   Rest assured they are calculated nevertheless.    \n\
                    We are looking for Graphs where as the categorical variable (X-axis) changes Price (Y-axis) also changes, this shows that\
                    they are related.")
        
        # Defining a function to find the statistical relationship with all the categorical variables
        def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):
            # Creating an empty list of final selected predictors
            SelectedPredictors=[]

            for predictor in CategoricalPredictorList:
                CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)
                AnovaResults = f_oneway(*CategoryGroupLists)

                # If the ANOVA P-Value is <0.05, that means we reject H0
                if (AnovaResults[1] < 0.05):
                    SelectedPredictors.append([predictor,AnovaResults[1]])


            return(SelectedPredictors)
        
        CategoricalPredictorList=list(self.df.loc[:, self.df.columns != 'Price'].select_dtypes(exclude=np.number))

        temp = FunctionAnova(inpData=self.df,
              TargetVariable='Price',
              CategoricalPredictorList=CategoricalPredictorList)
        
        ui.markdown("##### Results of ANOVA testing")
        with ui.list() as table:
            for cat in temp:
                ui.item(f"Predictor Name: {cat[0]}   P-Value: {cat[1]}")

        ui.markdown("##### Observations")
        ui.markdown("> Only predictors with less than a 0.05 P-value are chosen. A p-value, or probability value, is a number describing the likelihood of obtaining the observed data under the null hypothesis of a statistical test\
                    [Source](https://www.investopedia.com/terms/p/p-value.asp). These predictors are: UsedOrNew, Transmission, DriveType, FuelType, and BodyType.")
            
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
        
        #ui.markdown("#### Regression algorithms")
        #listofRegressionAlgorithms = ["Linear", "Decision Tree", "Random forest", "Adaboost", "XGBoost", "K-Nearest neighbour", "SVR"]
        #self.regressionAlgorithm = ui.radio(listofRegressionAlgorithms, value="Linear")   ### Make functional
        
        # endregion
        
        ui.markdown("#### Launch")
        ui.button("Start!", on_click=lambda: self.runML.refresh(type=0))

        self.runML(type=1)
        
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

            if self.df[text].dtype == 'category':
                ui.label("N\A")
                ui.label("N\A")
                ui.label("N\A")
                return
            else:
                ui.label(f"{self.df[text].mean():.2f}")
                ui.label(f"{self.df[text].median():.2f}")
                ui.label(f"{self.df[text].mode()[0]}")


    def refreshAll(self):
        self.dataExploration.refresh()
    
    # Passing the settings to machine learning file  
    @ui.refreshable    
    def runML(self, type=0):
        if type==1:
                ui.markdown("Please Click Start")
                return
        selectedSplit = self.trainTestSplitSlider.value / 100
        #print(f"The selected training/testing split is: {selectedSplit}") # Debugging
        
        cleanedDF = self.df
        
        mlInstance = ml.ML(cleanedDF)
        results, best = mlInstance.runAllAlgorithm(self.selectedPredictors, selectedSplit)

        ui.markdown("##### Results")
        ui.markdown(f"> The best performing algorithm was: {best.capitalize()}    \n\
                        It had a R^2 value of {results[best][0]:.5f}   \n\
                        It had a MAPE of {results[best][2]:.5f}%")
        for algo in results:
            if algo is not best:
                ui.markdown(f"> Algorithm: {algo}    \n\
                            R^2 value: {results[algo][0]:.5f}   \n\
                            MAPE: {results[algo][2]:.5f}%")
            

if __name__ in {"__main__", "__mp_main__"}:
    print("Run main.py, this won't work")
    print("Thank you")
    # Keeping this here if you really want to run this file without the dataset and watch it error out
    #gui = GUI()