import tkinter as tk
import numpy as np
import joblib 
from tkinter import messagebox

# Load the model and data map
model = joblib.load('XGBOOSTMODEL.pkl')    
valueMap = joblib.load('DATAMAP.pkl')       

#print(valueMap['ColourExtInt'])

def encodeCategorical(feature, value):
    '''
    Encode categorical values based on mappings
    
    Params:
    features (str): The name of the features 
    value (str): The categorical value
    
    Returns:
    int: Encoded value for the categorical features
    '''
    
    value = value.lower().replace('/', ' / ')  
    if feature in valueMap:
        for code, category in valueMap[feature].items():
            if category.lower() == value:  
                return code
        raise ValueError(f"Value '{value}' not found in mappings for feature '{feature}'.")
    else:
        raise ValueError(f"No mapping found for feature '{feature}'.")

def predictPrice(inputData):
    '''
    Take the input and returns the prediction
    '''
    
    inputdataArray = np.array([inputData])
    predictedPrice = model.predict(inputdataArray)
    return predictedPrice[0]

def predictFromGui():
    try:
        # Dynamically encode categorical input based on mappings
        brand = encodeCategorical('Brand', entryBrand.get())
        model = encodeCategorical('Model', entryModel.get())  
        title = encodeCategorical('Title', entryTitle.get())  
        carSuv = encodeCategorical('Car/Suv', entryCarSuv.get())  
        colourExInt = encodeCategorical('ColourExtInt', entryColourExInt.get())  
        location = encodeCategorical('Location', entryLocation.get())  
        bodyType = encodeCategorical('BodyType', entryBodyType.get())  
        year = int(entryYear.get())
        usedOrNew = encodeCategorical('UsedOrNew', entryUsedOrNew.get())
        transmission = encodeCategorical('Transmission', entryTransmission.get())
        driveType = encodeCategorical('DriveType', entryDriveType.get())
        fuelType = encodeCategorical('FuelType', entryFuelType.get())
        fuelConsumption = float(entryFuelConsumption.get())
        kilometres = float(entryKilometres.get())
        cylindersInEngine = int(entryCylindersInEngine.get())  
        engineSize = float(entryEngineSize.get())  

        doors = int(entryDoors.get())
        seats = int(entrySeats.get())
        
        inputData = [
            brand, model, title, carSuv, colourExInt, location, bodyType, 
            year, usedOrNew, transmission, driveType, fuelType, 
            fuelConsumption, kilometres, cylindersInEngine, engineSize, doors, seats
        ]
        
        # Predict
        predictedPrice = predictPrice(inputData)
        
        # Display the predicted price
        messagebox.showinfo("Prediction", f"Predicted Price: ${predictedPrice:.2f}")
    
    except ValueError as e:
        messagebox.showerror("Invalid input", str(e))

# Create the Tkinter window
window = tk.Tk()
window.title("Vehicle Price Predictor")

# Create the input fields
tk.Label(window, text="Brand:").grid(row=0)
entryBrand = tk.Entry(window)
entryBrand.grid(row=0, column=1)

tk.Label(window, text="Model:").grid(row=1)  
entryModel = tk.Entry(window)
entryModel.grid(row=1, column=1)

tk.Label(window, text="Title:").grid(row=2)  
entryTitle = tk.Entry(window)
entryTitle.grid(row=2, column=1)

tk.Label(window, text="Car/Suv:").grid(row=3)  
entryCarSuv = tk.Entry(window)
entryCarSuv.grid(row=3, column=1)

tk.Label(window, text="Colour Exterior/Interior:").grid(row=4)  
entryColourExInt = tk.Entry(window)
entryColourExInt.grid(row=4, column=1)

tk.Label(window, text="Location:").grid(row=5)  
entryLocation = tk.Entry(window)
entryLocation.grid(row=5, column=1)

tk.Label(window, text="Body Type:").grid(row=6)  
entryBodyType = tk.Entry(window)
entryBodyType.grid(row=6, column=1)

tk.Label(window, text="Year:").grid(row=7)
entryYear = tk.Entry(window)
entryYear.grid(row=7, column=1)

tk.Label(window, text="Used or New:").grid(row=8)
entryUsedOrNew = tk.Entry(window)
entryUsedOrNew.grid(row=8, column=1)

tk.Label(window, text="Transmission:").grid(row=9)
entryTransmission = tk.Entry(window)
entryTransmission.grid(row=9, column=1)

tk.Label(window, text="Drive Type:").grid(row=10)
entryDriveType = tk.Entry(window)
entryDriveType.grid(row=10, column=1)

tk.Label(window, text="Fuel Type:").grid(row=11)
entryFuelType = tk.Entry(window)
entryFuelType.grid(row=11, column=1)

tk.Label(window, text="Fuel Consumption:").grid(row=12)
entryFuelConsumption = tk.Entry(window)
entryFuelConsumption.grid(row=12, column=1)

tk.Label(window, text="Kilometres:").grid(row=13)
entryKilometres = tk.Entry(window)
entryKilometres.grid(row=13, column=1)

# Split Engine input into two fields: Cylinders in Engine and Engine Size
tk.Label(window, text="Cylinders in Engine:").grid(row=14)
entryCylindersInEngine = tk.Entry(window)
entryCylindersInEngine.grid(row=14, column=1)

tk.Label(window, text="Engine Size (in L):").grid(row=15)
entryEngineSize = tk.Entry(window)
entryEngineSize.grid(row=15, column=1)

tk.Label(window, text="Doors:").grid(row=16)
entryDoors = tk.Entry(window)
entryDoors.grid(row=16, column=1)

tk.Label(window, text="Seats:").grid(row=17)
entrySeats = tk.Entry(window)
entrySeats.grid(row=17, column=1)

# Button to predict
tk.Button(window, text="Predict Price", command=predictFromGui).grid(row=18, columnspan=2)

# Start the GUI loop
window.mainloop()
