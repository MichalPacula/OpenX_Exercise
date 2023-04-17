from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import dill
from tensorflow import keras
import pandas as pd

#creating app
app = FastAPI()

#creating class, which have model and all features
class ScoringItem(BaseModel):
    Model: str
    F1: int
    F2: int
    F3: int
    F4: int
    F5: int
    F6: int
    F7: int
    F8: int
    F9: int
    F10: int
    F11: int
    F12: int
    F13: int
    F14: int
    F15: int
    F16: int
    F17: int
    F18: int
    F19: int
    F20: int
    F21: int
    F22: int
    F23: int
    F24: int
    F25: int
    F26: int
    F27: int
    F28: int
    F29: int
    F30: int
    F31: int
    F32: int
    F33: int
    F34: int
    F35: int
    F36: int
    F37: int
    F38: int
    F39: int
    F40: int
    F41: int
    F42: int
    F43: int
    F44: int
    F45: int
    F46: int
    F47: int
    F48: int
    F49: int
    F50: int
    F51: int
    F52: int
    F53: int

#loading random forest classifier model
with open("modelRFC.pickle", "rb") as f:
    modelRFC = pickle.load(f)

#loading k-nearest neighbors model
with open("modelKNN.pickle", "rb") as f:
    modelKNN = pickle.load(f)

#loading heuristic algorithm
with open("heuristic.pickle", "rb") as f:
    heuristic = dill.load(f)

#loading neural network model
modelNN = keras.models.load_model("modelNN")

#creating post method, which sends model name and features to api
@app.post("/")
async def ScoringEndpoint(item:ScoringItem):

    #getting item to dictionary
    itemDict = item.dict()

    #checking what model is in item and predicting features in item in chosen model
    if itemDict["Model"] == "modelRFC":
        itemDict.pop("Model")

        df = pd.DataFrame([itemDict.values()], columns=itemDict.keys())
        yPredicted = modelRFC.predict(df)

        return {"prediction": int(yPredicted)}
    elif itemDict["Model"] == "modelKNN":
        itemDict.pop("Model")

        df = pd.DataFrame([itemDict.values()], columns=itemDict.keys())
        yPredicted = modelKNN.predict(df)

        return {"prediction": int(yPredicted)}
    elif itemDict["Model"] == "heuristic":
        itemDict.pop("Model")

        df = pd.DataFrame([itemDict.values()], columns=itemDict.keys())
        yPredicted = heuristic(df)

        return {"prediction": int(yPredicted)}
    elif itemDict["Model"] == "modelNN":
        itemDict.pop("Model")

        df = pd.DataFrame([itemDict.values()], columns=itemDict.keys())
        yProb = modelNN.predict(df)
        yPredicted = yProb.argmax(axis=-1)
        
        return {"prediction": int(yPredicted)}