import pandas as pd
from fastapi import FastAPI
from src.diamonds.registry import load_model
from .data_model import Diamond, DiamondResponse

app = FastAPI(
    title="Diamond API",
    description="api to get diamond price",
    version="0.1.0"
)

model = load_model("model", "local")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/price")
def predict_price(diamond: Diamond):
    diamond_df = pd.Dataframe([diamond.model_dump()])
    prediction = model.predict(diamond_df)
    diamond_reponse = DiamondResponse(**diamond.model_dump(), price=prediction[0])
    return diamond_reponse