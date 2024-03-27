from config.logging_config import info_log
import ml.model as model
import ml.data as data

from pydantic import BaseModel, Field
from fastapi import FastAPI
import pandas as pd
import pickle


app = FastAPI()

# Load only once per boot.
with open("artifacts/regressor_model.pkl", 'rb') as f:
    trained_model = pickle.load(f)

with open("artifacts/encoder.pkl", 'rb') as f:
    encoder = pickle.load(f)


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    class Config:
        allow_population_by_field_name = True


@app.get("/")
def root():
    info_log("Root path endpoint called.")
    return {"message": "Greetings. Please refer to /docs for OpenAPI specs."}


@app.post("/inference")
def inference(request_data: InputData):
    info_log("Inference endpoint called.")

    input_data = request_data.dict(by_alias=True)

    df = pd.DataFrame([input_data])

    encoded_data, _, _, _ = data.process_data(
        df, categorical_features=model.CAT_FEATURES, training=False, encoder=encoder)

    label = model.inference(trained_model, encoded_data)[0]

    return {"inference_result": str(label)}
