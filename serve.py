import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from predict import load_model, predict_demand

app = FastAPI(
    title="Bike Sharing Hourly Demand Prediction API",
    description="Predicts hourly bike rental demand (cnt) using a tuned lightGBM pipeline.",
    version="1.0.0",
)

model = load_model()


class BikeRentalRequest(BaseModel):
    season: int = Field(3, ge=1, le=4, description="1: Winter, 2: Spring, 3: Summer, 4: Fall")
    yr: int = Field(1, ge=0, le=1, description="0: 2011, 1: 2012")
    mnth: int = Field(8, ge=1, le=12, description="Month (1 to 12)")
    hr: int = Field(17, ge=0, le=23, description="Hour of day (0 to 23)")
    holiday: int = Field(0, ge=0, le=1, description="1 if holiday else 0")
    weekday: int = Field(4, ge=0, le=6, description="Day of week (0=Sunday to 6=Saturday)")
    workingday: int = Field(1, ge=0, le=1, description="1 if neither weekend nor holiday else 0")
    weathersit: int = Field(1, ge=1, le=4, description="1: Clear, 2: Mist, 3: Light Snow/Rain, 4: Heavy Rain/Ice")
    temp: float = Field(0.76, ge=0.0, le=1.0, description="Normalized temperature in Celsius")
    atemp: float = Field(0.70, ge=0.0, le=1.0, description="Normalized feeling temperature")
    hum: float = Field(0.50, ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(0.19, ge=0.0, le=1.0, description="Normalized wind speed")


class BikeRentalResponse(BaseModel):
    predicted_rentals: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=BikeRentalResponse)
def predict(request: BikeRentalRequest):
    pred = predict_demand(request.model_dump(), model=model)
    return BikeRentalResponse(predicted_rentals=pred)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)