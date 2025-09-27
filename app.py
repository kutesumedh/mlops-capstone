import logging
from fastapi import FastAPI
from pydantic import BaseModel
import joblib


# Configure logging to write to logs.txt
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Load model
model = joblib.load("model.pkl")


app = FastAPI(title="ML Model API with FastAPI")


class InputData(BaseModel):
    area: float


@app.post("/predict")
def predict(data: InputData):
    """Predict using the loaded ML model based on area input."""
    try:
        # Example validation: area must be positive
        if data.area <= 0:
            raise ValueError("Area must be positive")

        # Make prediction
        prediction = model.predict([[data.area]])[0]

        # Log success
        logging.info(
            f"Prediction successful: input={data.area}, output={prediction}"
        )

        return {"prediction": float(prediction)}

    except Exception as e:
        # Log error
        logging.error(f"Prediction failed: input={data.area}, error={str(e)}")
        return {"error": str(e)}
