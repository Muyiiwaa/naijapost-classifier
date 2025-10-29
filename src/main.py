from fastapi import FastAPI, HTTPException,status, Depends
from src.schema import HomeResponse, ModelRequest, ModelResponse, StatusResponse
from config import Settings
import logfire
from logging import getLogger
import logging
from typing import Tuple
from utils import predict
import uvicorn
import time


# Initialize settings
settings = Settings()

# Initialize logger
logfire.configure(token=settings.logfire_token)
logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


# initialize FastAPI app
app = FastAPI(
    title="Naija Text Classification API",
    description="An API for classifying Nigerian text into various categories using a pre-trained transformer model.",
    version="1.0.0"
    )

logfire.instrument_fastapi(app, capture_headers=True)

@app.get("/", response_model=HomeResponse,
status_code=status.HTTP_200_OK, summary="Home Endpoint", tags=["Home"])
async def home():
    """Home endpoint providing startup landing and basic information about the API."""
    logger.info("Home endpoint accessed")
    return HomeResponse(message="Welcome to Naija Text Classification API",
                        version="1.0.0",
                        endpoints=["/classify_text", "/status"])


@app.post("/classify_text", response_model=ModelResponse,tags=["Prediction"],
responses={200: {"description": "Successful Prediction","model": ModelResponse}})
def classify(payload: ModelRequest) -> ModelResponse:
    start_time = time.perf_counter()
    try:
        probability, predicted_class = predict(payload.text)
        output_payload = ModelResponse(text=payload.text,
                                       predicted_class=predicted_class,
                                       probability=probability)
        time_taken = time.perf_counter() - start_time
        with logfire.span("Successful Prediction"):
            logfire.info(f"{output_payload}")
            logfire.info(f"Time taken for prediction: {time_taken:.4f} seconds")

        return ModelResponse(**output_payload.model_dump())
    except Exception as err:
        logger.error(f"Error during prediction: {err}") 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Detail: {err}")
    

if __name__ == "__main__":
    uvicorn.run(app="src.main:app", host="localhost", port=8005, reload=True)