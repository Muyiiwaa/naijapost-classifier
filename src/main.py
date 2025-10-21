from fastapi import FastAPI, HTTPException,status, Depends
from src.schema import HomeResponse, ModelRequest, ModelResponse, StatusResponse
import logfire
from logging import getLogger
import logging
from typing import Tuple
from utils import predict


# Initialize logger
logging.basicConfig(level="INFO", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)


# initialize FastAPI app
app = FastAPI(
    title="Naija Text Classification API",
    description="An API for classifying Nigerian text into various categories using a pre-trained transformer model.",
    version="1.0.0"
    )

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
    try:
        probability, predicted_class = predict(payload.text)
        return ModelResponse(text=payload.text,predicted_class=predicted_class,
                             probability=probability)
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Detail: {err}")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app="src.main:app", host="localhost", port=8005, reload=True)