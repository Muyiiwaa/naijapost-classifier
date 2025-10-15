from fastapi import FastAPI, HTTPException,status, Depends
from schema import HomeResponse, ModelRequest, ModelResponse, StatusResponse
import logfire
from logging import getLogger
from typing import Tuple
from config import Settings
from utils import (
    get_device,load_model_and_tokenizer,
    preprocess_texts,make_prediction)



# Initialize logger
logger = getLogger(__name__)
logfire.basic_config(level="INFO", fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# initialize settings
settings = Settings()

def get_model_and_tokenizer():
    """Dependency to get the model and tokenizer."""
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"  # Example model
    num_labels = 10  # Example number of labels
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(model_name, num_labels)
    return model, tokenizer, device


model, tokenizer, device = get_model_and_tokenizer(model_name=settings.model_name)


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
                        endpoints=["/predict", "/status"])


@app.post("/predict", response_model=ModelResponse,tags=["Prediction"],
responses={200: {"description": "Successful Prediction","model": ModelResponse}})
def predict(payload: ModelRequest) -> ModelResponse:
    try:
        encoding = preprocess_texts()