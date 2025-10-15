from pydantic import BaseModel, Field
from typing import List,Literal



class HomeResponse(BaseModel):
    message: str = "Welcome to Naija Text Classification API"
    version: str = "1.0.0"
    endpoints: List[str] = Field(
        default=[
            "/predict",
            "/status"]
    )

class ModelRequest(BaseModel):
    text: str = Field(..., examples=["I love programming in Python it is so much fun!",
    "BurnaBoy is such an amazing artist!"],
    description="The text to be classified.")


class ModelResponse(BaseModel):
    text: str = Field(..., description="The input text that was classified.",
                      examples=["I love programming in Python it is so much fun!"])
    predicted_class: Literal["Politics", "Sports", "Entertainment", "Lifestyle", "Technology","Relationship", "Business", "Education", "Religion", "Health"] = Field(..., description="The predicted class label for the input text.", examples=["Sports", "Politics", "Entertainment", "Business"])
    probability: float = Field(..., description="The probability of the predicted class.",
    ge=0.0, le=1.0, examples=[0.95, 0.87])

class StatusResponse(BaseModel):
    status: Literal["ok"] = Field(..., description="The status of the API.")
    num_labels: int = Field(..., description="The number of labels the model can predict.", example=4)
    device: str = Field(..., description="The device the model is running on (CPU or GPU).", examples=["cuda","cpu"])