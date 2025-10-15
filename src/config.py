from pydantic_settings import BaseSettings,SettingsConfigDict
from typing import List



class Settings(BaseSettings):
    hugging_face_api_key: str 
    model_name: str 
    num_labels: int
    categories: List[str]

    model_config = SettingsConfigDict(  
        env_file = ".env",
        env_file_encoding = "utf-8")


if __name__ == "__main__":
    settings = Settings()
    print(settings.categories)
    print(settings.num_labels)